#!/usr/bin/env python3
"""
head_server.py — Sub-stage 3.2: Asynchronous GPU Head Worker Microservice
Runs on Port 8001.
Hosts ViTHead (Patch Conv Projection + CLS Token + Pos Embeddings + Blocks 0..5)
pinned to GPU (cuda:0 if available, else CPU fallback).

Endpoints:
  - GET  /health: Returns worker status and active hardware device.
  - POST /infer_head:
      * Accepts raw image tensor (B, 3, 224, 224).
      * Supports query params:
          - compress_fp16: bool (toggles FP16 activation quantization).
          - link_mbps: float (emulates wireless network bandwidth delay: Delta_t = Bytes * 8 / (Mbps * 1e6)).
      * Computes ViTHead forward pass on GPU.
      * Serializes intermediate activation (B, 197, 768).
      * Emulates bandwidth transfer delay.
      * Dispatches serialized payload to Tail Worker (Port 8002) via persistent async HTTP client.
      * Merges and returns full end-to-end pipeline telemetry and predictions.
"""

import asyncio
from contextlib import asynccontextmanager
import os
import time
from typing import Dict, Optional
from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse
import httpx
import torch
import uvicorn

from model_slices import ViTHead, create_vit_slices
from serialization import deserialize_tensor, serialize_tensor

# Configuration
TAIL_WORKER_URL = os.getenv("TAIL_WORKER_URL", "http://127.0.0.1:8002")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Global instances managed across application lifecycle
head_model: Optional[ViTHead] = None
http_client: Optional[httpx.AsyncClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager:
      1. Loads ViTHead onto target compute device (GPU cuda:0 / CPU fallback).
      2. Initializes shared, persistent httpx.AsyncClient with connection pooling.
      3. Cleanly releases resources on shutdown.
    """
    global head_model, http_client

    device_name = torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "Host CPU"
    print(f"[HEAD SERVER] Initializing ViTHead worker pinned to device: '{DEVICE}' ({device_name})...")

    try:
        # Load model slice (offline safe: weights=None)
        head, _, _ = create_vit_slices(weights=None)
        head_model = head.to(DEVICE).eval()
        print(f"[HEAD SERVER] ViTHead initialized successfully on {DEVICE}.")
    except Exception as e:
        print(f"[HEAD SERVER ERROR] Failed to initialize ViTHead: {e}")
        raise e

    # Initialize persistent connection pool to downstream workers
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(60.0, connect=5.0),
        limits=httpx.Limits(max_keepalive_connections=50, max_connections=100),
    )
    print(f"[HEAD SERVER] HTTP client pool connected. Target Tail URL: {TAIL_WORKER_URL}")

    yield

    print("[HEAD SERVER] Shutting down Head Worker microservice...")
    if http_client:
        await http_client.aclose()
    head_model = None


app = FastAPI(
    title="SLO-Aware Inference Pipeline - Head Worker",
    description="GPU Head Worker (Patch Embeddings + Blocks 0-5) on Port 8001",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check() -> Dict[str, str]:
    """Health check endpoint confirming worker status and hardware device."""
    return {
        "status": "ok",
        "worker": "head",
        "device": str(DEVICE),
    }


@app.post("/infer_head", status_code=status.HTTP_200_OK)
async def infer_head(
    request: Request,
    compress_fp16: bool = Query(default=False, description="Enable FP16 activation quantization"),
    link_mbps: float = Query(default=0.0, description="Emulated network bandwidth link speed in Mbps (0 for none)"),
) -> JSONResponse:
    """
    Stage 1 Inference Endpoint.
    Accepts raw image bytes or generates random test tensor if empty.
    Executes ViTHead, compresses activations, emulates network transmission, and forwards to Tail Worker.
    """
    global head_model, http_client
    if head_model is None or http_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Head Worker service not initialized.",
        )

    t_total_start = time.perf_counter()

    # 1. Parse / Deserialize incoming image tensor
    body_bytes = await request.body()
    if body_bytes and len(body_bytes) > 0:
        try:
            image_tensor = deserialize_tensor(body_bytes, device="cpu", upcast_fp32=True)
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to deserialize input image tensor.",
            )
    else:
        # Default fallback: generate standard batch of 1 sample (1, 3, 224, 224)
        torch.manual_seed(42)
        image_tensor = torch.randn(1, 3, 224, 224, dtype=torch.float32)

    # Validate image dimensions
    if image_tensor.dim() != 4 or image_tensor.shape[1] != 3 or image_tensor.shape[2] != 224 or image_tensor.shape[3] != 224:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Expected 4D image tensor (B, 3, 224, 224), got {list(image_tensor.shape)}",
        )

    batch_size = image_tensor.shape[0]

    # 2. Stage 1 Forward Computation on GPU
    t_w1_start = time.perf_counter()
    with torch.inference_mode():
        activations = head_model(image_tensor.to(DEVICE))
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    w1_compute_ms = (time.perf_counter() - t_w1_start) * 1000.0

    # 3. Intermediate Binary Serialization (with optional FP16 quantization)
    t_ser_start = time.perf_counter()
    activation_bytes = serialize_tensor(activations, compress_fp16=compress_fp16)
    serialization_bytes = len(activation_bytes)
    ser_time_ms = (time.perf_counter() - t_ser_start) * 1000.0

    # 4. Configurable Network Bandwidth Delay Emulation
    # Formula: Delta_t = (Bytes * 8) / (Bandwidth_Mbps * 1e6)
    if link_mbps > 0.0:
        payload_bits = serialization_bytes * 8.0
        emulated_delay_sec = payload_bits / (link_mbps * 1_000_000.0)
        emulated_delay_ms = emulated_delay_sec * 1000.0
        await asyncio.sleep(emulated_delay_sec)
    else:
        emulated_delay_ms = 0.0

    # 5. Asynchronous Forwarding to Tail Worker (Port 8002)
    tail_endpoint = f"{TAIL_WORKER_URL}/infer_tail"
    try:
        tail_response = await http_client.post(
            tail_endpoint,
            content=activation_bytes,
            headers={
                "Content-Type": "application/octet-stream",
                "x-compress-fp16": str(compress_fp16).lower(),
            },
        )
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Downstream Tail Worker unreachable at '{tail_endpoint}': {str(exc)}",
        )

    if tail_response.status_code != 200:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Tail Worker error [{tail_response.status_code}]: {tail_response.text}",
        )

    tail_data = tail_response.json()
    total_pipeline_ms = (time.perf_counter() - t_total_start) * 1000.0

    # 6. Assemble Combined Telemetry Response
    return JSONResponse(
        content={
            "batch_size": batch_size,
            "predictions": tail_data.get("predictions", []),
            "telemetry": {
                "worker1_compute_ms": round(w1_compute_ms, 2),
                "serialization_time_ms": round(ser_time_ms, 2),
                "serialization_bytes": serialization_bytes,
                "emulated_link_mbps": link_mbps,
                "emulated_network_delay_ms": round(emulated_delay_ms, 2),
                "worker2_compute_ms": tail_data.get("worker2_compute_ms", 0.0),
                "total_pipeline_ms": round(total_pipeline_ms, 2),
            },
        }
    )


if __name__ == "__main__":
    print(f"[INFO] Starting Head Worker on http://127.0.0.1:8001 ({DEVICE})...")
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="info")
