#!/usr/bin/env python3
"""
tail_server.py — Sub-stage 3.1: Asynchronous CPU Tail Worker Microservice
Runs on Port 8002.
Hosts ViTTail (Transformer Encoder Blocks 6..11 + LayerNorm + Classification Head)
strictly pinned to the host multi-core CPU.

Endpoints:
  - GET  /health: Health check returning worker metadata and device placement.
  - POST /infer_tail: Receives raw binary intermediate activation stream, deserializes
                     (with automatic FP32 upcasting if FP16), runs ViTTail forward pass,
                     and returns top-5 predictions with stage compute latency.
"""

from contextlib import asynccontextmanager
import time
from typing import Dict, List, Optional
from fastapi import FastAPI, Header, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
import torch
import uvicorn

from model_slices import ViTTail, create_vit_slices
from serialization import deserialize_tensor

# Global reference to CPU model slice
tail_model: Optional[ViTTail] = None
DEVICE = "cpu"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager that loads ViTTail onto host CPU on server startup
    and releases resources on shutdown.
    """
    global tail_model
    print(f"[TAIL SERVER] Initializing ViTTail worker pinned strictly to device: '{DEVICE}'...")
    try:
        # Load model slice (offline safe: weights=None if offline, or weights='DEFAULT')
        _, tail, _ = create_vit_slices(weights=None)
        tail_model = tail.to(DEVICE).eval()
        print("[TAIL SERVER] ViTTail initialized successfully and set to eval mode.")
    except Exception as e:
        print(f"[TAIL SERVER ERROR] Failed to initialize ViTTail: {e}")
        raise e

    yield

    print("[TAIL SERVER] Shutting down Tail Worker microservice.")
    tail_model = None


app = FastAPI(
    title="SLO-Aware Inference Pipeline - Tail Worker",
    description="CPU Tail Worker (Blocks 6-11 + Classification Head) on Port 8002",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check() -> Dict[str, str]:
    """Health check endpoint confirming worker status and CPU pinning."""
    return {
        "status": "ok",
        "worker": "tail",
        "device": DEVICE,
    }


@app.post("/infer_tail", status_code=status.HTTP_200_OK)
async def infer_tail(
    request: Request,
    x_compress_fp16: Optional[str] = Header(default="false"),
) -> JSONResponse:
    """
    Inference endpoint for Stage 2 (Tail).
    Consumes raw binary bytes containing serialized activation tensors from Head worker.
    """
    global tail_model
    if tail_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Tail model not initialized.",
        )

    # 1. Read raw binary body directly from request stream (zero disk I/O)
    body_bytes = await request.body()
    if not body_bytes or len(body_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Received empty activation byte payload.",
        )

    # 2. Deserialize binary stream with automatic FP32 upcast if FP16 was transmitted
    try:
        activation_tensor = deserialize_tensor(
            data=body_bytes,
            device=DEVICE,
            upcast_fp32=True,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Tensor deserialization failed: {str(e)}",
        )

    # 3. Shape validation (B, 197, 768)
    if activation_tensor.dim() != 3 or activation_tensor.shape[1] != 197 or activation_tensor.shape[2] != 768:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid activation shape: expected (B, 197, 768), got {list(activation_tensor.shape)}",
        )

    batch_size = activation_tensor.shape[0]

    # 4. Forward computation through ViTTail on CPU
    t_start = time.perf_counter()
    with torch.inference_mode():
        logits = tail_model(activation_tensor)
        probabilities = torch.softmax(logits, dim=-1)
        top5_probs, top5_classes = torch.topk(probabilities, k=5, dim=-1)
    compute_ms = (time.perf_counter() - t_start) * 1000.0

    # 5. Format top-5 predictions per batch element
    predictions: List[Dict] = []
    for i in range(batch_size):
        predictions.append({
            "top5_classes": top5_classes[i].tolist(),
            "top5_probs": [round(p, 6) for p in top5_probs[i].tolist()],
        })

    return JSONResponse(
        content={
            "batch_size": batch_size,
            "predictions": predictions,
            "worker2_compute_ms": round(compute_ms, 2),
        }
    )


if __name__ == "__main__":
    print("[INFO] Starting Tail Worker on http://127.0.0.1:8002 (CPU)...")
    uvicorn.run(app, host="127.0.0.1", port=8002, log_level="info")
