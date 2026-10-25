#!/usr/bin/env python3
"""
gateway.py — Sub-stage 5.1: Asynchronous Reverse Proxy Gateway
Runs on Port 8000.
Acts as the central network control plane and ingress router for the distributed edge pipeline:
  Client -> Gateway (Port 8000) -> Head Worker (Port 8001) -> Tail Worker (Port 8002)

Features:
  - Non-blocking request ingestion via asyncio.Queue.
  - Dynamic request collation and batching with configurable max_batch_size and max_wait_ms timeout.
  - Transparent event-driven request-response rendezvous using asyncio.Event.
  - Zero neural network execution on the gateway (pure asynchronous control plane).
"""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import os
import time
from typing import Any, Dict, List, Optional
import uuid
from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse
import httpx
import torch
import uvicorn

from serialization import deserialize_tensor, serialize_tensor

# Configuration
HEAD_WORKER_URL = os.getenv("HEAD_WORKER_URL", "http://127.0.0.1:8001")
DEFAULT_MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "8"))
DEFAULT_MAX_WAIT_MS = float(os.getenv("MAX_WAIT_MS", "35.0"))  # 35 ms collation window

# Global queues and clients
request_queue: Optional[asyncio.Queue] = None
http_client: Optional[httpx.AsyncClient] = None
scheduler_task: Optional[asyncio.Task] = None
shutdown_event = asyncio.Event()


@dataclass
class QueuedRequest:
    """Internal container for an enqueued inference request awaiting batch dispatch."""
    request_id: str
    image: torch.Tensor
    event: asyncio.Event
    response_container: Dict[str, Any] = field(default_factory=dict)
    arrival_time: float = field(default_factory=time.perf_counter)
    compress_fp16: bool = False
    link_mbps: float = 20.0


async def dispatch_batch(batch: List[QueuedRequest], client: httpx.AsyncClient):
    """
    Collates a gathered batch of requests, serializes them as a contiguous tensor,
    dispatches the batch to Head Worker (8001), and fulfills each client event.
    """
    batch_size = len(batch)
    t_batch_start = time.perf_counter()

    # Determine batch-level query flags from the gathered requests
    compress_fp16 = batch[0].compress_fp16
    link_mbps = batch[0].link_mbps

    try:
        # 1. Collate individual image tensors (1, 3, 224, 224) into a single batch (B, 3, 224, 224)
        collated_images = torch.cat([req.image.reshape(1, 3, 224, 224) for req in batch], dim=0)
        serialized_payload = serialize_tensor(collated_images, compress_fp16=False)

        # 2. Dispatch to Head Worker on Port 8001
        target_url = f"{HEAD_WORKER_URL}/infer_head?compress_fp16={str(compress_fp16).lower()}&link_mbps={link_mbps}"
        res = await client.post(
            target_url,
            content=serialized_payload,
            headers={"Content-Type": "application/octet-stream"},
            timeout=30.0,
        )

        if res.status_code != 200:
            err_msg = f"Head Worker error [{res.status_code}]: {res.text}"
            for req in batch:
                req.response_container["error"] = err_msg
                req.response_container["status_code"] = status.HTTP_502_BAD_GATEWAY
                req.event.set()
            return

        data = res.json()
        predictions = data.get("predictions", [])
        telemetry = data.get("telemetry", {})
        batch_dispatch_ms = (time.perf_counter() - t_batch_start) * 1000.0

        # 3. Unpack and distribute individual predictions to waiting client requests
        for idx, req in enumerate(batch):
            pred = predictions[idx] if idx < len(predictions) else {}
            queue_wait_ms = (t_batch_start - req.arrival_time) * 1000.0
            total_client_ms = (time.perf_counter() - req.arrival_time) * 1000.0

            req.response_container["data"] = {
                "request_id": req.request_id,
                "prediction": pred,
                "batch_size_executed": batch_size,
                "gateway_queue_wait_ms": round(queue_wait_ms, 2),
                "gateway_batch_dispatch_ms": round(batch_dispatch_ms, 2),
                "total_gateway_e2e_ms": round(total_client_ms, 2),
                "pipeline_telemetry": telemetry,
            }
            req.event.set()

    except Exception as exc:
        err_msg = f"Gateway dispatch exception: {str(exc)}"
        for req in batch:
            req.response_container["error"] = err_msg
            req.response_container["status_code"] = status.HTTP_504_GATEWAY_TIMEOUT
            req.event.set()


async def scheduler_loop():
    """
    Continuous background worker that dequeues requests and collates batches.
    Batches are formed when either:
      1. Batch reaches max_batch_size, OR
      2. Oldest request's wait time reaches max_wait_ms.
    """
    global request_queue, http_client, shutdown_event
    print(f"[GATEWAY SCHEDULER] Started background batch scheduler (Max Batch: {DEFAULT_MAX_BATCH_SIZE}, Max Wait: {DEFAULT_MAX_WAIT_MS} ms).")

    while not shutdown_event.is_set():
        try:
            # Block until at least one request is available
            first_req: QueuedRequest = await request_queue.get()
        except asyncio.CancelledError:
            break

        batch: List[QueuedRequest] = [first_req]
        request_queue.task_done()

        # 1. Immediately drain any requests already present in the queue
        while len(batch) < DEFAULT_MAX_BATCH_SIZE:
            try:
                next_req = request_queue.get_nowait()
                batch.append(next_req)
                request_queue.task_done()
            except asyncio.QueueEmpty:
                break

        # 2. Wait up to DEFAULT_MAX_WAIT_MS for subsequent requests
        collation_deadline = time.perf_counter() + (DEFAULT_MAX_WAIT_MS / 1000.0)

        while len(batch) < DEFAULT_MAX_BATCH_SIZE:
            remaining_time = collation_deadline - time.perf_counter()
            if remaining_time <= 0:
                break
            try:
                next_req: QueuedRequest = await asyncio.wait_for(
                    request_queue.get(),
                    timeout=remaining_time,
                )
                batch.append(next_req)
                request_queue.task_done()
            except asyncio.TimeoutError:
                break
            except asyncio.CancelledError:
                break

        # Asynchronously dispatch the formed batch without blocking the scheduler loop
        asyncio.create_task(dispatch_batch(batch, http_client))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes HTTP client pool, FIFO queue, and starts background scheduler."""
    global request_queue, http_client, scheduler_task, shutdown_event

    print("[GATEWAY] Initializing Asynchronous Reverse Proxy Gateway...")
    shutdown_event.clear()
    request_queue = asyncio.Queue()

    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(60.0, connect=5.0),
        limits=httpx.Limits(max_keepalive_connections=100, max_connections=200),
    )

    scheduler_task = asyncio.create_task(scheduler_loop())
    print("[GATEWAY] Ingress queue and persistent connection pool ready.")

    yield

    print("[GATEWAY] Initiating graceful shutdown...")
    shutdown_event.set()
    if scheduler_task:
        scheduler_task.cancel()
        try:
            await scheduler_task
        except asyncio.CancelledError:
            pass

    if http_client:
        await http_client.aclose()
    print("[GATEWAY] Shutdown complete.")


app = FastAPI(
    title="SLO-Aware Inference Pipeline - Gateway",
    description="Asynchronous Ingress Reverse Proxy & Dynamic Batching Gateway on Port 8000",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check() -> Dict[str, Any]:
    """Health check reporting gateway status and queue depth."""
    global request_queue
    depth = request_queue.qsize() if request_queue else 0
    return {
        "status": "ok",
        "service": "gateway",
        "current_queue_depth": depth,
        "max_batch_size": DEFAULT_MAX_BATCH_SIZE,
        "max_wait_ms": DEFAULT_MAX_WAIT_MS,
    }


@app.post("/predict", status_code=status.HTTP_200_OK)
async def predict(
    request: Request,
    compress_fp16: bool = Query(default=True, description="Enable FP16 intermediate activation quantization"),
    link_mbps: float = Query(default=20.0, description="Emulated edge link speed in Mbps"),
) -> JSONResponse:
    """
    Ingress client prediction endpoint.
    Enqueues the request and awaits batch collation and downstream execution.
    """
    global request_queue
    if request_queue is None or shutdown_event.is_set():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Gateway scheduler not accepting requests.",
        )

    # 1. Parse / Deserialize input image tensor
    body = await request.body()
    if body and len(body) > 0:
        try:
            image_tensor = deserialize_tensor(body, device="cpu", upcast_fp32=True)
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid binary image tensor format.",
            )
    else:
        # Default test sample
        image_tensor = torch.randn(1, 3, 224, 224, dtype=torch.float32)

    if image_tensor.dim() == 4 and image_tensor.shape[0] == 1:
        image_tensor = image_tensor.squeeze(0)

    if image_tensor.dim() != 3 or image_tensor.shape[0] != 3 or image_tensor.shape[1] != 224 or image_tensor.shape[2] != 224:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Expected image tensor with shape (3, 224, 224), got {list(image_tensor.shape)}",
        )

    # 2. Enqueue request
    req_id = str(uuid.uuid4())
    event = asyncio.Event()
    container: Dict[str, Any] = {}

    queued_item = QueuedRequest(
        request_id=req_id,
        image=image_tensor,
        event=event,
        response_container=container,
        arrival_time=time.perf_counter(),
        compress_fp16=compress_fp16,
        link_mbps=link_mbps,
    )

    await request_queue.put(queued_item)

    # 3. Await batch fulfillment (with 30-second deadline safeguard)
    try:
        await asyncio.wait_for(event.wait(), timeout=30.0)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Inference request timed out waiting for backend pipeline execution.",
        )

    if "error" in container:
        raise HTTPException(
            status_code=container.get("status_code", status.HTTP_502_BAD_GATEWAY),
            detail=container["error"],
        )

    return JSONResponse(content=container.get("data", {}))


if __name__ == "__main__":
    print("[INFO] Starting Gateway Proxy on http://127.0.0.1:8000...")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
