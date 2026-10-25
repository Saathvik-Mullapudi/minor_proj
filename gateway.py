#!/usr/bin/env python3
"""
gateway.py — Sub-stage 5.2: SLO-Aware AIMD Adaptive Batching Reverse Proxy Gateway
Runs on Port 8000.
Acts as the central network control plane and ingress router for the distributed edge pipeline:
  Client -> Gateway (Port 8000) -> Head Worker (Port 8001) -> Tail Worker (Port 8002)

Key MLSys Contribution:
  - Additive Increase / Multiplicative Decrease (AIMD) closed-loop feedback controller.
  - Dynamically adapts batch dispatch size B_t in [B_min, B_max] based on sliding-window p95 tail latency:
      B_{t+1} = min(B_max, B_t + alpha)   if L_p95 <= (1 - delta) * L_SLO
      B_{t+1} = max(B_min, floor(eta * B_t)) if L_p95 > L_SLO
      B_{t+1} = B_t                       otherwise (in deadband safety zone)
"""

import asyncio
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import os
import time
from typing import Any, Dict, List, Optional
import uuid
from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse
import httpx
import numpy as np
import torch
import uvicorn

from serialization import deserialize_tensor, serialize_tensor

# Network configuration
HEAD_WORKER_URL = os.getenv("HEAD_WORKER_URL", "http://127.0.0.1:8001")
DEFAULT_MAX_WAIT_MS = float(os.getenv("MAX_WAIT_MS", "35.0"))  # Max collation window timeout


class AIMDController:
    """
    SLO-Aware Additive Increase / Multiplicative Decrease (AIMD) Controller.
    Dynamically tunes batch size to maximize hardware throughput while strictly
    honoring tail latency (p95) SLO constraints.
    """

    def __init__(
        self,
        slo_target_ms: float = 60.0,
        safety_delta: float = 0.10,
        alpha: int = 1,
        eta: float = 0.5,
        b_min: int = 1,
        b_max: int = 16,
        window_size: int = 30,
        initial_b: int = 1,
    ):
        self.slo_target_ms = slo_target_ms
        self.safety_delta = safety_delta
        self.alpha = alpha
        self.eta = eta
        self.b_min = b_min
        self.b_max = b_max
        self.window_size = window_size
        self.current_batch_size = initial_b

        self.latency_window: deque = deque(maxlen=window_size)
        self.total_processed_requests = 0
        self.total_violations = 0

    def get_batch_size(self) -> int:
        """Returns the currently active batch dispatch limit B_t."""
        return self.current_batch_size

    def record_latencies(self, latencies: List[float]):
        """
        Records completed request latencies and applies the AIMD control update:
          - Safe Zone (p95 <= 0.90 * SLO): Additive Increase (B_t + 1)
          - SLO Breach Zone (p95 > SLO): Multiplicative Decrease (floor(0.5 * B_t))
          - Deadband Zone: Hold current batch size steady
        """
        for lat in latencies:
            self.latency_window.append(lat)
            self.total_processed_requests += 1
            if lat > self.slo_target_ms:
                self.total_violations += 1

        if len(self.latency_window) >= 5:
            arr = list(self.latency_window)
            p95 = float(np.percentile(arr, 95))
            safe_threshold = (1.0 - self.safety_delta) * self.slo_target_ms

            if p95 <= safe_threshold:
                # Additive Increase
                old_b = self.current_batch_size
                self.current_batch_size = min(self.b_max, self.current_batch_size + self.alpha)
                if self.current_batch_size != old_b:
                    print(f"[AIMD CONTROLLER] Additive Increase: B={old_b} -> {self.current_batch_size} (p95={p95:.1f}ms <= {safe_threshold:.1f}ms)")
            elif p95 > self.slo_target_ms:
                # Multiplicative Decrease
                old_b = self.current_batch_size
                self.current_batch_size = max(self.b_min, int(np.floor(self.eta * self.current_batch_size)))
                if self.current_batch_size != old_b:
                    print(f"[AIMD CONTROLLER] Multiplicative Decrease: B={old_b} -> {self.current_batch_size} (p95={p95:.1f}ms > SLO {self.slo_target_ms:.1f}ms)")
            else:
                # Deadband Zone (e.g. 54ms < p95 <= 60ms)
                pass

    def get_metrics(self, queue_depth: int = 0) -> Dict[str, Any]:
        """Calculates real-time telemetry metrics and percentiles over the sliding window."""
        if len(self.latency_window) > 0:
            arr = list(self.latency_window)
            p50 = float(np.percentile(arr, 50))
            p95 = float(np.percentile(arr, 95))
            p99 = float(np.percentile(arr, 99))
        else:
            p50, p95, p99 = 0.0, 0.0, 0.0

        violation_rate = (
            (self.total_violations / self.total_processed_requests * 100.0)
            if self.total_processed_requests > 0
            else 0.0
        )

        return {
            "current_batch_size": self.current_batch_size,
            "queue_depth": queue_depth,
            "target_slo_ms": self.slo_target_ms,
            "window_p50_ms": round(p50, 2),
            "window_p95_ms": round(p95, 2),
            "window_p99_ms": round(p99, 2),
            "slo_violation_rate_pct": round(violation_rate, 2),
            "total_processed_requests": self.total_processed_requests,
        }


# Global queues, clients, and AIMD controller
request_queue: Optional[asyncio.Queue] = None
http_client: Optional[httpx.AsyncClient] = None
scheduler_task: Optional[asyncio.Task] = None
shutdown_event = asyncio.Event()
aimd_controller: Optional[AIMDController] = None


@dataclass
class QueuedRequest:
    """Internal container for an enqueued inference request awaiting batch dispatch."""
    request_id: str
    image: torch.Tensor
    event: asyncio.Event
    response_container: Dict[str, Any] = field(default_factory=dict)
    arrival_time: float = field(default_factory=time.perf_counter)
    compress_fp16: bool = True
    link_mbps: float = 20.0


async def dispatch_batch(batch: List[QueuedRequest], client: httpx.AsyncClient):
    """
    Collates a gathered batch of requests, serializes them as a contiguous tensor,
    dispatches the batch to Head Worker (8001), and fulfills each client event.
    """
    global aimd_controller
    batch_size = len(batch)
    t_batch_start = time.perf_counter()

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
        batch_latencies = []
        for idx, req in enumerate(batch):
            pred = predictions[idx] if idx < len(predictions) else {}
            queue_wait_ms = (t_batch_start - req.arrival_time) * 1000.0
            total_client_ms = (time.perf_counter() - req.arrival_time) * 1000.0
            batch_latencies.append(total_client_ms)

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

        # 4. Feed measured latencies back into AIMD feedback loop
        if aimd_controller:
            aimd_controller.record_latencies(batch_latencies)

    except Exception as exc:
        err_msg = f"Gateway dispatch exception: {str(exc)}"
        for req in batch:
            req.response_container["error"] = err_msg
            req.response_container["status_code"] = status.HTTP_504_GATEWAY_TIMEOUT
            req.event.set()


async def scheduler_loop():
    """
    Continuous background worker that dequeues requests and collates batches.
    Batch size upper-bound is governed dynamically by AIMDController.
    """
    global request_queue, http_client, shutdown_event, aimd_controller
    print(f"[GATEWAY SCHEDULER] Started SLO-Aware AIMD Scheduler (Max Wait: {DEFAULT_MAX_WAIT_MS} ms).")

    while not shutdown_event.is_set():
        try:
            first_req: QueuedRequest = await request_queue.get()
        except asyncio.CancelledError:
            break

        batch: List[QueuedRequest] = [first_req]
        request_queue.task_done()

        # Dynamic batch limit from AIMD feedback controller
        current_max_batch = aimd_controller.get_batch_size() if aimd_controller else 8

        # 1. Immediately drain any requests already waiting in the queue
        while len(batch) < current_max_batch:
            try:
                next_req = request_queue.get_nowait()
                batch.append(next_req)
                request_queue.task_done()
            except asyncio.QueueEmpty:
                break

        # 2. Wait up to DEFAULT_MAX_WAIT_MS for subsequent requests if below current limit
        collation_deadline = time.perf_counter() + (DEFAULT_MAX_WAIT_MS / 1000.0)

        while len(batch) < current_max_batch:
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

        # Asynchronously dispatch the formed batch
        asyncio.create_task(dispatch_batch(batch, http_client))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes AIMD controller, HTTP client pool, FIFO queue, and starts background scheduler."""
    global request_queue, http_client, scheduler_task, shutdown_event, aimd_controller

    slo_target = float(os.getenv("TARGET_SLO_MS", "60.0"))
    print(f"[GATEWAY] Initializing SLO-Aware AIMD Gateway (SLO Target: {slo_target} ms)...")

    shutdown_event.clear()
    request_queue = asyncio.Queue()
    aimd_controller = AIMDController(slo_target_ms=slo_target, initial_b=1)

    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(60.0, connect=5.0),
        limits=httpx.Limits(max_keepalive_connections=100, max_connections=200),
    )

    scheduler_task = asyncio.create_task(scheduler_loop())
    print("[GATEWAY] Ingress queue, AIMD controller, and connection pool ready.")

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
    title="SLO-Aware Inference Pipeline - Gateway with AIMD",
    description="Asynchronous Ingress Reverse Proxy & AIMD Adaptive Batching Gateway on Port 8000",
    version="1.1.0",
    lifespan=lifespan,
)


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check() -> Dict[str, Any]:
    """Health check reporting gateway status and queue depth."""
    global request_queue, aimd_controller
    depth = request_queue.qsize() if request_queue else 0
    batch_size = aimd_controller.get_batch_size() if aimd_controller else 1
    return {
        "status": "ok",
        "service": "gateway",
        "current_queue_depth": depth,
        "current_batch_size": batch_size,
        "max_wait_ms": DEFAULT_MAX_WAIT_MS,
    }


@app.get("/metrics", status_code=status.HTTP_200_OK)
async def get_metrics() -> JSONResponse:
    """Telemetry endpoint reporting real-time AIMD controller metrics and percentiles."""
    global request_queue, aimd_controller
    depth = request_queue.qsize() if request_queue else 0
    if aimd_controller:
        metrics = aimd_controller.get_metrics(queue_depth=depth)
    else:
        metrics = {}
    return JSONResponse(content=metrics)


@app.post("/predict", status_code=status.HTTP_200_OK)
async def predict(
    request: Request,
    compress_fp16: bool = Query(default=True, description="Enable FP16 intermediate activation quantization"),
    link_mbps: float = Query(default=20.0, description="Emulated edge link speed in Mbps"),
) -> JSONResponse:
    """
    Ingress client prediction endpoint.
    Enqueues the request and awaits AIMD batch collation and downstream execution.
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
    print("[INFO] Starting SLO-Aware AIMD Gateway on http://127.0.0.1:8000...")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
