#!/usr/bin/env python3
"""
test_gateway_e2e.py — Sub-stage 5.1 Verification Test Suite
Tests 3-Tier End-to-End Inference:
  Client -> Gateway (Port 8000) -> Head Worker (Port 8001) -> Tail Worker (Port 8002)

Verifies:
  1. Gateway health and queue inspection.
  2. Single client request execution (B=1) with timeout-triggered dispatch.
  3. Concurrent client burst (C=5) verifying dynamic batch collation into a single batch (B=5).
"""

import asyncio
import time
from typing import Dict, List
import httpx
import torch

from serialization import serialize_tensor

GATEWAY_URL = "http://127.0.0.1:8000"
HEAD_URL = "http://127.0.0.1:8001"
TAIL_URL = "http://127.0.0.1:8002"


async def check_all_tiers():
    """Verifies that all three microservices in the distributed pipeline are operational."""
    print("[INIT] Verifying connectivity to all 3 pipeline tiers...")
    async with httpx.AsyncClient(timeout=5.0) as client:
        # Check Tail Worker (Port 8002)
        res = await client.get(f"{TAIL_URL}/health")
        assert res.status_code == 200
        print(f"  [+] Tier 3 - Tail Worker (8002): ONLINE ({res.json()['device']})")

        # Check Head Worker (Port 8001)
        res = await client.get(f"{HEAD_URL}/health")
        assert res.status_code == 200
        print(f"  [+] Tier 2 - Head Worker (8001): ONLINE ({res.json()['device']})")

        # Check Gateway (Port 8000)
        res = await client.get(f"{GATEWAY_URL}/health")
        assert res.status_code == 200
        data = res.json()
        print(f"  [+] Tier 1 - Gateway Proxy (8000): ONLINE (Queue Depth: {data['current_queue_depth']})")


async def send_single_prediction(client: httpx.AsyncClient, image_bytes: bytes, client_id: int = 1) -> Dict:
    """Helper coroutine sending an inference request to Gateway /predict."""
    t0 = time.perf_counter()
    res = await client.post(
        f"{GATEWAY_URL}/predict?compress_fp16=true&link_mbps=20.0",
        content=image_bytes,
        headers={"Content-Type": "application/octet-stream"},
    )
    client_latency_ms = (time.perf_counter() - t0) * 1000.0

    assert res.status_code == 200, f"Client {client_id} failed [{res.status_code}]: {res.text}"
    result = res.json()
    result["client_latency_ms"] = client_latency_ms
    result["client_id"] = client_id
    return result


async def run_gateway_verification():
    await check_all_tiers()

    # Generate synthetic image tensor
    torch.manual_seed(42)
    sample_image = torch.randn(3, 224, 224, dtype=torch.float32)
    image_bytes = serialize_tensor(sample_image, compress_fp16=False)

    async with httpx.AsyncClient(timeout=60.0) as client:
        # -------------------------------------------------------------
        # Test 1: Single Sequential Request (B=1)
        # -------------------------------------------------------------
        print("\n[TEST 1] Sending Single Isolated Request (Testing Timeout Dispatch B=1)...")
        res1 = await send_single_prediction(client, image_bytes, client_id=1)

        print(f"  - Request ID:            {res1['request_id']}")
        print(f"  - Batch Size Executed:   {res1['batch_size_executed']}")
        print(f"  - Gateway Queue Wait:    {res1['gateway_queue_wait_ms']:.2f} ms")
        print(f"  - Dispatch & Execution:  {res1['gateway_batch_dispatch_ms']:.2f} ms")
        print(f"  - Total Client Roundtrip:{res1['client_latency_ms']:.2f} ms")
        print(f"  - Top-5 Classes:         {res1['prediction']['top5_classes']}")

        assert res1["batch_size_executed"] == 1
        assert len(res1["prediction"]["top5_classes"]) == 5
        print("  [PASS] Test 1 (Single Request) verified successfully.")

        # -------------------------------------------------------------
        # Test 2: Concurrent Burst (C=5) -> Dynamic Batching
        # -------------------------------------------------------------
        print("\n[TEST 2] Firing Concurrent Burst of 5 Requests (Testing Batch Collation B=5)...")
        t_burst_start = time.perf_counter()

        tasks = [
            send_single_prediction(client, image_bytes, client_id=i + 1)
            for i in range(5)
        ]
        burst_results: List[Dict] = await asyncio.gather(*tasks)
        burst_total_time_ms = (time.perf_counter() - t_burst_start) * 1000.0

        print(f"  - Concurrent Requests Dispatched: 5")
        print(f"  - Total Time for All 5 Requests:  {burst_total_time_ms:.2f} ms")

        # Verify batch collation
        executed_batches = [r["batch_size_executed"] for r in burst_results]
        print(f"  - Batch Sizes Reported by Backend: {executed_batches}")

        # In a simultaneous async burst, all 5 requests should be gathered into a single batch of size 5
        assert all(b == 5 for b in executed_batches), f"Expected batch size 5 for all requests, got {executed_batches}"
        assert len(burst_results) == 5

        # Check unique request IDs
        req_ids = {r["request_id"] for r in burst_results}
        assert len(req_ids) == 5, "Duplicate request IDs found!"

        avg_client_lat = sum(r["client_latency_ms"] for r in burst_results) / 5.0
        print(f"  - Average Client Roundtrip Latency: {avg_client_lat:.2f} ms")
        print("  [PASS] Test 2 (Concurrent Batch Collation) verified successfully.")

    # -------------------------------------------------------------
    # 3. Formatted Summary Report
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SUB-STAGE 5.1: ASYNCHRONOUS GATEWAY EVALUATION SUMMARY")
    print("=" * 80)
    print(f"{'Workload Scenario':<36} | {'Batch Size':<12} | {'Queue Wait':<14} | {'Client Latency':<14}")
    print("-" * 80)
    print(f"{'Single Request (Timeout Dispatch)':<36} | {res1['batch_size_executed']:<12} | {res1['gateway_queue_wait_ms']:<11.2f} ms | {res1['client_latency_ms']:<11.2f} ms")
    print(f"{'Concurrent Burst (5 Simultaneous)':<36} | {burst_results[0]['batch_size_executed']:<12} | {burst_results[0]['gateway_queue_wait_ms']:<11.2f} ms | {avg_client_lat:<11.2f} ms")
    print("=" * 80)

    print("\n[SUCCESS] Sub-stage 5.1 (gateway.py) verification completed successfully!")


if __name__ == "__main__":
    asyncio.run(run_gateway_verification())
