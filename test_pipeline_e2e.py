#!/usr/bin/env python3
"""
test_pipeline_e2e.py — End-to-End Pipeline Integration Verification Suite
Tests the distributed multi-tier data plane across:
  Head Worker (Port 8001) -> Network Link -> Tail Worker (Port 8002)

Verifies:
  1. Health & connectivity of both distributed workers.
  2. FP32 Baseline Mode (Paper) with 10 Mbps emulated link.
  3. FP16 Proposed Mode (Ours) with 10 Mbps emulated link.
  4. Mathematical telemetry accuracy: 50% payload cut & 50% transfer delay reduction.
"""

import asyncio
import time
from typing import Dict
import httpx
import torch

from serialization import serialize_tensor


async def run_e2e_verification(
    head_url: str = "http://127.0.0.1:8001",
    tail_url: str = "http://127.0.0.1:8002",
):
    print("=" * 75)
    print("SUB-STAGE 3.2: DISTRIBUTED DATA-PLANE E2E INTEGRATION TEST")
    print("=" * 75)

    # 1. Health checks on both microservices
    print("\n[STEP 1] Checking worker health and readiness...")
    async with httpx.AsyncClient(timeout=10.0) as client:
        # Check Tail Worker (Port 8002)
        try:
            res_tail = await client.get(f"{tail_url}/health")
            assert res_tail.status_code == 200
            print(f"  [+] Tail Worker (Port 8002): ONLINE | Device: {res_tail.json()['device']}")
        except Exception as e:
            raise RuntimeError(f"Tail Worker at {tail_url} is not reachable! Make sure it is running. Error: {e}")

        # Check Head Worker (Port 8001)
        try:
            res_head = await client.get(f"{head_url}/health")
            assert res_head.status_code == 200
            print(f"  [+] Head Worker (Port 8001): ONLINE | Device: {res_head.json()['device']}")
        except Exception as e:
            raise RuntimeError(f"Head Worker at {head_url} is not reachable! Make sure it is running. Error: {e}")

    # Prepare standard input image tensor (B=1, 3, 224, 224)
    torch.manual_seed(42)
    sample_image = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    image_bytes = serialize_tensor(sample_image, compress_fp16=False)

    async with httpx.AsyncClient(timeout=60.0) as client:
        # -------------------------------------------------------------
        # Test 1: FP32 Baseline Pipeline (Paper Approach)
        # -------------------------------------------------------------
        print("\n[STEP 2] Evaluating Baseline Mode: Uncompressed FP32 (10 Mbps Link)...")
        t0 = time.perf_counter()
        res_fp32 = await client.post(
            f"{head_url}/infer_head?compress_fp16=false&link_mbps=10.0",
            content=image_bytes,
            headers={"Content-Type": "application/octet-stream"},
        )
        e2e_fp32_client_ms = (time.perf_counter() - t0) * 1000.0

        assert res_fp32.status_code == 200, f"FP32 request failed ({res_fp32.status_code}): {res_fp32.text}"
        data_fp32 = res_fp32.json()
        telem_fp32 = data_fp32["telemetry"]

        # -------------------------------------------------------------
        # Test 2: FP16 Proposed Pipeline (Our Optimization)
        # -------------------------------------------------------------
        print("\n[STEP 3] Evaluating Proposed Mode: Quantized FP16 (10 Mbps Link)...")
        t1 = time.perf_counter()
        res_fp16 = await client.post(
            f"{head_url}/infer_head?compress_fp16=true&link_mbps=10.0",
            content=image_bytes,
            headers={"Content-Type": "application/octet-stream"},
        )
        e2e_fp16_client_ms = (time.perf_counter() - t1) * 1000.0

        assert res_fp16.status_code == 200, f"FP16 request failed ({res_fp16.status_code}): {res_fp16.text}"
        data_fp16 = res_fp16.json()
        telem_fp16 = data_fp16["telemetry"]

        # -------------------------------------------------------------
        # Test 3: Zero-Delay Execution (Pure Compute Baseline)
        # -------------------------------------------------------------
        print("\n[STEP 4] Evaluating Pure Compute Baseline (0 Mbps Delay)...")
        res_zero = await client.post(
            f"{head_url}/infer_head?compress_fp16=false&link_mbps=0.0",
            content=image_bytes,
            headers={"Content-Type": "application/octet-stream"},
        )
        assert res_zero.status_code == 200
        telem_zero = res_zero.json()["telemetry"]
        assert telem_zero["emulated_network_delay_ms"] == 0.0

    # -------------------------------------------------------------
    # 5. Strict MLSys Assertions
    # -------------------------------------------------------------
    # Payload sizes
    assert 605000 <= telem_fp32["serialization_bytes"] <= 610000, "Unexpected FP32 payload size"
    assert 302000 <= telem_fp16["serialization_bytes"] <= 306000, "Unexpected FP16 payload size"
    payload_ratio = telem_fp16["serialization_bytes"] / telem_fp32["serialization_bytes"]
    assert 0.49 <= payload_ratio <= 0.51, f"Payload ratio {payload_ratio} is not ~0.50!"

    # Emulated network delays (Payload * 8 / 10e6)
    expected_fp32_delay = (telem_fp32["serialization_bytes"] * 8.0) / (10.0 * 1000.0)
    expected_fp16_delay = (telem_fp16["serialization_bytes"] * 8.0) / (10.0 * 1000.0)
    assert abs(telem_fp32["emulated_network_delay_ms"] - expected_fp32_delay) < 5.0
    assert abs(telem_fp16["emulated_network_delay_ms"] - expected_fp16_delay) < 5.0

    # Predictions check
    assert len(data_fp32["predictions"]) == 1
    assert len(data_fp16["predictions"]) == 1
    assert len(data_fp32["predictions"][0]["top5_classes"]) == 5
    assert len(data_fp16["predictions"][0]["top5_classes"]) == 5

    # -------------------------------------------------------------
    # 6. Formatted Comparison Table
    # -------------------------------------------------------------
    bytes_reduction_pct = (1.0 - payload_ratio) * 100.0
    latency_savings_ms = telem_fp32["total_pipeline_ms"] - telem_fp16["total_pipeline_ms"]
    latency_reduction_pct = (latency_savings_ms / telem_fp32["total_pipeline_ms"]) * 100.0

    print("\n" + "=" * 80)
    print("DISTRIBUTED PIPELINE E2E EVALUATION SUMMARY (10 Mbps Edge Link)")
    print("=" * 80)
    print(f"{'Pipeline Stage / Metric':<34} | {'FP32 Baseline (Paper)':<20} | {'FP16 Proposed (Ours)':<20}")
    print("-" * 80)
    print(f"{'Activation Payload':<34} | {telem_fp32['serialization_bytes'] / 1024:<17.2f} KB | {telem_fp16['serialization_bytes'] / 1024:<17.2f} KB")
    print(f"{'Payload Reduction':<34} | {'0.00% (Baseline)':<20} | {f'{bytes_reduction_pct:.2f}% (Half Wire)':<20}")
    print(f"{'Stage 1 (Head Compute)':<34} | {telem_fp32['worker1_compute_ms']:<17.2f} ms | {telem_fp16['worker1_compute_ms']:<17.2f} ms")
    print(f"{'Network Link Delay':<34} | {telem_fp32['emulated_network_delay_ms']:<17.2f} ms | {telem_fp16['emulated_network_delay_ms']:<17.2f} ms")
    print(f"{'Stage 2 (Tail Compute)':<34} | {telem_fp32['worker2_compute_ms']:<17.2f} ms | {telem_fp16['worker2_compute_ms']:<17.2f} ms")
    print("-" * 80)
    print(f"{'Total Server Pipeline Latency':<34} | {telem_fp32['total_pipeline_ms']:<17.2f} ms | {telem_fp16['total_pipeline_ms']:<17.2f} ms")
    print(f"{'Client Measured E2E Latency':<34} | {e2e_fp32_client_ms:<17.2f} ms | {e2e_fp16_client_ms:<17.2f} ms")
    print(f"{'E2E Latency Savings':<34} | {'---':<20} | {f'-{latency_savings_ms:.2f} ms ({latency_reduction_pct:.1f}%)':<20}")
    print("=" * 80)

    print("\n[PASS] All pipeline communications, delay emulations, and precision assertions passed.")
    print("[SUCCESS] Major Step 3 (Networked Asynchronous Microservices) is 100% complete!")


if __name__ == "__main__":
    asyncio.run(run_e2e_verification())
