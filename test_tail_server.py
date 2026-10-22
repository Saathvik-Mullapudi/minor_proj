#!/usr/bin/env python3
"""
test_tail_server.py — Automated Verification Suite for Sub-stage 3.1
Tests tail_server.py:
  1. Health check (GET /health)
  2. FP32 Activation Inference (POST /infer_tail) with batch size = 1
  3. FP16 Activation Inference (POST /infer_tail) with batch size = 2
  4. Bad Request / Error Handling (empty payload, corrupt payload, wrong tensor shape)
"""

import asyncio
import time
from typing import Dict
import httpx
import torch

from serialization import serialize_tensor


async def run_tail_server_tests(base_url: str = "http://127.0.0.1:8002"):
    print(f"[TEST SUITE] Connecting to Tail Worker at {base_url}...")

    # Wait for server to become responsive
    healthy = False
    for attempt in range(15):
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                res = await client.get(f"{base_url}/health")
                if res.status_code == 200:
                    data = res.json()
                    print(f"  [+] Connected! Worker Health: {data}")
                    assert data["status"] == "ok"
                    assert data["worker"] == "tail"
                    assert data["device"] == "cpu"
                    healthy = True
                    break
        except Exception:
            time.sleep(1.0)

    if not healthy:
        raise RuntimeError("Tail Server failed to respond to /health within 15 seconds!")

    async with httpx.AsyncClient(timeout=30.0) as client:
        # -------------------------------------------------------------
        # Test 1: FP32 Activation Stream (B=1)
        # -------------------------------------------------------------
        print("\n[TEST 1] Testing FP32 Intermediate Activation Transfer (Batch Size B=1)...")
        torch.manual_seed(42)
        act_fp32 = torch.randn(1, 197, 768, dtype=torch.float32)
        payload_fp32 = serialize_tensor(act_fp32, compress_fp16=False)
        print(f"  - Serialized FP32 payload: {len(payload_fp32):,} bytes ({len(payload_fp32)/1024:.2f} KB)")

        t0 = time.perf_counter()
        res_fp32 = await client.post(
            f"{base_url}/infer_tail",
            content=payload_fp32,
            headers={"Content-Type": "application/octet-stream", "x-compress-fp16": "false"},
        )
        roundtrip_fp32_ms = (time.perf_counter() - t0) * 1000.0

        assert res_fp32.status_code == 200, f"Expected 200 OK, got {res_fp32.status_code}: {res_fp32.text}"
        body_fp32 = res_fp32.json()
        print(f"  - Status: 200 OK")
        print(f"  - Roundtrip Latency: {roundtrip_fp32_ms:.2f} ms")
        print(f"  - Worker 2 Compute Latency: {body_fp32['worker2_compute_ms']} ms")
        print(f"  - Batch Size: {body_fp32['batch_size']}")
        print(f"  - Top-5 Classes: {body_fp32['predictions'][0]['top5_classes']}")
        print(f"  - Top-5 Probs:   {body_fp32['predictions'][0]['top5_probs']}")

        assert body_fp32["batch_size"] == 1
        assert len(body_fp32["predictions"]) == 1
        assert len(body_fp32["predictions"][0]["top5_classes"]) == 5
        assert len(body_fp32["predictions"][0]["top5_probs"]) == 5
        assert body_fp32["worker2_compute_ms"] > 0
        print("  [PASS] Test 1 (FP32) verified successfully.")

        # -------------------------------------------------------------
        # Test 2: FP16 Activation Stream (B=2)
        # -------------------------------------------------------------
        print("\n[TEST 2] Testing FP16 Intermediate Activation Transfer (Batch Size B=2)...")
        act_fp16 = torch.randn(2, 197, 768, dtype=torch.float32)
        payload_fp16 = serialize_tensor(act_fp16, compress_fp16=True)
        print(f"  - Serialized FP16 payload (B=2): {len(payload_fp16):,} bytes ({len(payload_fp16)/1024:.2f} KB)")

        t0 = time.perf_counter()
        res_fp16 = await client.post(
            f"{base_url}/infer_tail",
            content=payload_fp16,
            headers={"Content-Type": "application/octet-stream", "x-compress-fp16": "true"},
        )
        roundtrip_fp16_ms = (time.perf_counter() - t0) * 1000.0

        assert res_fp16.status_code == 200, f"Expected 200 OK, got {res_fp16.status_code}: {res_fp16.text}"
        body_fp16 = res_fp16.json()
        print(f"  - Status: 200 OK")
        print(f"  - Roundtrip Latency: {roundtrip_fp16_ms:.2f} ms")
        print(f"  - Worker 2 Compute Latency: {body_fp16['worker2_compute_ms']} ms")
        print(f"  - Batch Size: {body_fp16['batch_size']}")
        assert body_fp16["batch_size"] == 2
        assert len(body_fp16["predictions"]) == 2
        print("  [PASS] Test 2 (FP16 B=2) verified successfully.")

        # -------------------------------------------------------------
        # Test 3: Robust Error Handling (Empty, Corrupt, Wrong Dimensions)
        # -------------------------------------------------------------
        print("\n[TEST 3] Testing Error Handling & Bounds Protection...")

        # 3a. Empty payload
        res_empty = await client.post(f"{base_url}/infer_tail", content=b"")
        assert res_empty.status_code == 400
        print("  - Empty payload properly rejected (400 Bad Request).")

        # 3b. Corrupt binary payload
        res_corrupt = await client.post(f"{base_url}/infer_tail", content=b"GARBAGE_BYTES_12345")
        assert res_corrupt.status_code == 400
        print("  - Corrupt payload properly rejected (400 Bad Request).")

        # 3c. Wrong tensor dimensions (e.g. 2D instead of 3D)
        wrong_tensor = torch.randn(1, 768)
        wrong_bytes = serialize_tensor(wrong_tensor)
        res_wrong = await client.post(f"{base_url}/infer_tail", content=wrong_bytes)
        assert res_wrong.status_code == 400
        print("  - Invalid tensor dimensions rejected (400 Bad Request).")

        print("  [PASS] Test 3 (Error Handling) verified successfully.")

    print("\n" + "=" * 70)
    print("SUB-STAGE 3.1: TAIL WORKER VERIFICATION PASSED COMPLETELY")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_tail_server_tests())
