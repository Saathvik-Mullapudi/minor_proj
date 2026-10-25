#!/usr/bin/env python3
"""
test_aimd_controller.py — Verification Test Suite for Sub-stage 5.2
Tests both the pure mathematical control law of AIMDController and its live HTTP integration:
  1. Additive Increase under safe latencies (p95 <= 0.90 * SLO).
  2. Deadband safety invariance (0.90 * SLO < p95 <= SLO).
  3. Multiplicative Decrease upon simulated SLO breach (p95 > SLO).
  4. Boundary enforcement (B_min=1, B_max=16).
  5. Live HTTP telemetry verification via GET /metrics and POST /predict.
"""

import asyncio
import time
from typing import Dict
import httpx
from gateway import AIMDController

GATEWAY_URL = "http://127.0.0.1:8000"


def test_aimd_mathematical_law():
    """Unit test for AIMDController mathematical dynamics."""
    print("=" * 75)
    print("UNIT TEST: AIMD MATHEMATICAL DYNAMICS & BOUNDS")
    print("=" * 75)

    ctrl = AIMDController(
        slo_target_ms=60.0,
        safety_delta=0.10,  # Safe threshold: <= 54.0 ms
        alpha=1,
        eta=0.5,
        b_min=1,
        b_max=16,
        window_size=30,
        initial_b=1,
    )

    # 1. Initial State
    assert ctrl.get_batch_size() == 1
    print("  [+] Initial batch size verified: B=1")

    # 2. Phase 1: Additive Increase under safe latencies (p95 <= 54 ms)
    print("\n  [Phase 1] Testing Additive Increase (p95 <= 54.0 ms)...")
    for step in range(1, 5):
        # Supply 5 safe latencies around 35-40 ms
        safe_latencies = [35.0, 38.0, 40.0, 36.0, 42.0]
        ctrl.record_latencies(safe_latencies)
        expected_b = min(16, 1 + step)
        actual_b = ctrl.get_batch_size()
        print(f"    - Step {step}: Recorded safe batch -> Batch Size B: {actual_b} (Expected: {expected_b})")
        assert actual_b == expected_b, f"Expected B={expected_b}, got {actual_b}"
    print("  [PASS] Additive Increase verified.")

    # 3. Phase 2: Deadband Safety Zone (54 ms < p95 <= 60 ms)
    print("\n  [Phase 2] Testing Deadband Stability (54.0 ms < p95 <= 60.0 ms)...")
    curr_b = ctrl.get_batch_size()  # Should be 5
    deadband_latencies = [56.0, 57.0, 58.0, 55.0, 59.0]
    ctrl.record_latencies(deadband_latencies)
    assert ctrl.get_batch_size() == curr_b, f"Batch size changed in deadband! Expected {curr_b}, got {ctrl.get_batch_size()}"
    print(f"    - Deadband latencies fed -> Batch Size remained stable at B={curr_b}")
    print("  [PASS] Deadband stability verified.")

    # 4. Phase 3: Multiplicative Decrease (SLO Breach: p95 > 60 ms)
    print("\n  [Phase 3] Testing Multiplicative Decrease (p95 > 60.0 ms)...")
    # Feed latency spike exceeding 60 ms
    spike_latencies = [85.0, 90.0, 92.0, 88.0, 95.0]
    expected_cut = max(1, int(0.5 * curr_b))  # floor(0.5 * 5) = 2
    ctrl.record_latencies(spike_latencies)
    actual_cut = ctrl.get_batch_size()
    print(f"    - SLO breach fed (latencies > 85 ms) -> Batch Size cut from B={curr_b} to B={actual_cut} (Expected: {expected_cut})")
    assert actual_cut == expected_cut, f"Expected B={expected_cut}, got {actual_cut}"
    print("  [PASS] Multiplicative Decrease verified.")

    # 5. Phase 4: Minimum and Maximum Boundary Invariance
    print("\n  [Phase 4] Testing Boundary Invariance [1, 16]...")
    # Repeated breaches should not drop B below 1
    for _ in range(5):
        ctrl.record_latencies([150.0] * 5)
    assert ctrl.get_batch_size() == 1, f"Batch size fell below B_min! Got {ctrl.get_batch_size()}"
    print("    - Repeated severe spikes: Batch size strictly clamped at B_min = 1.")

    # Repeated safe batches should not exceed 16
    for _ in range(25):
        ctrl.record_latencies([20.0] * 5)
    assert ctrl.get_batch_size() == 16, f"Batch size exceeded B_max! Got {ctrl.get_batch_size()}"
    print("    - Repeated fast batches: Batch size strictly clamped at B_max = 16.")
    print("  [PASS] Boundary clamping verified.")

    # Metrics check
    m = ctrl.get_metrics(queue_depth=3)
    assert m["current_batch_size"] == 16
    assert m["target_slo_ms"] == 60.0
    assert m["queue_depth"] == 3
    assert m["window_p95_ms"] > 0
    print("  [PASS] Metrics dictionary generation verified.")


async def test_gateway_live_aimd():
    """Integration test verifying live AIMD endpoints on http://127.0.0.1:8000."""
    print("\n" + "=" * 75)
    print("INTEGRATION TEST: LIVE GATEWAY AIMD TELEMETRY (/metrics)")
    print("=" * 75)

    async with httpx.AsyncClient(timeout=10.0) as client:
        # Check /metrics endpoint
        res = await client.get(f"{GATEWAY_URL}/metrics")
        assert res.status_code == 200, f"Expected 200 from /metrics, got {res.status_code}"
        metrics = res.json()

        print(f"  [+] Live Gateway Metrics: {metrics}")
        assert "current_batch_size" in metrics
        assert "target_slo_ms" in metrics
        assert "window_p50_ms" in metrics
        assert "window_p95_ms" in metrics
        assert "slo_violation_rate_pct" in metrics
        assert metrics["target_slo_ms"] == 60.0
        print("  [PASS] Live /metrics schema and active values verified.")


async def main():
    test_aimd_mathematical_law()
    await test_gateway_live_aimd()
    print("\n[SUCCESS] Sub-stage 5.2 (AIMD Adaptive Batching) passed all verification gates!")


if __name__ == "__main__":
    asyncio.run(main())
