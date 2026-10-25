#!/usr/bin/env python3
"""
benchmark_baseline.py — Sub-stage 4.1: 2023 Paper Baseline Benchmark Profiler
Replicates the exact mathematical evaluation from Parthasarathy & Krishnamachari (USC, 2023):
  "Partitioning and Deployment of Deep Neural Networks on Edge Clusters"

Mathematical Framework:
  - Stage 1 Service Time: S1 = c1 + gamma1
      * c1 = Head compute latency on GPU
      * gamma1 = Intermediate serialization + network transmission delay
  - Stage 2 Service Time: S2 = c2
      * c2 = Tail compute latency on CPU
  - Bottleneck Stage Latency: beta = max(S1, S2) = max(c1 + gamma1, c2)
  - Steady-State Pipelined Throughput: Theta = 1 / beta (requests/sec)

Configuration:
  - Precision: Uncompressed FP32 (Paper Baseline)
  - Emulated Link Bandwidth: 20.0 Mbps (typical edge wireless link)
  - Workload: 30 sequential requests of batch size B=1 (5 discarded warmup iterations)
"""

import json
import time
from typing import Dict, List
import httpx
import numpy as np
import torch

from serialization import serialize_tensor

HEAD_URL = "http://127.0.0.1:8001"
TAIL_URL = "http://127.0.0.1:8002"
LINK_MBPS = 20.0
WARMUP_REQS = 5
BENCHMARK_REQS = 30


def check_servers_healthy() -> bool:
    """Verifies that both Head (8001) and Tail (8002) microservices are reachable."""
    print("[INIT] Verifying connectivity to distributed worker nodes...")
    try:
        with httpx.Client(timeout=5.0) as client:
            res_tail = client.get(f"{TAIL_URL}/health")
            if res_tail.status_code != 200:
                print(f"[ERROR] Tail Worker at {TAIL_URL} returned status {res_tail.status_code}")
                return False
            print(f"  [+] Tail Worker (Port 8002): ONLINE ({res_tail.json()['device']})")

            res_head = client.get(f"{HEAD_URL}/health")
            if res_head.status_code != 200:
                print(f"[ERROR] Head Worker at {HEAD_URL} returned status {res_head.status_code}")
                return False
            print(f"  [+] Head Worker (Port 8001): ONLINE ({res_head.json()['device']})")
        return True
    except Exception as e:
        print(f"[ERROR] Connection check failed: {e}")
        print("Please ensure both 'tail_server.py' (8002) and 'head_server.py' (8001) are running.")
        return False


def run_baseline_benchmark() -> Dict:
    """Executes the 2023 paper baseline profiling workload."""
    if not check_servers_healthy():
        raise SystemExit(1)

    print(f"\n[BENCHMARK] Generating deterministic test image (1, 3, 224, 224)...")
    torch.manual_seed(42)
    sample_image = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    image_bytes = serialize_tensor(sample_image, compress_fp16=False)

    infer_endpoint = f"{HEAD_URL}/infer_head?compress_fp16=false&link_mbps={LINK_MBPS}"

    with httpx.Client(timeout=60.0) as client:
        # 1. Warm-up Phase
        print(f"[WARMUP] Executing {WARMUP_REQS} discarded warm-up queries...")
        for i in range(WARMUP_REQS):
            res = client.post(
                infer_endpoint,
                content=image_bytes,
                headers={"Content-Type": "application/octet-stream"},
            )
            if res.status_code != 200:
                raise RuntimeError(f"Warmup request {i+1} failed: {res.text}")
        print("  [+] Warm-up phase complete. Steady-state reached.\n")

        # 2. Benchmark Measurement Phase
        print(f"[MEASUREMENT] Executing {BENCHMARK_REQS} sequential benchmark iterations...")
        c1_list: List[float] = []
        gamma1_list: List[float] = []
        s1_list: List[float] = []
        c2_list: List[float] = []
        s2_list: List[float] = []
        e2e_list: List[float] = []
        wire_bytes: int = 0

        for req_idx in range(1, BENCHMARK_REQS + 1):
            t0 = time.perf_counter()
            res = client.post(
                infer_endpoint,
                content=image_bytes,
                headers={"Content-Type": "application/octet-stream"},
            )
            t_e2e_client_ms = (time.perf_counter() - t0) * 1000.0

            if res.status_code != 200:
                raise RuntimeError(f"Request {req_idx} failed: {res.text}")

            data = res.json()
            telem = data["telemetry"]

            c1 = telem["worker1_compute_ms"]
            ser_ms = telem.get("serialization_time_ms", 0.0)
            net_delay = telem["emulated_network_delay_ms"]
            gamma1 = ser_ms + net_delay
            s1 = c1 + gamma1

            c2 = telem["worker2_compute_ms"]
            s2 = c2

            wire_bytes = telem["serialization_bytes"]

            c1_list.append(c1)
            gamma1_list.append(gamma1)
            s1_list.append(s1)
            c2_list.append(c2)
            s2_list.append(s2)
            e2e_list.append(t_e2e_client_ms)

            if req_idx % 10 == 0 or req_idx == BENCHMARK_REQS:
                print(f"  - Progress: {req_idx}/{BENCHMARK_REQS} requests completed | Last E2E: {t_e2e_client_ms:.1f} ms")

    # 3. Statistical Aggregation
    c1_mean = float(np.mean(c1_list))
    gamma1_mean = float(np.mean(gamma1_list))
    s1_mean = float(np.mean(s1_list))
    c2_mean = float(np.mean(c2_list))
    s2_mean = float(np.mean(s2_list))

    # Bottleneck Stage Latency beta = max(S1, S2)
    beta = max(s1_mean, s2_mean)
    bottleneck_stage = "Stage 1 (Head + Network)" if s1_mean >= s2_mean else "Stage 2 (Tail)"

    # Steady-State Throughput Theta = 1000 / beta (req/sec)
    throughput = (1000.0 / beta) if beta > 0 else 0.0

    # End-to-End Latency Percentiles
    e2e_mean = float(np.mean(e2e_list))
    e2e_p50 = float(np.percentile(e2e_list, 50))
    e2e_p95 = float(np.percentile(e2e_list, 95))
    e2e_p99 = float(np.percentile(e2e_list, 99))

    # 4. Formatted Mathematical Breakdown Table
    print("\n" + "=" * 80)
    print("PARTHASARATHY & KRISHNAMACHARI (2023) BASELINE MATHEMATICAL PROFILE")
    print("=" * 80)
    print(f"{'Parameter / Mathematical Symbol':<42} | {'Measured Value':<32}")
    print("-" * 80)
    print(f"{'Precision Mode':<42} | {'Uncompressed FP32 (Paper Baseline)':<32}")
    print(f"{'Intermediate Activation Payload (A_k)':<42} | {wire_bytes:,} bytes ({wire_bytes/1024:.2f} KB)")
    print(f"{'Emulated Link Bandwidth (B_link)':<42} | {LINK_MBPS:.1f} Mbps")
    print("-" * 80)
    print(f"{'Stage 1 Compute Latency (c_1)':<42} | {c1_mean:.2f} ms")
    print(f"{'Stage 1 Transfer & Ser. Overhead (gamma_1)':<42} | {gamma1_mean:.2f} ms")
    print(f"{'Total Stage 1 Service Time (S_1 = c_1 + gamma_1)':<42} | {s1_mean:.2f} ms")
    print(f"{'Stage 2 (Tail) Compute Latency (c_2)':<42} | {c2_mean:.2f} ms")
    print(f"{'Total Stage 2 Service Time (S_2 = c_2)':<42} | {s2_mean:.2f} ms")
    print("-" * 80)
    print(f"{'Bottleneck Stage Latency (beta = max(S_1, S_2))':<42} | {beta:.2f} ms ({bottleneck_stage})")
    print(f"{'Theoretical Steady-State Throughput (Theta = 1/beta)':<42} | {throughput:.2f} requests/sec")
    print("-" * 80)
    print(f"{'Client Measured E2E Latency (Mean)':<42} | {e2e_mean:.2f} ms")
    print(f"{'Client Measured E2E Latency (p50)':<42} | {e2e_p50:.2f} ms")
    print(f"{'Client Measured E2E Latency (p95)':<42} | {e2e_p95:.2f} ms")
    print(f"{'Client Measured E2E Latency (p99)':<42} | {e2e_p99:.2f} ms")
    print("=" * 80)

    # 5. Export JSON
    results = {
        "configuration": {
            "precision": "FP32",
            "link_mbps": LINK_MBPS,
            "benchmark_requests": BENCHMARK_REQS,
            "warmup_requests": WARMUP_REQS,
            "activation_bytes": wire_bytes,
        },
        "stage_metrics": {
            "c1_mean_ms": round(c1_mean, 2),
            "gamma1_mean_ms": round(gamma1_mean, 2),
            "s1_mean_ms": round(s1_mean, 2),
            "c2_mean_ms": round(c2_mean, 2),
            "s2_mean_ms": round(s2_mean, 2),
            "beta_bottleneck_ms": round(beta, 2),
            "bottleneck_stage": bottleneck_stage,
            "theoretical_throughput_rps": round(throughput, 2),
        },
        "e2e_latency_ms": {
            "mean": round(e2e_mean, 2),
            "p50": round(e2e_p50, 2),
            "p95": round(e2e_p95, 2),
            "p99": round(e2e_p99, 2),
        },
        "raw_samples": {
            "c1_ms": [round(x, 2) for x in c1_list],
            "gamma1_ms": [round(x, 2) for x in gamma1_list],
            "s1_ms": [round(x, 2) for x in s1_list],
            "c2_ms": [round(x, 2) for x in c2_list],
            "e2e_ms": [round(x, 2) for x in e2e_list],
        },
    }

    output_file = "baseline_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[SUCCESS] Baseline profile exported to '{output_file}'.")
    return results


if __name__ == "__main__":
    run_baseline_benchmark()
