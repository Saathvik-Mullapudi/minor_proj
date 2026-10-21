#!/usr/bin/env python3
"""
verify_slices.py — Sub-stage 1.2: Hardware-Isolated Execution & In-Memory Serialization Verification
Executes sliced ViTHead and ViTTail across designated hardware devices:
  - Head Node: GPU (cuda:0 if available, else CPU fallback with explicit logging)
  - Tail Node: Host CPU (cpu)

Passes intermediate activations across device boundaries via in-memory binary serialization (io.BytesIO),
benchmarks compute and transfer latency, and asserts strict numerical parity against the monolithic ViT-Base.
"""

import io
import time
from typing import Dict, Tuple
import torch
from model_slices import create_vit_slices


def resolve_execution_devices() -> Tuple[torch.device, torch.device]:
    """
    Resolves hardware placement for Head and Tail slices.
    Head: cuda:0 (NVIDIA RTX 3050 Laptop GPU) if available, else cpu fallback.
    Tail: cpu (Host Multi-core CPU).
    """
    if torch.cuda.is_available():
        head_device = torch.device("cuda:0")
        device_name = torch.cuda.get_device_name(0)
        print(f"[HARDWARE] Head Worker assigned to GPU: {head_device} ({device_name})")
    else:
        head_device = torch.device("cpu")
        print("[HARDWARE] Head Worker assigned to CPU (CUDA unavailable/fallback)")

    tail_device = torch.device("cpu")
    print(f"[HARDWARE] Tail Worker assigned to CPU: {tail_device}")
    return head_device, tail_device


def run_warmup(head: torch.nn.Module, tail: torch.nn.Module, ref_model: torch.nn.Module, head_dev: torch.device, tail_dev: torch.device):
    """Executes a single dummy forward pass to prime PyTorch JIT engines and kernels."""
    dummy_x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        _ = ref_model(dummy_x)
        h_dummy = head(dummy_x.to(head_dev))
        if head_dev.type == "cuda":
            torch.cuda.synchronize()
        _ = tail(h_dummy.to(tail_dev))


def verify_pipeline() -> Dict[str, float]:
    """
    Executes the end-to-end pipeline verification:
      1. Deterministic input generation
      2. Monolithic benchmark (CPU)
      3. Sliced execution with in-memory binary serialization
      4. Numerical parity assertion
    """
    head_device, tail_device = resolve_execution_devices()

    print("\n[INFO] Initializing model slices and canonical reference model...")
    head, tail, ref_model = create_vit_slices(weights=None)

    # Place modules onto their target hardware
    ref_model = ref_model.to("cpu").eval()
    head = head.to(head_device).eval()
    tail = tail.to(tail_device).eval()

    # Device warmup
    run_warmup(head, tail, ref_model, head_device, tail_device)

    # Deterministic evaluation input
    torch.manual_seed(42)
    x = torch.randn(1, 3, 224, 224, dtype=torch.float32)

    # -------------------------------------------------------------
    # 1. Monolithic Execution Benchmark (Baseline Reference)
    # -------------------------------------------------------------
    t0 = time.perf_counter()
    with torch.no_grad():
        monolithic_logits = ref_model(x)
    t_monolithic_ms = (time.perf_counter() - t0) * 1000.0

    # -------------------------------------------------------------
    # 2. Sliced Pipeline Execution
    # -------------------------------------------------------------
    # Stage 1: Head computation on head_device
    x_head = x.to(head_device)
    t1 = time.perf_counter()
    with torch.no_grad():
        head_activation = head(x_head)
    if head_device.type == "cuda":
        torch.cuda.synchronize()
    t_head_ms = (time.perf_counter() - t1) * 1000.0

    # Intermediate Serialization: Head GPU -> In-memory BytesIO Buffer
    t2 = time.perf_counter()
    buffer = io.BytesIO()
    torch.save(head_activation, buffer)
    t_ser_ms = (time.perf_counter() - t2) * 1000.0
    payload_bytes = buffer.getbuffer().nbytes

    # Intermediate Deserialization: In-memory BytesIO Buffer -> Tail CPU
    t3 = time.perf_counter()
    buffer.seek(0)
    tail_activation = torch.load(buffer, map_location=tail_device, weights_only=False)
    t_deser_ms = (time.perf_counter() - t3) * 1000.0

    # Stage 2: Tail computation on tail_device
    t4 = time.perf_counter()
    with torch.no_grad():
        sliced_logits = tail(tail_activation)
    t_tail_ms = (time.perf_counter() - t4) * 1000.0

    # Total pipelined latency (compute + serialization + deserialization)
    t_total_pipeline_ms = t_head_ms + t_ser_ms + t_deser_ms + t_tail_ms

    # -------------------------------------------------------------
    # 3. Shape & Mathematical Parity Assertions
    # -------------------------------------------------------------
    assert head_activation.shape == (1, 197, 768), f"Wrong head activation shape: {head_activation.shape}"
    assert tail_activation.shape == (1, 197, 768), f"Wrong tail input shape: {tail_activation.shape}"
    assert sliced_logits.shape == (1, 1000), f"Wrong sliced logits shape: {sliced_logits.shape}"
    assert monolithic_logits.shape == (1, 1000), f"Wrong monolithic logits shape: {monolithic_logits.shape}"

    # Verify payload byte size (FP32: 197 * 768 * 4 = 605,184 bytes + PyTorch archive header)
    raw_tensor_bytes = 1 * 197 * 768 * 4  # 605,184
    assert 605184 <= payload_bytes <= 610000, (
        f"Payload bytes {payload_bytes} outside expected range [605184, 610000]"
    )

    # Numerical deviation check
    max_abs_diff = (monolithic_logits - sliced_logits).abs().max().item()
    torch.testing.assert_close(
        sliced_logits,
        monolithic_logits,
        rtol=1e-4,
        atol=1e-4,
        msg="Sliced ViT output deviates from monolithic baseline!",
    )
    assert max_abs_diff < 1e-4, f"Max deviation {max_abs_diff} exceeded threshold 1e-4"

    # -------------------------------------------------------------
    # 4. Formatted Summary Report
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUB-STAGE 1.2: PIPELINE EXECUTION & PARITY AUDIT")
    print("=" * 70)
    print(f"{'Metric':<38} | {'Value':<28}")
    print("-" * 70)
    print(f"{'Head Hardware Device':<38} | {str(head_device):<28}")
    print(f"{'Tail Hardware Device':<38} | {str(tail_device):<28}")
    print(f"{'Intermediate Tensor Shape':<38} | {str(list(head_activation.shape)):<28}")
    print(f"{'Raw Element Bytes (FP32)':<38} | {raw_tensor_bytes:,} bytes")
    print(f"{'In-Memory Serialized Payload':<38} | {payload_bytes:,} bytes ({payload_bytes/1024:.2f} KB)")
    print(f"{'Maximum Absolute Logit Deviation':<38} | {max_abs_diff:.2e}")
    print("-" * 70)
    print(f"{'Stage 1 (Head Compute)':<38} | {t_head_ms:.2f} ms")
    print(f"{'Intermediate Serialization (Save)':<38} | {t_ser_ms:.2f} ms")
    print(f"{'Intermediate Deserialization (Load)':<38} | {t_deser_ms:.2f} ms")
    print(f"{'Stage 2 (Tail Compute)':<38} | {t_tail_ms:.2f} ms")
    print(f"{'Total Sliced Pipeline Latency':<38} | {t_total_pipeline_ms:.2f} ms")
    print(f"{'Monolithic CPU Baseline Latency':<38} | {t_monolithic_ms:.2f} ms")
    print("=" * 70)

    print("\n[PASS] All shape, memory payload, and numerical equivalence assertions passed.")
    print("[SUCCESS] Step 1 of the MLSys architecture is complete and fully verified.")

    return {
        "head_ms": t_head_ms,
        "ser_ms": t_ser_ms,
        "deser_ms": t_deser_ms,
        "tail_ms": t_tail_ms,
        "total_pipeline_ms": t_total_pipeline_ms,
        "monolithic_ms": t_monolithic_ms,
        "payload_bytes": payload_bytes,
        "max_abs_diff": max_abs_diff,
    }


if __name__ == "__main__":
    verify_pipeline()
