#!/usr/bin/env python3
"""
serialization.py — Sub-stage 2.1: Binary Tensor Serialization & FP16 Quantization
Provides production-grade, zero-disk binary serialization and precision compression
for intermediate activation tensors passing across distributed edge microservices.

Core Contribution:
  - Quantizes FP32 intermediate activations (197x768 = 151,296 floats) to FP16 (IEEE 754 half-precision)
    at the boundary cut-point, cutting network wire payload from ~606 KB to ~303 KB (~50% reduction).
  - Automatically upcasts back to FP32 at the receiving worker to preserve full precision in downstream layers.
"""

import io
import time
from typing import Dict, Tuple, Union
import torch


def serialize_tensor(tensor: torch.Tensor, compress_fp16: bool = False) -> bytes:
    """
    Serializes a PyTorch tensor into a compact in-memory byte stream via io.BytesIO.

    Args:
        tensor (torch.Tensor): PyTorch tensor to serialize.
        compress_fp16 (bool): If True, quantizes tensor to IEEE 754 half-precision (float16)
                             before serialization to halve wire payload size.

    Returns:
        bytes: Serialized binary byte stream.

    Raises:
        TypeError: If input is not a torch.Tensor.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")

    t = tensor.detach()
    if compress_fp16:
        t = t.half()

    buffer = io.BytesIO()
    torch.save(t, buffer)
    return buffer.getvalue()


def deserialize_tensor(
    data: bytes,
    device: Union[str, torch.device] = "cpu",
    upcast_fp32: bool = True,
) -> torch.Tensor:
    """
    Deserializes a binary byte stream into a PyTorch tensor.

    Args:
        data (bytes): Serialized binary byte stream.
        device (str or torch.device): Target hardware device ('cpu', 'cuda:0', etc.).
        upcast_fp32 (bool): If True and the loaded tensor is float16, upcasts back to float32
                           to ensure seamless compatibility with downstream compute layers.

    Returns:
        torch.Tensor: Deserialized PyTorch tensor ready for forward computation.

    Raises:
        ValueError: If data is empty or invalid.
    """
    if not data or len(data) == 0:
        raise ValueError("Cannot deserialize empty byte buffer.")

    buffer = io.BytesIO(data)
    tensor = torch.load(buffer, map_location=device, weights_only=False)

    if upcast_fp32 and tensor.dtype == torch.float16:
        tensor = tensor.float()

    return tensor


def benchmark_serialization(num_trials: int = 50) -> Dict[str, Dict[str, float]]:
    """
    Runs micro-benchmarks comparing FP32 vs. FP16 serialization:
      - Wire payload size (bytes and KB)
      - Serialization latency (ms)
      - Deserialization latency (ms)
      - Numerical reconstruction error
    """
    torch.manual_seed(42)
    # Target activation shape for ViT-Base (batch_size=1, tokens=197, hidden_dim=768)
    sample_tensor = torch.randn(1, 197, 768, dtype=torch.float32)

    # 1. Warmup runs
    _ = deserialize_tensor(serialize_tensor(sample_tensor, compress_fp16=False))
    _ = deserialize_tensor(serialize_tensor(sample_tensor, compress_fp16=True))

    # 2. Benchmark FP32 (Uncompressed Paper Baseline)
    t_ser_fp32_list, t_deser_fp32_list = [], []
    for _ in range(num_trials):
        t0 = time.perf_counter()
        raw_fp32 = serialize_tensor(sample_tensor, compress_fp16=False)
        t_ser_fp32_list.append((time.perf_counter() - t0) * 1000.0)

        t1 = time.perf_counter()
        rec_fp32 = deserialize_tensor(raw_fp32, device="cpu", upcast_fp32=False)
        t_deser_fp32_list.append((time.perf_counter() - t1) * 1000.0)

    fp32_bytes = len(raw_fp32)
    fp32_max_err = (sample_tensor - rec_fp32).abs().max().item()

    # 3. Benchmark FP16 (Our Optimization)
    t_ser_fp16_list, t_deser_fp16_list = [], []
    for _ in range(num_trials):
        t0 = time.perf_counter()
        raw_fp16 = serialize_tensor(sample_tensor, compress_fp16=True)
        t_ser_fp16_list.append((time.perf_counter() - t0) * 1000.0)

        t1 = time.perf_counter()
        rec_fp16 = deserialize_tensor(raw_fp16, device="cpu", upcast_fp32=True)
        t_deser_fp16_list.append((time.perf_counter() - t1) * 1000.0)

    fp16_bytes = len(raw_fp16)
    fp16_max_err = (sample_tensor - rec_fp16).abs().max().item()
    fp16_rel_err = ((sample_tensor - rec_fp16).abs() / (sample_tensor.abs() + 1e-7)).mean().item()

    # 4. Strict assertions
    assert 605000 <= fp32_bytes <= 610000, f"FP32 byte payload {fp32_bytes} outside expected range"
    assert 302000 <= fp16_bytes <= 306000, f"FP16 byte payload {fp16_bytes} outside expected range"
    assert rec_fp16.shape == (1, 197, 768), f"Wrong reconstructed shape: {rec_fp16.shape}"
    assert rec_fp16.dtype == torch.float32, f"Reconstructed tensor not upcast to float32: {rec_fp16.dtype}"
    assert fp32_max_err == 0.0, "FP32 roundtrip introduced non-zero error"
    assert fp16_max_err < 3.0e-3, f"FP16 quantization error {fp16_max_err} exceeds 3.0e-3 bound"

    results = {
        "FP32": {
            "bytes": fp32_bytes,
            "kb": fp32_bytes / 1024.0,
            "ser_ms": sum(t_ser_fp32_list) / len(t_ser_fp32_list),
            "deser_ms": sum(t_deser_fp32_list) / len(t_deser_fp32_list),
            "max_err": fp32_max_err,
            "rel_err": 0.0,
        },
        "FP16": {
            "bytes": fp16_bytes,
            "kb": fp16_bytes / 1024.0,
            "ser_ms": sum(t_ser_fp16_list) / len(t_ser_fp16_list),
            "deser_ms": sum(t_deser_fp16_list) / len(t_deser_fp16_list),
            "max_err": fp16_max_err,
            "rel_err": fp16_rel_err,
        },
    }

    return results


def main():
    print("[INFO] Benchmarking Binary Tensor Serialization & FP16 Quantization (Sub-stage 2.1)...")
    results = benchmark_serialization(num_trials=50)

    fp32 = results["FP32"]
    fp16 = results["FP16"]
    reduction_pct = (1.0 - (fp16["bytes"] / fp32["bytes"])) * 100.0

    print("\n" + "=" * 80)
    print("SUB-STAGE 2.1: SERIALIZATION & QUANTIZATION COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Metric':<34} | {'FP32 Baseline (Paper)':<20} | {'FP16 Proposed (Ours)':<20}")
    print("-" * 80)
    rel_err_str = f"{fp16['rel_err'] * 100:.3f}%"
    reduction_str = f"{reduction_pct:.2f}% (Half Wire)"
    print(f"{'Wire Payload (Bytes)':<34} | {int(fp32['bytes']):<20,} | {int(fp16['bytes']):<20,}")
    print(f"{'Wire Payload (KB)':<34} | {fp32['kb']:<20.2f} | {fp16['kb']:<20.2f}")
    print(f"{'Payload Reduction (%)':<34} | {'0.00% (Baseline)':<20} | {reduction_str:<20}")
    print(f"{'Serialization Latency (Avg)':<34} | {fp32['ser_ms']:<17.2f} ms | {fp16['ser_ms']:<17.2f} ms")
    print(f"{'Deserialization Latency (Avg)':<34} | {fp32['deser_ms']:<17.2f} ms | {fp16['deser_ms']:<17.2f} ms")
    print(f"{'Max Absolute Deviation':<34} | {fp32['max_err']:<20.2e} | {fp16['max_err']:<20.2e}")
    print(f"{'Mean Relative Quantization Error':<34} | {'0.00%':<20} | {rel_err_str:<20}")
    print("=" * 80)

    print("\n[PASS] All shape, dtype upcasting, byte payload bounds, and error tolerance assertions passed.")
    print("[SUCCESS] Sub-stage 2.1 (serialization.py) completed successfully.")


if __name__ == "__main__":
    main()
