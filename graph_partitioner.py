#!/usr/bin/env python3
"""
graph_partitioner.py — Sub-stage 1.0: Graph Partitioning Profiler
Evaluates parameter memory distribution vs. intermediate activation payload sizes
across candidate layer cut-points in ViT-Base (torchvision.models.vit_b_16).

Formulates the graph-cut objective matching Phase A of Parthasarathy & Krishnamachari (USC, 2023):
    Cost(k) = Activation_Bytes(k) + lambda * |Param_Memory_1(k) - Param_Memory_2(k)|
"""

import json
import sys
from typing import Dict, List, Tuple
import torch
import torchvision.models as models


def measure_vit_components() -> Tuple[int, List[int], int]:
    """
    Analytically measures the parameter counts of ViT-Base submodules:
      1. Base Embeddings: Conv patch projection, class token, positional embedding
      2. 12 Transformer Encoder Blocks: encoder.layers[0...11]
      3. Tail Head: Final LayerNorm + linear classifier head

    Returns:
        base_params (int): Total parameter count in initial embeddings.
        block_params (List[int]): Parameter counts for each of the 12 encoder blocks.
        tail_params (int): Total parameter count in final norm and head.
    """
    # Load model architecture offline (weights=None to avoid downloading checkpoints)
    model = models.vit_b_16(weights=None)

    # 1. Base embedding parameters (Conv2d patch proj + cls_token + pos_embedding)
    conv_proj_params = sum(p.numel() for p in model.conv_proj.parameters())
    class_token_params = model.class_token.numel()
    pos_embed_params = model.encoder.pos_embedding.numel()
    base_params = conv_proj_params + class_token_params + pos_embed_params

    # 2. Parameters per encoder block (12 blocks)
    block_params = [
        sum(p.numel() for p in layer.parameters())
        for layer in model.encoder.layers
    ]

    # 3. Final LayerNorm + classification head
    ln_params = sum(p.numel() for p in model.encoder.ln.parameters())
    head_params = sum(p.numel() for p in model.heads.parameters())
    tail_params = ln_params + head_params

    return base_params, block_params, tail_params


def evaluate_partition_candidates(
    base_params: int,
    block_params: List[int],
    tail_params: int,
    batch_size: int = 1,
    lambda_imbalance: float = 1.0,
) -> Tuple[List[Dict], int]:
    """
    Evaluates all candidate cut boundaries k in {0, ..., 10}.
    A cut at index k implies:
      - Stage 1 (Head / GPU): Base Embeddings + Encoder Blocks 0...k
      - Stage 2 (Tail / CPU): Encoder Blocks (k+1)...11 + Final LN + Head

    Intermediate Activation Shape:
      For ViT-Base: (B, N, D) = (batch_size, 197, 768)
      - FP32: 4 bytes/element
      - FP16: 2 bytes/element

    Returns:
        candidate_results (List[Dict]): Evaluation metrics per candidate boundary.
        optimal_cut_index (int): k* that minimizes Cost(k).
    """
    total_blocks = len(block_params)  # 12
    num_tokens = 197                  # 1 CLS token + (224/16)^2 = 196 patch tokens
    hidden_dim = 768                  # ViT-Base hidden embedding dimension

    act_elements = batch_size * num_tokens * hidden_dim
    act_bytes_fp32 = act_elements * 4  # 605,184 bytes
    act_bytes_fp16 = act_elements * 2  # 302,592 bytes

    candidate_results = []
    min_cost = float("inf")
    optimal_cut = -1

    # Candidate cut-points: after block k, where k in [0, 10] (11 valid cut points)
    for k in range(total_blocks - 1):
        stage1_blocks = list(range(0, k + 1))
        stage2_blocks = list(range(k + 1, total_blocks))

        stage1_p_count = base_params + sum(block_params[i] for i in stage1_blocks)
        stage2_p_count = tail_params + sum(block_params[j] for j in stage2_blocks)

        stage1_bytes = stage1_p_count * 4  # Assuming standard FP32 model weights
        stage2_bytes = stage2_p_count * 4

        mem_imbalance_bytes = abs(stage1_bytes - stage2_bytes)

        # Optimization objective: minimize activation transfer + weighted memory disparity
        cost = act_bytes_fp32 + (lambda_imbalance * mem_imbalance_bytes)

        candidate_results.append({
            "cut_block_index": k,
            "stage1_blocks": stage1_blocks,
            "stage2_blocks": stage2_blocks,
            "stage1_params": stage1_p_count,
            "stage2_params": stage2_p_count,
            "stage1_mem_mb": stage1_bytes / (1024 ** 2),
            "stage2_mem_mb": stage2_bytes / (1024 ** 2),
            "mem_imbalance_mb": mem_imbalance_bytes / (1024 ** 2),
            "activation_shape": [batch_size, num_tokens, hidden_dim],
            "activation_bytes_fp32": act_bytes_fp32,
            "activation_bytes_fp16": act_bytes_fp16,
            "cost": cost,
        })

        if cost < min_cost:
            min_cost = cost
            optimal_cut = k

    return candidate_results, optimal_cut


def print_evaluation_table(candidates: List[Dict], optimal_cut: int):
    """Prints a formatted evaluation table across all candidate partition cuts."""
    print("=" * 115)
    print(f"{'Cut (k)':^8} | {'Stage 1 Blocks':^16} | {'Stage 2 Blocks':^16} | {'Stage 1 (MB)':^14} | {'Stage 2 (MB)':^14} | {'Diff Mem (MB)':^14} | {'FP32 Act (KB)':^14} | {'Cost':^12}")
    print("-" * 115)
    for row in candidates:
        k = row["cut_block_index"]
        s1_str = f"0..{k} ({len(row['stage1_blocks'])})"
        s2_str = f"{k+1}..11 ({len(row['stage2_blocks'])})"
        s1_mb = f"{row['stage1_mem_mb']:.2f}"
        s2_mb = f"{row['stage2_mem_mb']:.2f}"
        diff_mb = f"{row['mem_imbalance_mb']:.2f}"
        act_kb = f"{row['activation_bytes_fp32'] / 1024:.1f}"
        cost_str = f"{row['cost']:.0f}"

        prefix = ">> " if k == optimal_cut else "   "
        marker = " (OPT)" if k == optimal_cut else ""
        print(f"{prefix}{k:^4}{marker:<4} | {s1_str:^16} | {s2_str:^16} | {s1_mb:^14} | {s2_mb:^14} | {diff_mb:^14} | {act_kb:^14} | {cost_str:^12}")
    print("=" * 115)


def main():
    print("[INFO] Profiling torchvision.models.vit_b_16 computational graph...")
    base_params, block_params, tail_params = measure_vit_components()

    print(f"[INFO] Base Embedding Parameters: {base_params:,} ({base_params * 4 / (1024**2):.2f} MB)")
    print(f"[INFO] Encoder Block Parameters (each): {block_params[0]:,} ({block_params[0] * 4 / (1024**2):.2f} MB)")
    print(f"[INFO] Tail Classification Head Parameters: {tail_params:,} ({tail_params * 4 / (1024**2):.2f} MB)")
    total_params = base_params + sum(block_params) + tail_params
    print(f"[INFO] Total ViT-Base Parameters: {total_params:,} ({total_params * 4 / (1024**2):.2f} MB)\n")

    candidates, optimal_cut = evaluate_partition_candidates(
        base_params=base_params,
        block_params=block_params,
        tail_params=tail_params,
        batch_size=1,
        lambda_imbalance=1.0,
    )

    print_evaluation_table(candidates, optimal_cut)

    opt = candidates[optimal_cut]
    print(f"\n[OPTIMAL CUT SELECTED]: Block {optimal_cut}")
    print(f"  - Stage 1 (GPU Head): Blocks {opt['stage1_blocks']} ({opt['stage1_mem_mb']:.2f} MB)")
    print(f"  - Stage 2 (CPU Tail): Blocks {opt['stage2_blocks']} ({opt['stage2_mem_mb']:.2f} MB)")
    print(f"  - Memory Disparity (|M1 - M2|): {opt['mem_imbalance_mb']:.4f} MB")
    print(f"  - Intermediate Activation Shape: {opt['activation_shape']}")
    print(f"  - FP32 Wire Payload: {opt['activation_bytes_fp32']} bytes ({opt['activation_bytes_fp32'] / 1024:.2f} KB)")
    print(f"  - FP16 Wire Payload: {opt['activation_bytes_fp16']} bytes ({opt['activation_bytes_fp16'] / 1024:.2f} KB)")

    # Save to partition_config.json
    config_data = {
        "cut_block_index": opt["cut_block_index"],
        "stage1_blocks": opt["stage1_blocks"],
        "stage2_blocks": opt["stage2_blocks"],
        "activation_shape": opt["activation_shape"],
        "activation_bytes_fp32": opt["activation_bytes_fp32"],
        "activation_bytes_fp16": opt["activation_bytes_fp16"],
        "stage1_param_count": opt["stage1_params"],
        "stage2_param_count": opt["stage2_params"],
        "stage1_mem_bytes": opt["stage1_params"] * 4,
        "stage2_mem_bytes": opt["stage2_params"] * 4,
    }

    config_filename = "partition_config.json"
    with open(config_filename, "w") as f:
        json.dump(config_data, f, indent=2)

    print(f"\n[SUCCESS] Configuration successfully exported to '{config_filename}'.")


if __name__ == "__main__":
    main()
