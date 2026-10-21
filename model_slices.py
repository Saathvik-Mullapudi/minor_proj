#!/usr/bin/env python3
"""
model_slices.py — Sub-stage 1.1: Model Slicing & Layer Extraction
Partitioning torchvision.models.vit_b_16 into modular ViTHead and ViTTail modules.

Architecture Cut Boundary (k=5):
  - ViTHead (Stage 1 / GPU):
      * Input: Raw RGB Image (B, 3, 224, 224)
      * Operations: Conv patch projection + class token prepend + pos embedding + Encoder Blocks 0..5
      * Output: Intermediate Activation Tensor (B, 197, 768)
  - ViTTail (Stage 2 / CPU):
      * Input: Intermediate Activation Tensor (B, 197, 768)
      * Operations: Encoder Blocks 6..11 + LayerNorm + CLS token extraction + Linear Head
      * Output: Class Logits (B, 1000)
"""

import json
import os
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torchvision.models as models


class ViTHead(nn.Module):
    """
    Stage 1 Partition of ViT-Base (targeted for GPU execution).
    Processes raw input images up through Encoder Block 5.
    Output: Intermediate activation tensor of shape (B, 197, 768).
    """

    def __init__(
        self,
        conv_proj: nn.Conv2d,
        class_token: nn.Parameter,
        pos_embedding: nn.Parameter,
        dropout: nn.Dropout,
        layers: nn.Sequential,
        image_size: int = 224,
        patch_size: int = 16,
        hidden_dim: int = 768,
    ):
        super().__init__()
        self.conv_proj = conv_proj
        self.class_token = class_token
        self.pos_embedding = pos_embedding
        self.dropout = dropout
        self.layers = layers
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass executing tokenization, positional encoding, and initial encoder blocks.

        Args:
            x (torch.Tensor): Input batch of shape (B, 3, 224, 224).

        Returns:
            torch.Tensor: Intermediate activation tensor of shape (B, 197, 768).
        """
        torch._assert(x.dim() == 4, f"Expected 4D image tensor (B, 3, H, W), got {x.shape}")
        b, c, h, w = x.shape
        torch._assert(
            h == self.image_size and w == self.image_size,
            f"Expected image resolution ({self.image_size}, {self.image_size}), got ({h}, {w})",
        )

        n_h = h // self.patch_size
        n_w = w // self.patch_size

        # 1. Patch projection: (B, 3, 224, 224) -> (B, 768, 14, 14)
        x = self.conv_proj(x)

        # 2. Reshape and permute: (B, 768, 14, 14) -> (B, 196, 768)
        x = x.reshape(b, self.hidden_dim, n_h * n_w).permute(0, 2, 1)

        # 3. Prepend class token: (B, 1, 768) concat (B, 196, 768) -> (B, 197, 768)
        batch_cls = self.class_token.expand(b, -1, -1)
        x = torch.cat([batch_cls, x], dim=1)

        # 4. Add positional embedding & apply dropout
        x = self.dropout(x + self.pos_embedding)

        # 5. Execute Stage 1 Transformer Encoder Blocks (0 through 5)
        # Tensor shape remains strictly (B, 197, 768) throughout all blocks
        x = self.layers(x)

        return x


class ViTTail(nn.Module):
    """
    Stage 2 Partition of ViT-Base (targeted for CPU execution).
    Accepts intermediate activation tensor from ViTHead, completes the remaining
    encoder blocks, applies LayerNorm, extracts the CLS token, and classifies.
    Output: Class logits tensor of shape (B, 1000).
    """

    def __init__(
        self,
        layers: nn.Sequential,
        ln: nn.LayerNorm,
        head: nn.Linear,
        hidden_dim: int = 768,
    ):
        super().__init__()
        self.layers = layers
        self.ln = ln
        self.head = head
        self.hidden_dim = hidden_dim

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Forward pass executing remaining encoder blocks, LayerNorm, and classification head.

        Args:
            h (torch.Tensor): Intermediate activation tensor of shape (B, 197, 768).

        Returns:
            torch.Tensor: Predicted class logits of shape (B, 1000).
        """
        torch._assert(h.dim() == 3, f"Expected 3D activation tensor (B, 197, 768), got {h.shape}")
        torch._assert(
            h.shape[2] == self.hidden_dim,
            f"Expected hidden dimension {self.hidden_dim}, got {h.shape[2]}",
        )

        # 1. Execute Stage 2 Transformer Encoder Blocks (6 through 11)
        # Tensor shape remains (B, 197, 768)
        h = self.layers(h)

        # 2. Final LayerNorm: (B, 197, 768) -> (B, 197, 768)
        h = self.ln(h)

        # 3. Extract CLS token at index 0: (B, 197, 768) -> (B, 768)
        cls_token = h[:, 0]

        # 4. Classification Head: (B, 768) -> (B, 1000)
        logits = self.head(cls_token)

        return logits


def load_partition_config(config_path: str = "partition_config.json") -> int:
    """Reads cut_block_index from configuration file or defaults to block 5."""
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = json.load(f)
            return cfg.get("cut_block_index", 5)
    return 5


def create_vit_slices(
    weights: Optional[str] = None,
    config_path: str = "partition_config.json",
) -> Tuple[ViTHead, ViTTail, models.VisionTransformer]:
    """
    Factory function that instantiates canonical ViT-Base and slices its components cleanly
    into ViTHead and ViTTail without duplicating or re-randomizing weights.

    Args:
        weights: Torchvision model weights (e.g., 'DEFAULT' or None for offline verification).
        config_path: Path to partition_config.json.

    Returns:
        head (ViTHead): Configured Head module.
        tail (ViTTail): Configured Tail module.
        ref_model (models.VisionTransformer): Canonical reference model for equivalence checks.
    """
    cut_block_index = load_partition_config(config_path)

    # Instantiate canonical reference model
    ref_model = models.vit_b_16(weights=weights)
    ref_model.eval()

    # Total encoder blocks is 12: 0..11
    # Head gets 0 .. cut_block_index (e.g. 0..5 = 6 blocks)
    # Tail gets cut_block_index + 1 .. 11 (e.g. 6..11 = 6 blocks)
    head_layers = nn.Sequential(*[ref_model.encoder.layers[i] for i in range(cut_block_index + 1)])
    tail_layers = nn.Sequential(*[ref_model.encoder.layers[j] for j in range(cut_block_index + 1, 12)])

    head = ViTHead(
        conv_proj=ref_model.conv_proj,
        class_token=ref_model.class_token,
        pos_embedding=ref_model.encoder.pos_embedding,
        dropout=ref_model.encoder.dropout,
        layers=head_layers,
        image_size=ref_model.image_size,
        patch_size=ref_model.patch_size,
        hidden_dim=ref_model.hidden_dim,
    ).eval()

    tail = ViTTail(
        layers=tail_layers,
        ln=ref_model.encoder.ln,
        head=ref_model.heads.head,
        hidden_dim=ref_model.hidden_dim,
    ).eval()

    return head, tail, ref_model


def main():
    print("[INFO] Testing ViT-Base model slicing (Sub-stage 1.1)...")
    head, tail, ref_model = create_vit_slices(weights=None)

    # 1. Parameter counts verification
    head_params = sum(p.numel() for p in head.parameters())
    tail_params = sum(p.numel() for p in tail.parameters())
    total_slice_params = head_params + tail_params
    ref_params = sum(p.numel() for p in ref_model.parameters())

    print(f"  - ViTHead Parameters: {head_params:,} ({head_params * 4 / (1024**2):.2f} MB)")
    print(f"  - ViTTail Parameters: {tail_params:,} ({tail_params * 4 / (1024**2):.2f} MB)")
    print(f"  - Combined Slices:    {total_slice_params:,} parameters")
    print(f"  - Canonical Model:    {ref_params:,} parameters")

    assert head_params == 43269888, f"Expected 43,269,888 params for Head, got {head_params}"
    assert tail_params == 43297768, f"Expected 43,297,768 params for Tail, got {tail_params}"
    assert total_slice_params == ref_params, "Slice parameter sum does not match monolithic ViT!"
    print("[PASS] Parameter count assertions passed.")

    # 2. Shape and numerical equivalence assertions
    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224)

    with torch.no_grad():
        ref_out = ref_model(x)
        h = head(x)
        tail_out = tail(h)

    print(f"  - Input shape:            {list(x.shape)}")
    print(f"  - Head output shape (h):  {list(h.shape)}")
    print(f"  - Tail output shape (y):  {list(tail_out.shape)}")

    assert h.shape == (batch_size, 197, 768), f"Unexpected head activation shape: {h.shape}"
    assert tail_out.shape == (batch_size, 1000), f"Unexpected tail logits shape: {tail_out.shape}"
    print("[PASS] Tensor shape assertions passed.")

    # 3. Exact mathematical parity check
    max_diff = (ref_out - tail_out).abs().max().item()
    print(f"  - Maximum numerical deviation |Ref - Slices|: {max_diff:.2e}")
    assert torch.allclose(ref_out, tail_out, atol=1e-6), "Numerical deviation exceeds atol=1e-6!"
    print("[PASS] Exact mathematical equivalence verified (Tail(Head(x)) == ViT(x)).")

    print("\n[SUCCESS] Sub-stage 1.1 (model_slices.py) self-test completed successfully.")


if __name__ == "__main__":
    main()
