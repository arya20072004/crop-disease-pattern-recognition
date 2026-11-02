"""
multimodal.py
=============
Experiment C: Multimodal Cross-Attention Architecture with Adaptive Visual Projection.
Adapts the visual projection layer W_cnn_proj from R^(256 x 512) to R^(256 x k),
preserving the 256-d attention space, 8 attention heads, BiLSTM weather context,
and metadata projection.
"""

from typing import Dict, Optional, Tuple, Any
import torch
import torch.nn as nn


class AdaptiveCrossAttentionFusion(nn.Module):
    """
    Multimodal cross-attention module adapted for k-dimensional visual embeddings.
    Preserves exact dimensions from increment_3_fusion.py:
    - cnn_dim: k (adapted from 512)
    - weather_dim: 256 (fixed from Weather BiLSTM)
    - meta_dim: 21 (fixed crop metadata)
    - attn_dim: 256
    - n_heads: 8
    - n_diseases: 26
    """
    def __init__(
        self,
        cnn_dim: int = 512,
        weather_dim: int = 256,
        meta_dim: int = 21,
        n_diseases: int = 26,
        attn_dim: int = 256,
        n_heads: int = 8,
        dropout: float = 0.3
    ):
        super().__init__()
        self.cnn_dim = cnn_dim
        self.weather_dim = weather_dim
        self.meta_dim = meta_dim
        self.attn_dim = attn_dim
        self.n_heads = n_heads
        self.n_diseases = n_diseases

        # Adapted Visual Projection: accepts k dimensions, projects to attn_dim (256)
        self.cnn_proj = nn.Linear(cnn_dim, attn_dim)
        
        # Weather Projection: projects BiLSTM context (256) to attn_dim (256)
        self.weather_proj = nn.Linear(weather_dim, attn_dim)

        # Multi-Head Cross-Attention: CNN queries weather context
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(attn_dim)

        # Self-Attention within CNN stream (captures inter-disease visual relationships)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.self_norm = nn.LayerNorm(attn_dim)

        # Metadata Projection: maps 21 metadata features to 64
        self.meta_proj = nn.Sequential(
            nn.Linear(meta_dim, 64),
            nn.LayerNorm(64),
            nn.GELU()
        )

        # Final Multi-stream Fusion Head: (256 + 256 + 64 = 576) -> 512 -> 256 -> 26
        fusion_dim = attn_dim + attn_dim + 64
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_diseases)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        cnn_feat: torch.Tensor,
        weather_feat: torch.Tensor,
        meta_feat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass:
        - cnn_feat: (B, k)
        - weather_feat: (B, 256)
        - meta_feat: (B, 21)
        Returns:
        - logits: (B, 26)
        - cross_weights: (B, 1, 1)
        """
        batch_size = cnn_feat.size(0)

        # Verify input tensor shapes
        if cnn_feat.size(1) != self.cnn_dim:
            raise ValueError(f"Visual feature dimension mismatch: expected {self.cnn_dim}, got {cnn_feat.size(1)}")
        if weather_feat.size(1) != self.weather_dim:
            raise ValueError(f"Weather dimension mismatch: expected {self.weather_dim}, got {weather_feat.size(1)}")
        if meta_feat.size(1) != self.meta_dim:
            raise ValueError(f"Metadata dimension mismatch: expected {self.meta_dim}, got {meta_feat.size(1)}")

        # 1. Project visual and weather streams to common attention dimension (256)
        # Adding seq_len=1 for multihead attention
        cnn_q = self.cnn_proj(cnn_feat).unsqueeze(1)         # (B, 1, 256)
        weather_k = self.weather_proj(weather_feat).unsqueeze(1) # (B, 1, 256)
        weather_v = weather_k                                # (B, 1, 256)

        # 2. Cross-Attention: CNN queries weather context
        cross_out, cross_weights = self.cross_attn(
            query=cnn_q,
            key=weather_k,
            value=weather_v
        )  # (B, 1, 256)
        cross_out = self.attn_norm(cross_out + cnn_q)        # Residual + LayerNorm
        cross_out = cross_out.squeeze(1)                     # (B, 256)

        # 3. Self-Attention within CNN stream
        self_out, _ = self.self_attn(
            query=cnn_q,
            key=cnn_q,
            value=cnn_q
        )  # (B, 1, 256)
        self_out = self.self_norm(self_out + cnn_q)          # Residual + LayerNorm
        self_out = self_out.squeeze(1)                       # (B, 256)

        # 4. Metadata stream projection
        meta_out = self.meta_proj(meta_feat)                 # (B, 64)

        # 5. Multimodal concatenation & fusion head
        fused = torch.cat([cross_out, self_out, meta_out], dim=1)  # (B, 576)
        logits = self.fusion_head(fused)                     # (B, 26)

        return logits, cross_weights

    def get_structure_summary(self) -> Dict[str, Any]:
        """Returns parameter count and tensor dimension specifications."""
        total_params = sum(p.numel() for p in self.parameters())
        cnn_proj_params = sum(p.numel() for p in self.cnn_proj.parameters())
        return {
            "model_type": "AdaptiveCrossAttentionFusion",
            "k_visual_dim": self.cnn_dim,
            "weather_dim": self.weather_dim,
            "meta_dim": self.meta_dim,
            "attn_dim": self.attn_dim,
            "n_heads": self.n_heads,
            "output_classes": self.n_diseases,
            "total_parameters": total_params,
            "cnn_proj_parameters": cnn_proj_params,
            "cnn_proj_weight_shape": list(self.cnn_proj.weight.shape),
            "fusion_input_dim": self.attn_dim + self.attn_dim + 64
        }

