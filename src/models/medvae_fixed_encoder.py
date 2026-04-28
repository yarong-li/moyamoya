import os
import sys
from typing import Optional

import torch
import torch.nn as nn


def _ensure_medvae_importable():
    """
    Make local ./MedVAE/ importable as `medvae` without requiring installation.
    """
    try:
        import medvae  # noqa: F401
        return
    except Exception:
        pass

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    local_medvae_root = os.path.join(project_root, "MedVAE")
    if os.path.isdir(local_medvae_root) and local_medvae_root not in sys.path:
        sys.path.insert(0, local_medvae_root)


class MedVAEFixedEncoderClassifier(nn.Module):
    """
    3D MRI -> (frozen) MedVAE encoder -> latent -> classifier -> logits

    Notes:
    - Uses official `MVAE` wrapper for latent extraction.
    - Keeps MedVAE frozen and always in eval() mode.
    - `sample_posterior` is kept only for API compatibility with the previous
      implementation. In the current MVAE.encode path it is not used.
    """

    def __init__(
        self,
        num_classes: int,
        medvae_model_name: str = "medvae_4_1_3d",
        modality: str = "mri",
        existing_weight: Optional[str] = None,
        state_dict: bool = True,
        sample_posterior: bool = False,
        head_dropout: float = 0.2,
        head_hidden_mult: int = 2,
    ):
        super().__init__()

        _ensure_medvae_importable()
        from medvae import MVAE 

        self.mvae = MVAE(model_name=medvae_model_name, modality=modality)
        if existing_weight is not None:
            self.mvae.init_from_ckpt(existing_weight, state_dict=state_dict)

        # Kept for backward compatibility with old constructor signature.
        # Not consumed by MVAE.encode in current MedVAE implementation.
        self.sample_posterior = bool(sample_posterior)

        # freeze MedVAE parameters
        self.mvae.requires_grad_(False)
        self.mvae.eval()

        embed_dim = int(getattr(self.mvae.model, "embed_dim"))
        hidden = max(embed_dim * int(head_hidden_mult), embed_dim)

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(head_dropout),
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(hidden, num_classes),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.mvae.eval()
        return self

    def _encode_latent(self, x: torch.Tensor) -> torch.Tensor:
        # Official MedVAE entry: MVAE.encode(...)
        # For 3D models this path does not expose a sample/mode switch.
        with torch.no_grad():
            latent = self.mvae.encode(x)
        return latent

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self._encode_latent(x)  # [B, C(=embed_dim), d, h, w]
        feat = self.pool(latent)         # [B, C, 1, 1, 1]
        logits = self.head(feat)         # [B, num_classes]
        return logits

