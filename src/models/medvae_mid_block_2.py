from typing import Optional

import torch
import torch.nn as nn

from src.medvae_local import get_local_mvae_class


class MedVAEMidBlock2Classifier(nn.Module):
    """
    3D MRI -> (frozen) MedVAE encoder.mid.block_2 -> classifier -> logits

    Notes:
    - Uses the local MedVAE encoder internals directly to extract the
      `encoder.mid.block_2` feature map.
    - Keeps MedVAE frozen and always in eval() mode.
    - `sample_posterior` is kept only for API compatibility with the previous
      implementation. It is not used in this mid-feature classifier.
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

        MVAE = get_local_mvae_class()

        self.mvae = MVAE(model_name=medvae_model_name, modality=modality)
        if existing_weight is not None:
            self.mvae.init_from_ckpt(existing_weight, state_dict=state_dict)

        # Kept for backward compatibility with old constructor signature.
        self.sample_posterior = bool(sample_posterior)

        self.mvae.requires_grad_(False)
        self.mvae.eval()

        block = self.mvae.model.encoder.mid.block_2
        embed_dim = int(getattr(block, "out_channels"))
        self._debug_forward_printed = False

        # Version 1
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

        # # Version 2
        # self.avg_pool = nn.AdaptiveAvgPool3d(1)
        # self.max_pool = nn.AdaptiveMaxPool3d(1)

        # pooled_dim = embed_dim * 2
        # hidden = max(pooled_dim * int(head_hidden_mult), pooled_dim)

        # self.head = nn.Sequential(
        #     nn.Flatten(),
        #     nn.LayerNorm(pooled_dim),
        #     nn.Dropout(head_dropout),
        #     nn.Linear(pooled_dim, hidden),
        #     nn.GELU(),
        #     nn.LayerNorm(hidden),
        #     nn.Dropout(head_dropout),
        #     nn.Linear(hidden, num_classes),
        # )

        print(
            "[MedVAEMidBlock2Classifier] initialized | "
            f"medvae_model_name={medvae_model_name} modality={modality} "
            f"num_classes={num_classes} mid_block_2_channels={embed_dim} "
            f"head_hidden={hidden} existing_weight={existing_weight}"
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.mvae.eval()
        return self

    def _encode_mid_block_2(self, x: torch.Tensor) -> torch.Tensor:
        encoder = self.mvae.model.encoder
        captured = {}

        def _capture_output(_module, _inputs, output):
            captured["mid_block_2"] = output

        hook = encoder.mid.block_2.register_forward_hook(_capture_output)
        try:
            with torch.no_grad():
                _ = encoder(x)
        finally:
            hook.remove()

        feat = captured.get("mid_block_2")
        if feat is None:
            raise RuntimeError("Failed to capture MedVAE encoder.mid.block_2 output.")
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self._encode_mid_block_2(x)  # [B, C, d, h, w]

        # Version 1
        pooled = self.pool(feat)            # [B, C, 1, 1, 1]
        logits = self.head(pooled)          # [B, num_classes]

        # Version 2
        # avg_feat = self.avg_pool(feat)
        # max_feat = self.max_pool(feat)

        # pooled = torch.cat([avg_feat, max_feat], dim=1)

        # logits = self.head(pomeme
        # oled)
        
        # Debugging output
        if not self._debug_forward_printed:
            print(
                "[MedVAEMidBlock2Classifier] first forward | "
                f"input_shape={tuple(x.shape)} "
                f"mid_block_2_shape={tuple(feat.shape)} "
                f"pooled_shape={tuple(pooled.shape)} logits_shape={tuple(logits.shape)} "
                f"input_dtype={x.dtype} logits_dtype={logits.dtype}"
            )
            self._debug_forward_printed = True
        
        return logits
        
