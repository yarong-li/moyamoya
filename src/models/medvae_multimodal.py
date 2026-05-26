from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn

from src.medvae_local import get_local_mvae_class


class FrozenMedVAEMidBlock2Encoder(nn.Module):
    """
    Extract a pooled feature vector from the frozen MedVAE encoder.mid.block_2.
    """

    def __init__(
        self,
        medvae_model_name: str = "medvae_4_1_3d",
        modality: str = "mri",
        existing_weight: Optional[str] = None,
        state_dict: bool = True,
        pooling: str = "avg",
    ):
        super().__init__()

        MVAE = get_local_mvae_class()
        self.mvae = MVAE(model_name=medvae_model_name, modality=modality)
        if existing_weight is not None:
            self.mvae.init_from_ckpt(existing_weight, state_dict=state_dict)

        self.mvae.requires_grad_(False)
        self.mvae.eval()

        block = self.mvae.model.encoder.mid.block_2
        self.feature_dim = int(getattr(block, "out_channels"))
        if pooling == "avg":
            self.pool = nn.AdaptiveAvgPool3d(1)
        elif pooling == "max":
            self.pool = nn.AdaptiveMaxPool3d(1)
        else:
            raise ValueError(f"Unsupported pooling: {pooling}")

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
        feat = self._encode_mid_block_2(x)
        pooled = self.pool(feat).flatten(1)
        return pooled


class BaseFusion(nn.Module):
    def __init__(self, modality_names: Sequence[str], feature_dim: int):
        super().__init__()
        self.modality_names = list(modality_names)
        self.feature_dim = int(feature_dim)

    @property
    def output_dim(self) -> int:
        raise NotImplementedError

    def forward(self, features: Dict[str, torch.Tensor], mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError

    def _stack_features(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        missing = [name for name in self.modality_names if name not in features]
        if missing:
            raise KeyError(f"Missing modality features for: {missing}")
        return torch.stack([features[name] for name in self.modality_names], dim=1)


class ConcatFusion(BaseFusion):
    @property
    def output_dim(self) -> int:
        return len(self.modality_names) * self.feature_dim

    def forward(self, features: Dict[str, torch.Tensor], mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return torch.cat([features[name] for name in self.modality_names], dim=1)


class WeightedSumFusion(BaseFusion):
    def __init__(self, modality_names: Sequence[str], feature_dim: int):
        super().__init__(modality_names=modality_names, feature_dim=feature_dim)
        self.logits = nn.Parameter(torch.zeros(len(self.modality_names)))

    @property
    def output_dim(self) -> int:
        return self.feature_dim

    def forward(self, features: Dict[str, torch.Tensor], mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        stacked = self._stack_features(features)
        weights = torch.softmax(self.logits, dim=0).view(1, -1, 1)
        if mask is not None:
            weights = weights * mask.unsqueeze(-1)
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return (stacked * weights).sum(dim=1)


class AttentionFusion(BaseFusion):
    def __init__(
        self,
        modality_names: Sequence[str],
        feature_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__(modality_names=modality_names, feature_dim=feature_dim)
        if feature_dim % num_heads != 0:
            raise ValueError(
                f"feature_dim={feature_dim} must be divisible by num_heads={num_heads}."
            )
        self.attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(feature_dim)

    @property
    def output_dim(self) -> int:
        return self.feature_dim

    def forward(self, features: Dict[str, torch.Tensor], mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        stacked = self._stack_features(features)
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = mask <= 0
        attended, _ = self.attn(
            stacked,
            stacked,
            stacked,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        attended = self.norm(attended + stacked)
        if mask is None:
            return attended.mean(dim=1)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return (attended * mask.unsqueeze(-1)).sum(dim=1) / denom


class MLPClassificationHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: Optional[Iterable[int]] = None,
        dropout: float = 0.2,
    ):
        super().__init__()

        hidden_dims = [int(dim) for dim in (hidden_dims or []) if int(dim) > 0]
        dims: List[int] = [int(input_dim), *hidden_dims, int(num_classes)]

        layers: List[nn.Module] = []
        for idx in range(len(dims) - 1):
            in_dim = dims[idx]
            out_dim = dims[idx + 1]
            layers.append(nn.Linear(in_dim, out_dim))
            is_last = idx == len(dims) - 2
            if not is_last:
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_fusion_module(
    fusion_method: str,
    modality_names: Sequence[str],
    feature_dim: int,
    attention_num_heads: int = 4,
    attention_dropout: float = 0.1,
) -> BaseFusion:
    if fusion_method == "concat":
        return ConcatFusion(modality_names=modality_names, feature_dim=feature_dim)
    if fusion_method == "weighted_sum":
        return WeightedSumFusion(modality_names=modality_names, feature_dim=feature_dim)
    if fusion_method == "attention":
        return AttentionFusion(
            modality_names=modality_names,
            feature_dim=feature_dim,
            num_heads=attention_num_heads,
            dropout=attention_dropout,
        )
    raise ValueError(f"Unsupported fusion_method: {fusion_method}")


class MedVAEMultimodalClassifier(nn.Module):
    """
    Frozen MedVAE mid-block encoder per modality -> feature fusion -> classifier.
    """

    def __init__(
        self,
        modality_names: Sequence[str],
        num_classes: int,
        medvae_model_name: str = "medvae_4_1_3d",
        medvae_modality: str = "mri",
        existing_weight: Optional[str] = None,
        state_dict: bool = True,
        fusion_method: str = "concat",
        classifier_hidden_dims: Optional[Sequence[int]] = None,
        head_dropout: float = 0.2,
        encoder_pooling: str = "avg",
        attention_num_heads: int = 4,
        attention_dropout: float = 0.1,
    ):
        super().__init__()

        self.modality_names = list(modality_names)
        if len(self.modality_names) < 2:
            raise ValueError("MedVAEMultimodalClassifier expects at least two modalities.")

        self.encoders = nn.ModuleDict(
            {
                name: FrozenMedVAEMidBlock2Encoder(
                    medvae_model_name=medvae_model_name,
                    modality=medvae_modality,
                    existing_weight=existing_weight,
                    state_dict=state_dict,
                    pooling=encoder_pooling,
                )
                for name in self.modality_names
            }
        )

        feature_dims = {name: encoder.feature_dim for name, encoder in self.encoders.items()}
        if len(set(feature_dims.values())) != 1:
            raise ValueError(f"All modality encoders must share the same feature dim, got {feature_dims}")
        self.feature_dim = next(iter(feature_dims.values()))

        self.fusion = build_fusion_module(
            fusion_method=fusion_method,
            modality_names=self.modality_names,
            feature_dim=self.feature_dim,
            attention_num_heads=attention_num_heads,
            attention_dropout=attention_dropout,
        )

        if classifier_hidden_dims is None:
            if fusion_method == "concat":
                classifier_hidden_dims = [self.fusion.output_dim]
            else:
                classifier_hidden_dims = [self.feature_dim * 2]

        self.classifier = MLPClassificationHead(
            input_dim=self.fusion.output_dim,
            num_classes=num_classes,
            hidden_dims=classifier_hidden_dims,
            dropout=head_dropout,
        )
        self._debug_forward_printed = False

        print(
            "[MedVAEMultimodalClassifier] initialized | "
            f"modalities={self.modality_names} num_classes={num_classes} "
            f"feature_dim={self.feature_dim} fusion_method={fusion_method} "
            f"classifier_hidden_dims={list(classifier_hidden_dims)} existing_weight={existing_weight}"
        )

    def forward(self, batch) -> torch.Tensor:
        if not isinstance(batch, dict):
            raise TypeError("MedVAEMultimodalClassifier expects a dict input with 'images'.")

        images = batch.get("images")
        if not isinstance(images, dict):
            raise KeyError("Expected batch['images'] to be a modality->tensor dict.")
        mask = batch.get("mask")

        modality_features = {
            name: self.encoders[name](images[name])
            for name in self.modality_names
        }
        fused = self.fusion(modality_features, mask=mask)
        logits = self.classifier(fused)

        if not self._debug_forward_printed:
            shape_map = {name: tuple(feat.shape) for name, feat in modality_features.items()}
            print(
                "[MedVAEMultimodalClassifier] first forward | "
                f"feature_shapes={shape_map} fused_shape={tuple(fused.shape)} "
                f"logits_shape={tuple(logits.shape)}"
            )
            self._debug_forward_printed = True

        return logits
