import os
import sys
from typing import Callable, Optional, Sequence

import torch
import torch.nn.functional as F


def _ensure_medvae_importable():
    try:
        import medvae  # noqa: F401
        return
    except Exception:
        pass

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    local_medvae_root = os.path.join(project_root, "MedVAE")
    if os.path.isdir(local_medvae_root) and local_medvae_root not in sys.path:
        sys.path.insert(0, local_medvae_root)


def build_medvae_transform(
    model_name: str = "medvae_4_1_3d",
    modality: str = "mri",
    target_size: Optional[Sequence[int]] = (80, 64, 64),
) -> Callable[[str], torch.Tensor]:
    """
    Build a path-based transform for MedVAE that calls MVAE.apply_transform(fpath).
    Output shape per sample: [1, D, H, W], value range expected by MedVAE: [-1, 1].
    """
    _ensure_medvae_importable()
    from medvae import MVAE  # type: ignore

    mvae = MVAE(model_name=model_name, modality=modality)

    def medvae_transform(fpath: str) -> torch.Tensor:
        x = mvae.apply_transform(fpath)  # [1, 1, D, H, W]
        if not isinstance(x, torch.Tensor):
            raise TypeError("MVAE.apply_transform must return torch.Tensor")
        if x.ndim == 5 and x.shape[0] == 1:
            x = x.squeeze(0)  # [1, D, H, W]
        x = x.to(dtype=torch.float32)

        # MedVAE upstream 3D loader uses foreground crop and can produce variable
        # spatial sizes. Force a fixed size for DataLoader collation.
        if target_size is not None:
            if len(target_size) != 3:
                raise ValueError("target_size must be 3 integers like (D, H, W)")
            x = F.interpolate(
                x.unsqueeze(0),  # [1, 1, D, H, W]
                size=tuple(int(v) for v in target_size),
                mode="trilinear",
                align_corners=False,
            ).squeeze(0)

        # Keep MedVAE expected input range.
        x = torch.clamp(x, -1.0, 1.0)
        return x

    return medvae_transform
