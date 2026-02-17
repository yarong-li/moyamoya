import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np

class MedicalImageDataset(Dataset):
    def __init__(self, image_paths, labels, normalize=False, clip_percentiles=(1, 99)):
        self.image_paths = image_paths
        self.labels = labels
        self.normalize = normalize
        self.clip_percentiles = clip_percentiles

    def __len__(self):
        return len(self.image_paths)

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        # 防 NaN/inf
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # 稳健裁剪 + z-score（可选）
        # lo, hi = np.percentile(x, self.clip_percentiles)
        # x = np.clip(x, lo, hi)
        x = (x - x.mean()) / (x.std() + 1e-8)
        return x.astype(np.float32, copy=False)

    def __getitem__(self, idx):
        path = self.image_paths[idx]

        # 1 load + canonical orientation (RAS)
        # nii = nib.as_closest_canonical(nib.load(path))
        nii = nib.load(path)
        x = nii.get_fdata(dtype=np.float32)  # (D,H,W)

        # 2 optional normalize
        if self.normalize:
            x = self._normalize(x)

        # 3 to tensor: [C, D, H, W]
        x = torch.from_numpy(x).unsqueeze(0)  # float32

        # 4 classification label: scalar long
        y = torch.tensor(self.labels[idx], dtype=torch.long)

        return x, y
