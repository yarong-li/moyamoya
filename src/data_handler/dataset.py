import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
from typing import Callable, Optional

class MedicalImageDataset(Dataset):
    def __init__(
        self,
        image_paths,
        labels,
        normalize=False,
        clip_percentiles=(1, 99),
        enable_augmentation=True,
        transform: Optional[Callable[[str], torch.Tensor]] = None,
        return_dict: bool = False,
    ):
        # Data augmentation: duplicate samples with label > 1 and mark for flipping
        self.image_paths = []
        self.labels = []
        self.flip_flags = []  # True if this sample should be flipped
        
        for path, label in zip(image_paths, labels):
            # Add original sample
            self.image_paths.append(path)
            # ----- Multi-class (original): uncomment to restore multi-class -----
            # self.labels.append(label)
            # ----- Binary classification (label<=1 vs label>1): 0 = label<=1, 1 = label>1 -----
            self.labels.append(1 if label > 1 else 0)
            self.flip_flags.append(False)
            
            # Add flipped version for labels > 1 (only if augmentation is enabled)
            if enable_augmentation and label > 1:
                self.image_paths.append(path)
                # ----- Multi-class (original) -----
                # self.labels.append(label)
                # ----- Binary -----
                self.labels.append(1 if label > 1 else 0)
                self.flip_flags.append(True)
        
        self.normalize = normalize
        self.clip_percentiles = clip_percentiles
        self.transform = transform
        self.return_dict = return_dict

    def __len__(self):
        return len(self.image_paths)

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        # 防 NaN/inf
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = (x - x.mean()) / (x.std() + 1e-8)
        return x.astype(np.float32, copy=False)

    def __getitem__(self, idx):
        fpath = self.image_paths[idx]
        label = self.labels[idx]
        should_flip = self.flip_flags[idx]

        # For MedVAE path-based transforms, image generation happens here.
        if self.transform is not None:
            x = self.transform(fpath)
            if not isinstance(x, torch.Tensor):
                raise TypeError("transform must return torch.Tensor")
            x = x.to(dtype=torch.float32)
        else:
            # 1 load + canonical orientation (RAS)
            # nii = nib.as_closest_canonical(nib.load(fpath))
            nii = nib.load(fpath)
            x = nii.get_fdata(dtype=np.float32)  # (D,H,W)

            # 2 optional normalize
            if self.normalize:
                x = self._normalize(x)

            # 3 to tensor: [C, D, H, W]
            x = torch.from_numpy(x).unsqueeze(0)  # float32

        # Data augmentation: flip on tensor domain to support all transforms.
        if should_flip:
            x = torch.flip(x, dims=(-1,))

        y = torch.tensor(label, dtype=torch.long)

        if self.return_dict:
            return {"image": x, "label": y}
        return x, y
