import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
from typing import Callable, Dict, List, Optional, Any

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


class PatientMedicalImageDataset(Dataset):
    """
    Patient-level dataset supporting:
      - mode="single": one selected modality, return x:[1,D,H,W]
      - mode="multi_stack": early fusion, return x:[C,D,H,W]
      - mode="multi_dict": late fusion, return dict of modality->tensor

    Backward compatibility:
      - when mode="single" and return_dict=False, returns exactly (x, y)
    """

    def __init__(
        self,
        patients_df,
        images_df,
        selected_image_keys,
        label_col: str = "label",
        patient_id_col: str = "patient_id",
        image_key_col: str = "image_type",
        path_col: str = "path",
        mode: str = "single",
        normalize: bool = False,
        transform: Optional[Callable[[str], torch.Tensor]] = None,
        label_fn: Optional[Callable[[Any], int]] = None,
        return_dict: bool = False,
        allow_missing: bool = False,
        enable_augmentation: bool = False,
    ):
        self.patients_df = patients_df.copy()
        self.images_df = images_df.copy()
        self.selected_image_keys = list(selected_image_keys)
        self.label_col = label_col
        self.patient_id_col = patient_id_col
        self.image_key_col = image_key_col
        self.path_col = path_col
        self.mode = mode
        self.normalize = normalize
        self.transform = transform
        self.label_fn = label_fn or (lambda y: 1 if y > 1 else 0)
        self.return_dict = return_dict
        self.allow_missing = allow_missing

        if mode not in {"single", "multi_stack", "multi_dict"}:
            raise ValueError(f"Unsupported mode: {mode}")
        if len(self.selected_image_keys) == 0:
            raise ValueError("selected_image_keys cannot be empty")
        if mode == "single" and len(self.selected_image_keys) != 1:
            raise ValueError("mode='single' requires exactly one selected image key")

        required_patient_cols = {self.patient_id_col, self.label_col}
        required_image_cols = {self.patient_id_col, self.image_key_col, self.path_col}
        missing_patient_cols = required_patient_cols - set(self.patients_df.columns)
        missing_image_cols = required_image_cols - set(self.images_df.columns)
        if missing_patient_cols:
            raise ValueError(f"patients_df missing columns: {sorted(missing_patient_cols)}")
        if missing_image_cols:
            raise ValueError(f"images_df missing columns: {sorted(missing_image_cols)}")

        self.patient_samples = self._build_patient_samples()

        if enable_augmentation:
            augmented_samples = []
            for sample in self.patient_samples:
                augmented_samples.append(sample)
                sample_flip = dict(sample)
                sample_flip["flip"] = True
                augmented_samples.append(sample_flip)
            self.patient_samples = augmented_samples

    def __len__(self):
        return len(self.patient_samples)

    def _normalize_np(self, x: np.ndarray) -> np.ndarray:
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = (x - x.mean()) / (x.std() + 1e-8)
        return x.astype(np.float32, copy=False)

    def _load_one_image(self, path: str) -> torch.Tensor:
        if self.transform is not None:
            x = self.transform(path)
            if not isinstance(x, torch.Tensor):
                raise TypeError("transform must return torch.Tensor")
            x = x.to(dtype=torch.float32)
            # Keep consistent shape [1,D,H,W]
            if x.ndim == 3:
                x = x.unsqueeze(0)
            if x.ndim != 4:
                raise ValueError(f"Expected transformed image shape [1,D,H,W] or [D,H,W], got {tuple(x.shape)}")
            return x

        nii = nib.load(path)
        arr = nii.get_fdata(dtype=np.float32)
        if self.normalize:
            arr = self._normalize_np(arr)
        x = torch.from_numpy(arr).unsqueeze(0).to(dtype=torch.float32)
        return x

    def _build_patient_samples(self) -> List[Dict[str, Any]]:
        filtered_images = self.images_df[self.images_df[self.image_key_col].isin(self.selected_image_keys)].copy()

        # Build mapping: patient_id -> {image_key: path}
        patient_to_images: Dict[str, Dict[str, str]] = {}
        grouped = filtered_images.groupby([self.patient_id_col, self.image_key_col], sort=False)
        for (pid, image_key), df_group in grouped:
            df_group = df_group.sort_values(self.path_col)
            # If duplicate rows exist for same patient+image_key, choose the first deterministically
            chosen_path = str(df_group.iloc[0][self.path_col])
            if pid not in patient_to_images:
                patient_to_images[pid] = {}
            patient_to_images[pid][image_key] = chosen_path

        samples: List[Dict[str, Any]] = []
        for _, row in self.patients_df.iterrows():
            pid = row[self.patient_id_col]
            if pid not in patient_to_images:
                continue

            image_map = patient_to_images[pid]
            missing_keys = [k for k in self.selected_image_keys if k not in image_map]
            if (not self.allow_missing) and len(missing_keys) > 0:
                continue
            if self.allow_missing and len(missing_keys) == len(self.selected_image_keys):
                # Skip patients with no selected modalities at all
                continue

            sample = {
                "patient_id": pid,
                "label": int(self.label_fn(row[self.label_col])),
                "image_paths": {k: image_map.get(k, None) for k in self.selected_image_keys},
                "flip": False,
            }
            samples.append(sample)

        return samples

    def __getitem__(self, idx):
        sample = self.patient_samples[idx]
        patient_id = sample["patient_id"]
        label = torch.tensor(sample["label"], dtype=torch.long)
        should_flip = sample["flip"]

        loaded_images: Dict[str, Optional[torch.Tensor]] = {}
        mask_vals: List[float] = []
        reference_shape = None

        for key in self.selected_image_keys:
            path = sample["image_paths"][key]
            if path is None:
                loaded_images[key] = None
                mask_vals.append(0.0)
                continue

            x = self._load_one_image(path)
            if should_flip:
                x = torch.flip(x, dims=(-1,))
            loaded_images[key] = x
            mask_vals.append(1.0)
            if reference_shape is None:
                reference_shape = x.shape

        mask = torch.tensor(mask_vals, dtype=torch.float32)

        if self.mode == "single":
            key = self.selected_image_keys[0]
            x = loaded_images[key]
            if x is None:
                raise ValueError(f"Missing required image '{key}' for patient {patient_id} in single mode.")
            if self.return_dict:
                return {"image": x, "label": label, "patient_id": patient_id, "image_key": key, "mask": mask}
            return x, label

        if self.mode == "multi_stack":
            if reference_shape is None:
                raise ValueError(f"No available images for patient {patient_id} in multi_stack mode.")
            channels = []
            for key in self.selected_image_keys:
                x = loaded_images[key]
                if x is None:
                    if not self.allow_missing:
                        raise ValueError(f"Missing required image '{key}' for patient {patient_id}.")
                    x = torch.zeros(reference_shape, dtype=torch.float32)
                channels.append(x)
            x_stack = torch.cat(channels, dim=0)
            if self.return_dict:
                return {
                    "image": x_stack,
                    "label": label,
                    "patient_id": patient_id,
                    "mask": mask,
                    "image_keys": list(self.selected_image_keys),
                }
            return x_stack, label

        # mode == "multi_dict"
        return {
            "images": loaded_images,
            "mask": mask,
            "label": label,
            "patient_id": patient_id,
        }
