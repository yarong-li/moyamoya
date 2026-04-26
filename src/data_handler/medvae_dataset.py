from src.data_handler.dataset import MedicalImageDataset


class MedVAEDataset(MedicalImageDataset):
    def __init__(self, image_paths, labels, medvae_transform, enable_augmentation=True, return_dict=True):
        super().__init__(
            image_paths=image_paths,
            labels=labels,
            normalize=False,
            enable_augmentation=enable_augmentation,
            transform=medvae_transform,
            return_dict=return_dict,
        )
