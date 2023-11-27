f"""

"""
import numpy as np
import nibabel as nib
import os
import gc

from typing import Iterable
from ..file_loaders.tif_iterable_folder import Tif3DVolumeIterableFolder
from ..file_loaders.tif_file_loader import TifFileLoader

def sample_random_window_context_indexes(
        n_slices: int,
        context_length: int,
        n_samples: int
    ) -> list[tuple[int]]:
    """
    
    """
    context_centers = [idx for idx in range(context_length, n_slices - context_length)]
    sampled_context_centers = np.random.choice(context_centers, n_samples, replace=False)
    context_windows = [
        (
            context_center - context_length, 
            context_center + context_length
        )
        for context_center in sampled_context_centers
    ]
    return context_windows

def sample_random_window_context(
        iterable_to_sample: Iterable,
        context_length: int = 50,
        n_samples: int = 1
    ) -> list[dict[str, np.ndarray]]:
    """
    
    """
    n_slices = len(iterable_to_sample)
    context_window_indexes = sample_random_window_context_indexes(
        n_slices, 
        context_length,
        n_samples
    )
    volumes = []
    for (l_slice_id, u_slice_id) in context_window_indexes:
        context_window_paths = iterable_to_sample[l_slice_id: u_slice_id]
        image_volume = []
        mask_volume = []
        for paths in context_window_paths:
            image_path = paths["image"]
            mask_path = paths["label"]
            image_file_loader = TifFileLoader(image_path)
            mask_file_loader = TifFileLoader(mask_path)
            image_array = image_file_loader.image_array
            mask_array = mask_file_loader.image_array
            image_volume.append(image_array)
            mask_volume.append(mask_array)
        image_volume = np.stack(image_volume, axis = 0)
        mask_volume = np.stack(mask_volume, axis = 0)
        assert image_volume.ndim == 3, f"{image_volume.shape=}"
        assert mask_volume.ndim == 3, f"{mask_volume.shape=}"
        volumes.append(
            {
                "image": image_volume,
                "label": mask_volume
            }
        )
    return volumes
        
def save_context_volumes_to_nii_gz(
        dataset_name: str,
        split_id: int,
        volumes: list[dict[str, np.ndarray]],
        dump_folder: str = "./data/splits_sampled_volumes"
    ) -> None:
    dataset_folder = os.path.join(
        dump_folder,
        dataset_name
    )
    if not os.path.exists(dataset_folder):
        os.mkdir(dataset_folder)
    split_folder = os.path.join(
        dataset_folder,
        f"#{split_id}"
    )
    if not os.path.exists(split_folder):
        os.mkdir(split_folder)
    split_folder_images = os.path.join(
        split_folder, 
        "images"
    )
    split_folder_masks = os.path.join(
        split_folder, 
        "masks"
    )
    if not os.path.exists(split_folder_images):
        os.mkdir(split_folder_images)
    if not os.path.exists(split_folder_masks):
        os.mkdir(split_folder_masks)
    n_saved_volumes = len(os.listdir(split_folder_images))
    for idx, volumes in enumerate(volumes):
        image_volume = volumes["image"]
        mask_volume = volumes["label"]
        volume_id = f"volume_{idx + n_saved_volumes}.nii.gz"
        image_dump_path = os.path.join(
            split_folder_images, 
            volume_id
        )
        mask_dump_path = os.path.join(
            split_folder_masks, 
            volume_id
        )
        image_volume = nib.Nifti1Image(image_volume, None)
        mask_volume = nib.Nifti1Image(mask_volume, None)
        nib.save(image_volume, image_dump_path)
        nib.save(mask_volume, mask_dump_path)
        del image_volume, mask_volume
        gc.collect()
