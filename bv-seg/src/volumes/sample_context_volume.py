f"""

"""
import numpy as np
import torch

from ..file_loaders.tif_iterable_folder import Tif3DVolumeIterableFolder
from ..file_loaders.tif_file_loader import TifFileLoader

def sample_random_window_context_indexes(
        n_slices: int,
        context_length: int,
        n_samples: int
    ) -> list[tuple[int]]:
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
        iterable_folder: Tif3DVolumeIterableFolder,
        context_length: int = 50,
        n_samples: int = 50
    ) -> list[dict[str, torch.Tensor]]:
    assert iterable_folder.mode == "train"
    n_slices = len(iterable_folder)
    context_window_indexes = sample_random_window_context_indexes(
        n_slices, 
        context_length,
        n_samples
    )
    volumes = []
    for (l_slice_id, u_slice_id) in context_window_indexes:
        context_window_paths = iterable_folder[l_slice_id: u_slice_id]
        image_volume = []
        mask_volume = []
        for slice_id, image_path, mask_path in context_window_paths:
            image_file_loader = TifFileLoader(image_path)
            mask_file_loader = TifFileLoader(mask_path)
            image_tensor = image_file_loader.image_tensor
            mask_tensor = mask_file_loader.image_tensor
            image_volume.append(image_tensor)
            mask_volume.appens(mask_tensor)
        image_volume = torch.concat(image_volume)
        mask_volume = torch.concat(mask_volume)
        volumes.append(
            {
                "image": image_volume,
                "label": mask_volume
            }
        )
    return volumes
        


