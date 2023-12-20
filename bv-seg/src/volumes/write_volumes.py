f"""

sampling random context window from each group
for creating 3D volumes to feed to the model
"""
import gc
from .sample_context_volume import sample_random_window_context, save_context_volumes_to_nii_gz

def write_volumes_to_tif(
        train_splits_groups: dict,
        context_length: int,
        n_samples: int,
        subsample: bool = True,
        dump_folder: str = "./data/splits_sampled_volumes"
    ) -> None:
    train_splits_sample_volumes = {
        dataset_name: {
            split_id: sample_random_window_context(
                split,
                context_length = context_length,
                n_samples = n_samples,
                subsample = subsample
            )
            for split_id, split in split_groups.items()
        }
        for dataset_name, split_groups in train_splits_groups.items()
    }
    for dataset_name, splits in train_splits_sample_volumes.items():
        for splt_id, split_volumes in splits.items():
            save_context_volumes_to_nii_gz(
                dataset_name,
                splt_id,
                split_volumes,
                dump_folder = dump_folder
            )
            del split_volumes
            gc.collect()