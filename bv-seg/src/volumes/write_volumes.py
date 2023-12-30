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
        train: bool,
        subsample: bool,
        dump_folder: str = "./data/splits_sampled_volumes",
    ) -> None:
    for dataset_name, split_groups in train_splits_groups.items():
        for split_id, split in split_groups.items():
            print(f"Writing volumes of {dataset_name} dataset, {split_id=}")
            sample_random_window_context(
                dataset_name,
                split_id,
                split,
                context_length = context_length,
                n_samples = n_samples,
                subsample = subsample,
                train = train
            )