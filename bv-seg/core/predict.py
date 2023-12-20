import torch

from monai.networks.nets import SwinUNETR

from ..src.data_utils.utils import get_volumes_fold_splits, get_datasets_from_data_path, dump_dataset_metadata
from ..src.feature_engineering.monai_transformations import get_monai_transformations


def predict(
        args,
        device
    ) -> None:
    test_data_path = args.test_data_path
    patch_size = args.patch_size

    
    test_datasets_paths = get_datasets_from_data_path(
        test_data_path
    )

    test_iterable_folders = {
        dataset_name: Tif3DVolumeIterableFolder(dataset_path, "test")
        for dataset_name, dataset_path in test_datasets_paths.items()
    }
    _, _, test_transforms = get_monai_transformations(
        patch_size
    )
    