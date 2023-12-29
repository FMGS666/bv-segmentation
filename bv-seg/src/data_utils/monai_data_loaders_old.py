f"""

"""
import os

from monai.data import (
    ThreadDataLoader,
    CacheDataset,
    set_track_meta,
    load_decathlon_datalist
)


def create_data_loaders_from_splits_metadata(
        splits_metadata_path,
        train_transforms,
        val_transforms,
        train_batch_size = 1,
        validation_batch_size = 1,
    ) -> dict[str, dict[int, dict[str, ThreadDataLoader]]]:
    """
    
    """
    result_dictionary = dict()
    dataset_names = os.listdir(splits_metadata_path)
    for dataset_id, dataset_name in enumerate(dataset_names):
        dataset_results = dict()
        dataset_path = os.path.join(
            splits_metadata_path, 
            dataset_name
        )
        splits = os.listdir(dataset_path)
        for split_id, split in enumerate(splits):
            split_metadata_path = os.path.join(
                dataset_path, 
                split
            )
            print(f"{dataset_name=}, {split_id=} {split_metadata_path=}")
            train_files = load_decathlon_datalist(split_metadata_path, True, "training", base_dir = "./")
            train_ds = CacheDataset(
                data=train_files, transform=train_transforms, cache_num=24, cache_rate=1.0, num_workers=2
            )
            train_loader = ThreadDataLoader(train_ds, num_workers=0, batch_size=1, shuffle=True)

            val_files = load_decathlon_datalist(split_metadata_path, True, "validation", base_dir = "./")
            val_ds = CacheDataset(
                data=val_files, transform=val_transforms, cache_num=6, cache_rate=1.0, num_workers=2
            )
            val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=validation_batch_size)
            yield dataset_id, split_id, train_loader, val_loader