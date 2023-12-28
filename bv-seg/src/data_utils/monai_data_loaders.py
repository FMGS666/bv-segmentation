f"""

"""
import os
import json

from monai.data import (
    ThreadDataLoader,
    CacheDataset,
    load_decathlon_datalist
)

def merge_list_of_dictionaries(
        list_of_dictionaries
    ):
    merged_metadata_dict = list_of_dictionaries[0]
    for metadata_dict in list_of_dictionaries[1:]:
        merged_metadata_dict["training"] += metadata_dict["training"]
        merged_metadata_dict["validation"] += metadata_dict["validation"]
    return merged_metadata_dict

def merge_split_metadata_file(
        K, 
        metadata_base_path,
    ):
    current_split_metadata = []
    splits_metadata_path = os.path.join(
        metadata_base_path, 
        "individual_datasets"
    )
    splits_merged_paths = os.path.join(
        metadata_base_path, 
        "merged_splits"
    )
    dataset_names = os.listdir(splits_metadata_path)
    print(f"{K=}")
    for dataset_id, dataset_name in enumerate(dataset_names):
        print(f"{dataset_id=}, {dataset_name=}")
        dataset_path = os.path.join(
            splits_metadata_path, 
            dataset_name
        )
        splits = sorted(os.listdir(dataset_path))
        for split_id, split in enumerate(splits):
            print(f"{split_id=}, {split=}")
            if split_id == K:
                split_metadata_path = os.path.join(
                    dataset_path, 
                    split
                )
                with open(split_metadata_path, "r") as metadata_file:
                    split_metadata = json.load(metadata_file)
                print(f"{len(split_metadata['training'])=}")
                print(f"{len(split_metadata['validation'])=}")
                current_split_metadata.append(split_metadata)
    merged_metadata_split = merge_list_of_dictionaries(current_split_metadata)
    merged_metadata_split_path = os.path.join(
        splits_merged_paths,
        f"#{K}.json"
    )
    with open(merged_metadata_split_path, "w") as merged_metadata_file:
        json.dump(merged_metadata_split, merged_metadata_file)
    return merged_metadata_split_path

def create_data_loaders_from_splits_metadata(
        K,
        metadata_base_path,
        train_transforms,
        val_transforms,
        train_batch_size = 1,
        validation_batch_size = 1,
    ) -> dict[str, dict[int, dict[str, ThreadDataLoader]]]:
    """
    
    """
    split_merged_metadata_path = merge_split_metadata_file(
        K,
        metadata_base_path
    )
    train_files = load_decathlon_datalist(split_merged_metadata_path, True, "training", base_dir = "./")
    train_ds = CacheDataset(
        data=train_files, transform=train_transforms, cache_num=24, cache_rate=1.0, num_workers=2
    )
    train_loader = ThreadDataLoader(train_ds, num_workers=0, batch_size=train_batch_size, shuffle=True)

    val_files = load_decathlon_datalist(split_merged_metadata_path, True, "validation", base_dir = "./")
    val_ds = CacheDataset(
        data=val_files, transform=val_transforms, cache_num=6, cache_rate=1.0, num_workers=2
    )
    val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=validation_batch_size)
    return train_loader, val_loader