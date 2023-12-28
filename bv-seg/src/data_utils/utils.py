f"""

"""
import json
import os
from pathlib import Path

def dump_dataset_metadata(
        metadata_dir: str,
        split_id: int,
        training_paths: list[dict[str, str]],
        validation_paths: list[dict[str, str]],
    ) -> None:
    """
    
    """
    dataset_json = {
        "labels": {
            "0": "background",
            "1": "blood vessel",
        },
        "tensorImageSize": "3D",
        "training": training_paths,
        "validation": validation_paths
    }
    split_path = os.path.join(metadata_dir, f"#{split_id}.json")
    with open(split_path, 'w') as outfile:
        json.dump(dataset_json, outfile)
        
def get_datasets_from_data_path(
        data_path: str | Path
    ) -> dict[str, str | Path]:
    """
    
    """
    datasets_paths = dict()
    for dataset in os.listdir(data_path):
        dataset_path = os.path.join(
            data_path,
            dataset
        )
        datasets_paths[dataset] = dataset_path
    return datasets_paths

def get_volumes_fold_splits(
        train_volumes_directory: str | Path
    ) -> dict:
    """
    
    """
    splits = os.listdir(train_volumes_directory)
    splits_path = [
        os.path.join(
            train_volumes_directory,
            split_path
        )
        for split_path in splits
    ]
    splits_dictionary = dict()
    for idx, val_split_path in enumerate(splits_path):
        validation_images_paths = os.path.join(
            val_split_path, 
            "images"
        )
        validation_masks_paths = os.path.join(
            val_split_path, 
            "masks"
        )
        train_splits_paths = [
            train_split_path for idx_, train_split_path in enumerate(splits_path)
            if idx_ != idx
        ]
        train_images_split_path = [
            os.path.join(
                train_split_path, 
                "images"
            )
            for train_split_path in train_splits_paths
        ]
        train_masks_split_path = [
            os.path.join(
                train_split_path, 
                "masks"
            )
            for train_split_path in train_splits_paths
        ]
        train_paths = [
            {
                "image": os.path.join(
                    train_image_path, 
                    volume_id
                ),
                "label": os.path.join(
                    train_masks_split_path[idx],
                    volume_id
                )
            }
            for idx, train_image_path in enumerate(train_images_split_path)
            for volume_id in os.listdir(train_image_path)
        ]
        val_paths = [
            {
                "image": os.path.join(
                    validation_images_paths,
                    validation_image_path
                ),
                "label": os.path.join(
                    validation_masks_paths,
                    validation_image_path
                )
            }
            for idx, validation_image_path in enumerate(os.listdir(validation_images_paths))
        ]
        splits_dictionary[idx] = {
            "training": train_paths,
            "validation": val_paths
        }
    return splits_dictionary

