import glob
import json
import os


def dump_dataset_metadata(
        metadata_dir: str,
        split_id: int,
        training_paths: list[dict[str, str]],
        validation_paths: list[dict[str, str]],
    ) -> None:
    dataset_json = {
        "labels": {
            "0": "background",
            "1": "blood vessel",
        },
        "tensorImageSize": "2D",
        "training": training_paths,
        "validation": validation_paths
    }
    split_path = os.path.join(metadata_dir, f"#{split_id}.json")
    with open(split_path, 'w') as outfile:
        json.dump(dataset_json, outfile)