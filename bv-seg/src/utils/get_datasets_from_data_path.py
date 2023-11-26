f"""

"""
import os

from pathlib import Path

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
