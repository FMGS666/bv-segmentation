f"""

file: {__file__}

Contents:

* `retrieve_filenames_from_split_indexes`

Function that can be used to perform the k-fold split on a `Tif3DVolumeIterableFolder`.
It will retrieve the file names from a generator containing the indeces of 
the `training` and `validation` elements of each split.

Example of use: 

```
>>> from src.file_utils.tif_iterable_folder import Tif3DVolumeIterableFolder
>>> from src.file_utils.tif_file_loader import TifFileLoader
>>>
>>> kidney_1_dense_path = "./data/train/kidney_1_dense"
>>> kidney_1_dense_iterable_folder = Tif3DVolumeIterableFolder(kidney_1_dense_path, "train")
>>> kidney_1_dense_splits = k_fold_split_iterable_folder(
>>>        kidney_1_dense_iterable_folder,
>>>        K = 3,
>>>        random_state = None,
>>>        shuffle = False
>>> ) # This object will contain the generator with the indexes
>>> kidiney_1_dense_splits = retrieve_filenames_from_split_indexes(
>>>     kidney_1_dense_iterable_folder, 
>>>     kidney_1_dense_splits
>>> ) # This object will contain the actual paths
```


"""

from typing import Iterable, Callable, Generator, Any

from ..file_loaders.tif_iterable_folder import Tif3DVolumeIterableFolder

def retrieve_filenames_from_split_indexes(
        dataset_folder: Tif3DVolumeIterableFolder,
        splits_generator: Generator
    ) -> dict[int, dict[str, list[str]]]:
    """
    
    Arguments:
        * `dataset_folder: Tif3DVolumeIterableFolder` -> the iterable folder of the volume to split
        * `splits_generator: Generator` -> the generator containing the indexes for each split
    
    Returns: 
        * `dict` -> dictionary containing the training and validation paths for each split
            - `keys()`: `int` -> the id of the split
            - `values()`: `dict` -> the dictionary containing the paths of the split
                - `keys()`: `str` -> mode of the split (i.e.: "train" or "validation")
                - `values()`: `dict` -> dictionary containing the paths to the images and masks
                    - `keys()`: `str` -> whether the paths are images or labels (i.e.: "images" or "labels")
                    - `values()`: `list[str]` -> list containing the path to the `.tif` file

    """
    splits_dictionary = dict()
    slice_masks_paths = dataset_folder.slice_masks_paths
    slice_images_paths = dataset_folder.slice_images_paths
    for fold_id, (training_indexes, validation_indexes) in enumerate(splits_generator):
        train_images_paths = [
            slice_images_paths[training_index]
            for training_index in training_indexes
        ]
        train_masks_paths = [
            slice_masks_paths[training_index]
            for training_index in training_indexes 
        ]
        validation_images_paths = [
            slice_images_paths[validation_index]
            for validation_index in validation_indexes
        ]
        validation_masks_paths = [
            slice_masks_paths[validation_index]
            for validation_index in validation_indexes 
        ]
        train_paths = list(zip(train_images_paths, train_masks_paths))
        validation_paths = list(zip(validation_images_paths, validation_masks_paths))
        train_paths_list = []
        validation_paths_list = []
        for idx, (train_image_path, train_mask_path)  in enumerate(train_paths):
            data_dictionary = {
                "image": train_image_path,
                "label": train_mask_path
            }
            train_paths_list.append(data_dictionary)
        for idx, (validation_image_path, validation_mask_path)  in enumerate(validation_paths):
            data_dictionary = {
                "image": validation_image_path,
                "label": validation_mask_path
            }
            validation_paths_list.append(data_dictionary)
        splits_dictionary[fold_id] = {
            "training": train_paths_list, 
            "validation": validation_paths_list
        }
    return splits_dictionary
