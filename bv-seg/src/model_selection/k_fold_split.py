f"""

file: {__file__}

Contents:

* `k_fold_split_iterable_folder`

Function that can be used to perform the k-fold split on a `Tif3DVolumeIterableFolder`.
It will return a generator containing the indeces of the `training` and `validation`
elements of each split.

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
>>> )
>>> for fold_id, (training_indexes, validation_indexes) in enumerate(kidney_1_dense_splits):
>>>     ...
```


"""

from sklearn.model_selection import KFold
from typing import Iterable, Callable, Generator, Any

from ..file_loaders.tif_iterable_folder import Tif3DVolumeIterableFolder

def k_fold_split_iterable_folder(
        dataset_folder: Tif3DVolumeIterableFolder,
        K: int = 3,
        random_state: int | None = None,
        shuffle: bool = False
    ) -> Generator:
    """
    
    Arguments:
        * `dataset_folder: Tif3DVolumeIterableFolder` -> the iterable folder of the volume to split
        * `K: int` -> the number of splits
        * `random_state: int | None` -> the random state used to shuffle the slices when `shuffle == True`
        * `shuffle: bool` -> whether to shuffle the sliced or not
    
    Returns: 
        * `Generator` -> the generator containing the  `training` and `validation` indices of the elements of each split

    """
    slice_masks_paths = dataset_folder.slice_masks_paths
    slice_images_paths = dataset_folder.slice_images_paths
    k_folder = KFold(
        n_splits = K,
        random_state = random_state,
        shuffle = shuffle
    )
    return k_folder.split(
        slice_images_paths, 
        y = slice_masks_paths
    )
