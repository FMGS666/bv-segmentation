f"""

file: {__file__}

Contents:

* `Tif3DVolumeIterableFolder`

This class provides an interface for iterating over a
folder of one of the datasets of the competitions. 

Example of use: 

```
>>> from src.file_utils.tif_iterable_folder import Tif3DVolumeIterableFolder
>>>
>>> kidney_1_dense_path = "./data/train/kidney_1_dense" # training dataset
>>> kidney_3_dense_path = "./data/train/kidney_1_dense" # labels-only dataset
>>> kidney_6_path = "./data/test/kidney_6"              # test dataset
>>> kidney_1_dense_iterable_folder = Tif3DVolumeIterableFolder(kidney_1_dense_path, "train")
>>> kidney_3_dense_iterable_folder = Tif3DVolumeIterableFolder(kidney_3_dense_path, "labels-only")
>>> kidney_6_iterable_folder = Tif3DVolumeIterableFolder(kidney_6_path, "test")
>>> kidney_1_dense_slice_id, kidney_1_dense_image_path, kidney_1_dense_mask_path = kidney_1_dense_iterable_folder[0] 
>>> kidney_3_dense_slice_id, _, kidney_3_dense_mask_path = kidney_3_dense_iterable_folder[0] 
>>> kidney_6_slice_id, kidney_6_mask_path, _ = kidney_6_iterable_folder[0] 

```

"""

import os
import re

from pathlib import Path

class Tif3DVolumeIterableFolder(object):
    def __init__(
            self,
            volume_folder: str | Path,
            mode: str
        ) -> None:
        """
        Arguments:
            * `self`
            * `volume_folder`: str | Path -> the path of the volume to iterate over (i.e.: `data/train/kidney_1_voi`)
            * `mode`: str -> specifies the way in which to iterate over the folder 
        
        Returns: 
            * `None`

        With `mode == "train"`, the `Tif3DVolumeIterableFolder` will expect its `self.volume_folder` folder attribute
        to contain both the `images` and `labels` folders.

        With `mode == "test"`, it will only expect it to contain the `images` folder, as there is not available mask for
        the testing data.

        Finally, with `mode == "labels-only"`, it will only expect it to contain the `labels` folder, as it is necessary
        for loading the `kidney_3_dense` dataset (cfr `/README.md`)

        """
        assert mode in [
            "train",
            "test",
            "labels-only"
        ]
        self.mode = mode
        self.volume_folder = volume_folder
        if self.mode in ["train", "test"]:
            self.volume_images_folder = os.path.join(
                self.volume_folder,
                "images"
            )
            self.slice_images_paths = [
                os.path.join(
                    self.volume_images_folder,
                    slice_image_file
                )
                for slice_image_file in sorted(os.listdir(
                    self.volume_images_folder
                ))
            ]
            self.slice_images_ids = [
                int(re.findall(
                    "[0-9]{4}", 
                    os.path.basename(slice_file_name)
                )[0])
                for slice_file_name in sorted(os.listdir(
                    self.volume_images_folder
                ))
            ]
        if self.mode in ["train", "labels-only"]:
            self.volume_masks_folder = os.path.join(
                self.volume_folder,
                "labels"
            )
            self.slice_masks_paths = [
                os.path.join(
                    self.volume_masks_folder,
                    slice_mask_file
                )
                for slice_mask_file in sorted(os.listdir(
                    self.volume_masks_folder
                ))
            ]
            self.slice_masks_ids = [
                int(re.findall(
                    "[0-9]{4}", 
                    os.path.basename(slice_file_name)
                )[0])
                for slice_file_name in sorted(os.listdir(
                    self.volume_masks_folder
                ))
            ]
    
    @property
    def name(
            self
        ) -> str:
        """
        Arguments:
            * `self`

        Returns:
            * `str` -> the name of the dataset
        """
        path = os.path.normpath(self.volume_folder)
        return path.split(os.sep)[-1]

    def __len__(
            self
        ) -> int:
        """
        Arguments:
            * `self`

        Returns:
            * `int` -> the length of the dataset

        """
        if self.mode == "train":
            images_dir_len = len(self.slice_images_paths)
            masks_dir_len = len(self.slice_masks_paths)
            assert(images_dir_len == masks_dir_len), f"{images_dir_len=} and {masks_dir_len=} differ"
            return images_dir_len
        if self.mode == "test":
            return len(self.slice_images_paths)
        if self.mode == "labels-only":
            masks_dir_len = len(self.slice_masks_paths)
            return masks_dir_len

    def __getitem__(
            self,
            idx: int
        ) -> tuple[int, str | None, str | None]:
        """
        Arguments:
            * `self`
            * `idx`: int -> The index of the slice to be taken
        
        Returns:
            * `tuple[int, str | None, str | None]` -> tuple of three elements containing the data for the slice

                - `tuple[0]`: `int` -> The id of the slice
                - `tuple[1]`: `str | None` -> The path to the `.tif` image file, `None` if `self.mode == "labels-only"` 
                - `tuple[2]`: `str | None` -> The path to the `.tif` mask file, `None` if `self.mode == "test"` 
                
        """
        if self.mode == "train":
            slice_image_id = self.slice_images_ids[idx]
            slice_mask_id = self.slice_masks_ids[idx]
            assert(slice_image_id == slice_mask_id), f"{slice_image_id=} and {slice_mask_id=} differ"
            slice_image_path = self.slice_images_paths[idx]
            slice_mask_path = self.slice_masks_paths[idx]
            return slice_image_id, slice_image_path, slice_mask_path
        elif self.mode == "test":
            slice_image_id = self.slice_images_ids[idx]
            slice_image_path = self.slice_images_paths[idx]
            return slice_image_id, slice_image_path, None 
        elif self.mode == "labels_only":
            slice_mask_id = self.slice_masks_ids[idx]
            slice_mask_path = self.slice_masks_paths[idx]
            return slice_mask_id, None, slice_mask_path
