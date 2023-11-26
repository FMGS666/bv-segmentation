f"""

file: {__file__}

Contents:

* `TifFileLoader`

This class provides an interface for loading `.tif` files from disk
into memory.
The file will be loaded inside a `PIL.Image`, a `np.array` and a `torch.Tensor`
objects.

Example of use: 

```
>>> from src.file_utils.tif_iterable_folder import Tif3DVolumeIterableFolder
>>> from src.file_utils.tif_file_loader import TifFileLoader
>>>
>>> kidney_1_dense_path = "./data/train/kidney_1_dense"
>>> kidney_1_dense_iterable_folder = Tif3DVolumeIterableFolder(kidney_1_dense_path, "train")
>>> kidney_1_dense_slice_id, kidney_1_dense_image_path, kidney_1_dense_mask_path = kidney_1_dense_iterable_folder[0]
>>> kidney_1_dense_image = TifFileLoader(kidney_1_dense_image_path)
>>> kidney_1_dense_mask = TifFileLoader(kidney_1_dense_mask_path)

```

"""

import os
import re
import torch

import numpy as np

from pathlib import Path
from PIL import Image



class TifFileLoader(object):
    def __init__(
            self,
            tif_file_path: str | Path,
        ) -> None:
        """
        
        Arguments:
            * `self`
            * `tif_file`: str | Path -> the path of the volume to iterate over (i.e.: `data/train/kidney_1_voi`)
        
        Returns: 
            * `None`

        """
        self.tif_file_path = tif_file_path
        self.image = Image.open(
            self.tif_file_path
        )    
        self.image_array = np.array(
            self.image,
            dtype = np.float32
        )
        self.image_tensor = torch.from_numpy(
            self.image_array
        )
    
    @property
    def slice_id(
            self
        ) -> int:
        """
        Arguments:
            * `self`

        Returns:
            * `str` -> the id of the slice represented in the image
        """
        slice_id_string = re.findall(
            "[0-9]{4}",
            os.path.basename(self.tif_file_path)
        )[0]
        return int(slice_id_string)

    @property
    def shape(
            self
        ) -> tuple:
        """
        Arguments:
            * `self`

        Returns:
            * `tuple` -> the shape of the underlying `self.image_array` `np.ndarray`
        """
        return self.image_array.shape
    
    def size(
            self
        ) -> torch.Size:
        """
        Arguments:
            * `self`

        Returns:
            * `torch.Size` -> the size of the underlying `self.image_tensor` `torch.Tensor`
        """
        return self.image_tensor.size()