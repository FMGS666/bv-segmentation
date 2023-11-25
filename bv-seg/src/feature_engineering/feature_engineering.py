f"""

file: {__file__}

Contents:

* `BloodVesselSegFeatureEngineering`

This class provides an interface for performing feature engineering on the images
It is possible to pass an instantiated `albumentations.Compose` object 
as the `transformations` argument 

Example of use: 

```
>>> from albumentations import Compose
>>> from src.file_utils.tif_iterable_folder import Tif3DVolumeIterableFolder
>>> from src.file_utils.tif_file_loader import TifFileLoader
>>>
>>> kidney_1_dense_path = "./data/train/kidney_1_dense"
>>> kidney_1_dense_iterable_folder = Tif3DVolumeIterableFolder(kidney_1_dense_path, "train")
>>> kidney_1_dense_slice_id, kidney_1_dense_image_path, kidney_1_dense_mask_path = kidney_1_dense_iterable_folder[0]
>>> kidney_1_dense_image = TifFileLoader(kidney_1_dense_image_path)
>>> kidney_1_dense_mask = TifFileLoader(kidney_1_dense_mask_path)
>>> transformations = Compose(
>>>     [
>>>         # define your transformations here
>>>     ]
>>> )
>>> feature_engineering = BVSegFeatureEngineering(
>>>     transformations = transformations, 
>>>     patch = True, 
>>>     patch_size = 256
>>> )
>>> transformation_results_dictionary = feature_engineering.transform(
>>>     kidney_1_dense_image,
>>>     kidney_1_dense_mask
>>> )
>>> transformed_image = transformation_results_dictionary["image"]
>>> transformed_mask = transformation_results_dictionary["mask"]
>>> is_padding = transformation_results_dictionary["padding"]
```

"""


import numpy as np

from albumentations import Compose
from patchify import patchify, unpatchify

from ..file_utils.tif_file_loader import TifFileLoader


class BVSegFeatureEngineering(object):
    def __init__(
            self,
            transformations: Compose | None = None,
            patch: bool = True,
            patch_size: int = 128,
            image_normalization_factor: np.float32 = np.float32(2**16), # this is the maximum pixel values of the images in our dataset
            mask_normalization_factor: np.float32 = np.float32(255), # only one, binary mask
        ) -> None:
        """
        Arguments:
            * `self`
            * `transformations: Compose | None` -> the `albumentation.Compose` transformations to be applied to the images
            * `patch: bool` -> whether to slice images
            * `patch_size: int` -> the size of the slices (used if `patch == True`)
            * `image_normalization_factor: np.float32` -> the normalization factor for the images' pixels
            * `mask_normalization_factor: np.float32` -> the normalization factor for the masks' pixels
        Returns: 
            * `None`

        """
        self.transformations = transformations
        self.patch_size = patch_size
        self.patch = patch
        self.image_normalization_factor = image_normalization_factor
        self.mask_normalization_factor = mask_normalization_factor

    def normalize_image(
            self,
            image_array: np.ndarray
        ) -> np.ndarray:
        """
        Arguments:
            * `self`
            * `image_array: np.ndarray` -> the array containing the image to be normalized

        Returns:
            * `np.ndarray` -> the nornalized image array
        
        """
        image_array = image_array / self.image_normalization_factor
        return image_array
    
    def normalize_mask(
            self,
            mask_array: np.ndarray
        ) -> np.ndarray:
        """
        Arguments:
            * `self`
            * `mask_array: np.ndarray` -> the array containing the mask to be normalized

        Returns:
            * `np.ndarray` -> the nornalized mask array
        
        """
        mask_array = mask_array // self.mask_normalization_factor
        return mask_array

    def get_target_paddings(
            self,
            image_array: np.ndarray
        ) -> tuple[tuple[int, int]]:
        """
        Arguments:
            * `self`
            * `image: np.ndarray` -> the array containing the image to pad

        Returns:
            * `tuple` -> the target paddings for making the shape divisible by `self.patch_size`
                - `tuple[0]`: `tuple` -> the target vertical paddings
                    - `tuple[0][0]`: `int` -> the target upper vertical padding
                    - `tuple[0][1]`: `int` -> the target lower vertical padding
                - `tuple[1]`: `tuple` -> the target horizontal paddings
                    - `tuple[1][0]`: `int` -> the target left horizontal padding
                    - `tuple[1][1]`: `int` -> the target right horizontal padding
        """
        image_height, image_width = image_array.shape
        num_pixels_to_pad_vertically = self.patch_size - image_height % self.patch_size
        num_pixels_to_pad_horizontally = self.patch_size - image_width % self.patch_size
        if num_pixels_to_pad_vertically % 2 == 0:
            v_pad = (num_pixels_to_pad_vertically // 2, num_pixels_to_pad_vertically // 2)
        else: 
            v_pad = (num_pixels_to_pad_vertically // 2, num_pixels_to_pad_vertically // 2 + 1)
        if num_pixels_to_pad_horizontally % 2 == 0:
            h_pad = (num_pixels_to_pad_horizontally // 2, num_pixels_to_pad_horizontally // 2)
        else: 
            h_pad = (num_pixels_to_pad_horizontally // 2, num_pixels_to_pad_horizontally // 2 + 1)
        assert np.sum(v_pad) == num_pixels_to_pad_vertically, f"{np.sum(v_pad)=} and {num_pixels_to_pad_vertically=} differ"
        assert np.sum(h_pad) == num_pixels_to_pad_horizontally, f"{np.sum(h_pad)=} and {num_pixels_to_pad_horizontally=} differ"
        return (v_pad, h_pad)

    def pad_image(
            self,
            image_array: np.ndarray
        ) -> np.ndarray:
        """
        Arguments:
            * `self`
            * `image: np.ndarray` -> the array containing the image to pad

        Returns:
            * `np.ndarray` -> the padded image array 
        """
        target_padding = self.get_target_paddings(
            image_array
        )
        padded_array =  np.pad(
            image_array,
            target_padding, 
            'constant',
            constant_values = -1.0
        )
        padding_mask = np.ones_like(
            padded_array
        )
        padding_indexes = np.where(
            padded_array == -1.0
        )
        padding_mask[padding_indexes] = 0
        padded_array[padding_indexes] = 0
        assert padded_array.shape[0] % self.patch_size == 0, print(f"{padded_array.shape[0] % self.patch_size=}")
        assert padded_array.shape[1] % self.patch_size == 0, print(f"{padded_array.shape[1] % self.patch_size=}")
        return padded_array, padding_mask

    def patch_image(
            self,
            padded_image_array: np.ndarray
        ) -> np.ndarray:
        """
        Arguments:
            * `self`
            * `padded_image_array: np.ndarray` -> the array containing the padded image to slice

        Returns:
            * `np.ndarray` -> the sliced image array 
        """
        patches = patchify(
            padded_image_array, 
            (
                self.patch_size,
                self.patch_size
            )
        )
        return slices
    
    def transform(
            self,
            slice_loaded_image: TifFileLoader, 
            slice_loaded_mask: TifFileLoader
        ) -> dict[str, np.ndarray]:
        """
        Arguments:
            * `self`
            * `slice_loaded_image: TifFileLoader` -> the `TifFileLoader` object containing the image to be transformed
            * `slice_loaded_mask: TifFileLoader` -> the `TifFileLoader` object containing the mask to be transformed

        Returns:
            * `dict` -> the dictionary containing the transformed data
                - `keys(): str` -> the identifier of the data (i.e.: "image", "mask" or "padding")
                - `values(): np.ndarray` -> the array containing the transformed data
        
        """
        image_array = slice_loaded_image.image_array
        mask_array = slice_loaded_mask.image_array
        image_array = self.normalize_image(
            image_array
        )
        mask_array = self.normalize_mask(
            mask_array
        )
        result_dictionary = {
            "image": image_array,
            "mask": mask_array
        }
        if self.transformations:
            image_array = self.transformations(
                image = image_array 
            )["image"]
            result_dictionary["image"] = image_array
        if self.patch:
            image_array, is_padding_mask = self.pad_image(
                image_array
            )
            mask_array, _ = self.pad_image(
                mask_array
            )
            image_array = self.patch_image(
                image_array
            )
            mask_array = self.patch_image(
                mask_array
            )
            is_padding_mask = self.patch_image(
                is_padding_mask
            )
            result_dictionary["image"] = image_array
            result_dictionary["mask"] = mask_array
            result_dictionary["padding"] = is_padding_mask
        return result_dictionary
        
