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
>>> transformations = Compose(
>>>     [
>>>         # define your transformations here
>>>     ]
>>> )
>>> feature_engineering = BloodVesselSegFeatureEngineering(
>>>     transformations = transformations, 
>>>     slice_images = True, 
>>>     slice_size = 256
>>> )
>>> image_paths = kidney_1_dense_iterable_folder.slice_images_paths
>>> masks_paths = kidney_1_dense_iterable_folder.slice_masks_paths
>>> dataset = BloodVesselSegDataset(
>>>      image_paths, 
>>>      masks_paths, 
>>>      feature_engineering
>>> )
```


"""

from torch.utils.data import Dataset

from ..feature_engineering.feature_engineering import BloodVesselSegFeatureEngineering

class BloodVesselSegDataset(Dataset):
    def __init__(
            self,
            images_files_list: list[str],
            masks_files_list: list[str],
            image_preprocesser: BloodVesselSegFeatureEngineering
        ) -> None:
        """
        Arguments:
            * `self`
            * `images_files_list: list[str]` -> list containing the paths to the images in the dataset
            * `masks_files_list: list[str]` -> list containing the paths to the masks in the dataset
            * `mode`: str -> specifies the way in which to iterate over the folder 
        
        Returns: 
            * `None`
        
        """
        super(BloodVesselSegDataset, self).__init__()
        self.images_files_list = images_files_list
        self.masks_files_list = masks_files_list
        self.image_preprocesser = image_preprocesser

    def __len__(
            self
        ) -> int:
        """
        Arguments:
            * `self`

        Returns: 
            * `int` -> the length of the dataset
        """
        return len(self.images_files_list)

    def __getitem__(
            self,
            idx: int
        ) -> dict[str, torch.Tensor]:
        """
        Arguments:
            * `self`
            * `idx`: int -> The index of the slice to be taken
        
        Returns:
            * `dict[str, torch.Tensor]` -> tuple of three elements containing the data for the slice
                - `keys()`: `str` -> whether the tensors are images or labels (i.e.: "images" or "labels")
                - `values(): torch.Tensor` -> tensors image

        """
        image_path = self.images_files_list[idx]
        mask_path = self.masks_files_list[idx]
        image = TifFileLoader(
            image_path
        )
        mask = TifFileLoader(
            mask_path
        )
        transformations = self.image_preprocesser.transform(
            image, 
            mask
        )
        transformed_image = transformations["image"]
        transformed_mask = transformations["mask"]
        transformed_image = torch.from_numpy(
            transformed_image
        ).float()
        transformed_image = torch.unsqueeze(transformed_image, dim = -3)
        transformed_mask = torch.from_numpy(
            transformed_mask
        ).long()
        transformed_mask = torch.unsqueeze(transformed_mask, dim = -3)
        item_dictionary = {
            "image": transformed_image, 
            "mask": transformed_mask
        }
        if self.image_preprocesser.slice:
            padding_mask = transformations["padding"]
            padding_mask = torch.from_numpy(
                padding_mask
            ).long()
            item_dictionary["padding"] = padding_mask
        return item_dictionary