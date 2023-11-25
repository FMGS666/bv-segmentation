# The `bv-seg` folder.

The `bv-seg` folder is structured in the following way.

* `src`: The folder containing all the source code for the project.

* `__main__.py`: The file containing the main script for running the module.

Instructions about how to run the module are provided in the `/README.md` file.

## The `bv-seg/src` folder

This folder contains all the source code used for the project.

It is structured in the following subfolders:

* `file_loaders`: containing the files with the definition of the object used to interact with the file and the folder structure in which the data is presented.
* `visualization`: containing the file with the definition of functions for visualizing the images in different ways.
* `feature_engineering`: containing the file with the definition of the object used for performing feature engineering (i.e.: image preprocessing) on the provided images. The package used for performing such transformations is `albumentations`.
* `model_selection`: containing the filed with the definition of the functions for performing validation split on the data. This module heavily relies on the `sklearn.model_selection` from `scikit-learn`. The main idea for performing validation split on the data is to do `KFold` cross validation.
* `data`: containing the definition of the file for creating the `torch.Dataset` for the data.

---

### `file_loaders`

The `file_loaders` folder contains two files:

* 1) `tif_iterable_folder.py`
* 2) `tif_file_loader.py`

(1) contains the definition of the `Tif3DVolumeIterableFolder` class, an interface for iterating over a folder of images and/or labels, as in the format of the competition, while (2) provides the `TifFileLoader` class, which gives a clear way to load the `.tif` files into memory. For a better description of their methods and usage, check their documentation.

---

### `visualization`

The `visualization` folder contains the following three files:

* 1) `image_mask_pair_plotter.py`
* 2) `volume_animations.py`
* 3) `pixel_distribution_plotter.py`

(1) contains the definition of the `tif_image_mask_pair_plotter` function, that takes a pair of image-mask `TifFileLoader` and plots a view of both images side by side. (2) contains the definition of the `create_volume_animations` function, that, given a `Tif3DVolumeIterableFolder` creates an animation of the 3D slice over time. The animation have already been generated and are now stored in the `animations` folder, in the root of the repository. (3) contains the definition of the `tif_image_mask_pair_distribution_plotter`. This function, given a pair of image-mask `TifFileLoader` plots the value distribution for their pixels. For a better description of their methods and usage, check their documentation (**I still need to write it tho**).

---

### `feature_engineering`


The `feature_engineering` folder contains the following file:

* 1) `feature_engineering.py`

This file contains the definition of the `BVSegFeatureEngineering` object, which can be used to perform custom transformations on the images by using the `albumentations` package. For a better description of its functionalities and how to use it, check the module documentation.

---

### `data`

The `data` folder contains the following file:

* 1) `dataset.py`

This file contains the definition of the `BVSegDataset` class, that inherits the `torch.utils.data.Dataset` class and can be used to create the dataset for training the models. For a better description of its functionalities, the format of the data and how to use it, check the module documentation.

---

### `model_selection`

The `model_selection` folder contains the following two files:

* 1) `k_fold_split.py`
* 2) `utils.py`*

(1) contains the definition of the `k_fold_split_iterable_folder` function, that, given  a `Tif3DVolumeIterableFolder` object, returns a generator with the training and validation indexes for each split. Note that this function will only return a *generator  of the indexes of the splits* and not the splits themselves (i.e.: the **actual** file names). This is why in (2), the `retrieve_filenames_from_split_indexes` is defined. This function takes the `Tif3DVolumeIterableFolder` object and the generator, output of the `k_fold_split_iterable_folder` functions, retrievs the file names from the indexes, and returns them organized inside a dictionary. For a better description of their usage, check the module's documentation.
