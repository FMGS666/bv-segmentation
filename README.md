# Blood Vessel Segmentation

This repository contains the code that we will use for the Blood Vessel Segmentation competition hosted on `kaggle.com`


<video src="animations/kidney_1_dense.mp4" controls="controls" style="max-width: 730px;">
</video>

---

The data of this competition is under the form of three dimensional tomographies of the human kidneys. 

The goal of the competition is to develop a model which could be used to automatically annonate the pixels containing a blood-vessel, thus freeing the radiologist from the cumbersome task of doing this manually.

A more detailed description of the datasets and the additional data can be found on the <a href = "https://www.kaggle.com/competitions/blood-vessel-segmentation">competition's website </a>

## The Images

We have a total of 5 training datasets:

* 1) `kidney_3_sparse`, (1706 $\times$ 1510 pixels)
* 2) `kidney_3_dense`, (1706 $\times$ 1510 pixels)
* 3) `kidney_1_dense`, (1303 $\times$ 912 pixels)
* 4) `kidney_1_voi`, (1928 $\times$ 1928 pixels)
* 5) `kidney_2`, (1041 $\times$ 1511 pixels)

Each of them contains the 3D tomography of a different kidney, represented as vertical slices of the kidney.

Each of this contain inside them two folders, named `images` and `labels`, 
that contain, respectively the slice tomography and the ground truth mask.
The only exception is for the dataset 2, that only contains the `labels` folder, 
as the respective images are a subset of the images in `kidney_3_sparse`.
The difference between the two datasets is due to a different sparsity of the 
ground truth segmentation.

For what concerns the testing dataset, they are inside the `data/test` folder.
We have the following two different datasets:

* 1) `kidney_5`
* 2) `kidney_6`

## 0. Cloning the repository

To clone the repository, run the following command:

```
git clone git@github.com:FMGS666/bv-segmentation.git;
```
Then `cd` into the cloned repository:
```
cd bv-segmentation;
```

## 1. Installing the required libraries

Using anaconda or miniconda, the working environment can be created by running:
 
```
conda env create -f env.yml;
```

## 2. Downloading the data

The data can be downloaded folder by using the following command from the `kaggle` api:

```
kaggle competitions download -c blood-vessel-segmentation;
```

Then, you can unzip the `blood-vessel-segmentation.zip` into the `bv-segmentation/data` folder by running:

```
unzip blood-vessel-segmentation.zip -d data;
rm blood-vessel-segmentation.zip;
```

Note: This commands were not tested, please report if they are not working.

## 3. Structure of the project

All the code that in this repository can be used via the `bv-seg` module.
The source code is placed into the `bv-seg/src` folder, and organized into different subfolders.

For a better description of the structure of the `bv-seg/src` folder, confront the `bv-seg/README.md` file.

The `models` directory is the one where all the trained models' state dictionaries will be dumped. In this way, we will be able to load them later to make inference on the test images.

The `data` directory, with its self-explainatory name, should contain the data for the project. The images are too big to be store in github (check point 2 of this readme).

The `animations` directory contains the `.mp4` files of the 3D animations of the kidneys.

The `.tmp` contains temporary files used for logging the `stdout` of the training runs.


## 4. Running the module

For now, we do not have a `__main__.py` file yet. Anyway, it will (should) be possible to run the `sv-seg` module from the root of the project's repository by running:

```
python -m bv-seg [args] [--flags]
```

## TODO

This will be our TODO list, please add your TODOs at the bottom of this page and mark them as solved once they are solved:

- [] 1) Define Models
- [] 2) Write `__main__.py` file for running the module
- [] 3)  Check better the `bv-seg/training/training.py` file for bugs and write its documentation 
- [] 4) add wandb logging

- [] 5) update `README.md` for the dependency on `monai`, and show the command to download the pretrained weights (`wget -O models/pretrained/model_swinvit.pt https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/model_swinvit.pt
`)

- [] 6) Add command to update `env.yml` in the `README.md` (`conda env export --no-builds | grep -v "prefix" > env.yml`)
