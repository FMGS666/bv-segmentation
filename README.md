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


## 4. Running the module

For now, we do not have a `__main__.py` file yet. Anyway, it will (should) be possible to run the `sv-seg` module from the root of the project's repository by running:

```
python -m bv-seg [args] [--flags]
```