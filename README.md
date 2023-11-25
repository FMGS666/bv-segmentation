# Blood Vessel Segmentation

This repository contains the code that we will use for the Blood Vessel Segmentation competition hosted on `kaggle.com`


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

---

Here's the example animation for the `kidney_1_dense` dataset

<video src="animations/kidney_1_dense.mp4" controls="controls" style="max-width: 730px;">
</video>


---

## Installing the required libraries

Using anaconda or miniconda, the working environment can be created by `cd`ing into the repository root folder and run:
 
```
conda env create -f env.yml
```
