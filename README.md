# Blood Vessel Segmentation

This repository contains the code that we will use for the Blood Vessel Segmentation competition hosted on `kaggle.com`


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

## The Model

We will use the implementation of the "Swin UnetR" Transformers Convolutional architecture, provided by <a href = "https://monai.io/">MonAi</a>. This model achieved state of art performance on semantic segmentation tasks on MRI images. For a better recap about state of art model for semantic segmentation of medical images, check <a href="https://paperswithcode.com/task/medical-image-segmentation">this link</a>

The paper with which this architecture was firstly published can be found on <a href = "https://arxiv.org/pdf/2201.01266.pdf">ArXiv</a>.

An example notebook provided by MonAi itself can be found <a href="https://colab.research.google.com/drive/1IqdpUPM_CoKYj6EHNb-IYaCiHvEiM08D#scrollTo=RmiwUf43FiMa">here</a>.

This model takes as input 3D volumes, whose size in each spatial dimension must be divisible by 32. Our data, however, is under the for of `.tif` files. Each of them contains a vertical slice of the 3D volume. For a better description of the expected pipeline to follow, please go to part 6 of this document.

## 0. Cloning the repository and prepare the folder structure

To clone the repository, run the following command:

```
git clone git@github.com:FMGS666/bv-segmentation.git;
```
Then `cd` into the cloned repository:
```
cd bv-segmentation;
```
As the `./data` folder is in the `.gitignore` file, it only contains the `.gitkeep` file, once the repository is cloned. We want also to create the `./models/pretrained` folder where to store third-party pre-trained weight. You can achieve all this by running:
```
rm ./data/.gikeep;
mkdir ./data/splits_sampled_volumes;
mkdir ./data/splits_metadata;
mkdir ./models/pretrained;
```

To download the pre-trained weights, please run:

```
wget -O models/pretrained/model_swinvit.pt https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/model_swinvit.pt;
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

The `scripts` directory (will) contain utility scripts for running the module in different 
environments.

The `.tmp` contains temporary files used for logging the `stdout` of the training runs.


## 4. Installing the required dependencies

Before running the module, you need to install the required dependencies. The `env.yml` file is there to help you. You can automatically install all the dependencies using `conda` by simply running:

```
conda env create --name blood-vessel-seg --file=env.yml;
```

For creating the environment, then run:

```
conda activate blood-vessel-seg;
```

If, as it could occur, you happen to have found an additional dependency for the project, please update the `env.yml` by running:

```
conda env export --no-builds | grep -v "prefix" > env.yml;
```


## 5. Running the module

It will (should) be possible to use the `sv-seg` module from the root of the project's repository by running:

```
python -m bv-seg [command] [--flags args];
```

A comprehensive list of all the possible arguments could be viewed by running:

```
python -m bv-seg -h
```

## 6. Using the code on other platforms

In order to use the `bv-seg` module on other platforms, you can download it as a `.zip` file, with the requirements needed for creating the anaconda environment, by running:

```
wget https://fmgs666.github.io/bv-seg.zip;
wget https://fmgs666.github.io/env.yml;
unzip bv-seg.zip;
conda env create --name envname --file=env.yml;
```

Whenever changes are made to the `bv-seg` module, you can upload them to the github pages by running the following commands:
```
cd ..; # cd to parent directory in order to be outside of this repository's folder
git clone git@github.com:FMGS666/FMGS666.github.io.git;
zip -r ./FMGS666.github.io/bv-seg.zip blood-vessel-seg/bv-seg;
cd FMGS666.github.io;
git add .;
git commit -m "Updating bv-seg.zip";
git push origin main;
cd ../;
rm -r -f ./FMGS666.github.io;
cd blood-vessel-seg;
```
On the other hand, commits to the `env.yml` can be pushed to the github page by running, from the repository's root directory:
```
conda env export --no-builds | grep -v "prefix" > env.yml;
cd ../;
git clone git@github.com:FMGS666/FMGS666.github.io.git;
cp ./blood-vessel-seg/env.yml ./FMGS666.github.io/env.yml
git add .;
git commit -m "Updating env.yml";
git push origin main;
cd ../;
rm -r -f ./FMGS666.github.io;
cd blood-vessel-seg;
```

## 6. The Pipeline

### Data Split

As required by a standard machine learning pipeline, we need to split our training data into training and validation set, in order to perform model validation and being able to estimate the generalization errors of our models. Since we have at our disposal multiple datasets we will perform a split on each of them. 
The train, validation split technique that we will use is the KFold Cross Validation, which consists in splitting the whole dataset in K Folds and use each one of them to validate the model at each validation iteration. We will set $K$ (i.e.: the number of folds which to split the dataset in), to 3. During the split, it is fundamental **not** to shuffle the data, otherwise the 3D volume reconstruction would be invalidated. In this way, each split will constitute one third of the whole kidney volume.

Since we have 4 datasets at our disposal, at first we will split each of them into $K=3$ groups and consider each split over the whole 4 dataset as a unique group (i.e.: Use split 0 of dataset 0, split 0 of dataset 1, etc. for the first group, then use split 1 of dataset 0, split 1 of dataset 1, etc. for the second and so on).

### Subsampling 3D Volumes

We will then randomly take vertical slices to subsample blocks of volumes from a whole group. The slices will have the same depth and a randomly chosen axial center. Note that, in this way, volumes within the same split could overlap, while overlap between two splits will not occur, as the subsampling happens at a split level.

Since the data is not provided under this form, we need to process it accordingly. It is possible to create such volume samples by running:

```
python3 -m bv-seg sample --n-samples <n_samples> --context-length <contex_length>
```

This command will sample 5 volumes of depth `2*context_length + 1` and write them as `nii.gz` files in the `data/splits_sampled_volumes/<dataset-name>/#<split-id>/<type>/` folder, where `<type>` could be either `images` or `masks` and write their metadata to the `data/splits_metadata/<dataset-name>/#<split-id>.json` file.
The metadata is required to use the `monai` data loaders.


### Training the model

Once that we have sample enough volumes and they are saved in the `./data/splits_sampled_volumes` folder, and that the metadata are correctly stored in the `./data/splits_metadata` folder, we can proceed by training the model with its default settings, by running the following command:

```
python3 -m bv-seg train
```

There is plenty of additional flags that could be added to the program, please use the module helper flag to get a brief description of each additional argument. 

---

<video src="animations/kidney_1_dense.mp4" controls="controls" style="max-width: 730px;">
</video>
