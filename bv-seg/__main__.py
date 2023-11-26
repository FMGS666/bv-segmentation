f"""

"""
import os
import torch

from monai.transforms import (
    AsDiscrete,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    EnsureTyped,
)

from monai.losses import DiceCELoss
from monai.networks.nets import SwinUNETR
from monai.data import WSIReader

from torch import cuda
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.optim import Adam, AdamW, SGD, RMSprop
from torch.optim.lr_scheduler import StepLR, ExponentialLR, PolynomialLR

from .src.file_loaders.tif_iterable_folder import Tif3DVolumeIterableFolder
from .src.file_loaders.tif_file_loader import TifFileLoader
from .src.file_loaders.monai_image_reader import MonAiImageReader
from .src.file_loaders.utils import construct_volume_tensor
from .src.utils.get_datasets_from_data_path import get_datasets_from_data_path
from .src.model_selection.k_fold_split import k_fold_split_iterable_folder
from .src.model_selection.utils import retrieve_filenames_from_split_indexes
from .src.utils.argument_parser import BVSegArgumentParser
from .src.utils.dump_dataset_metadata import dump_dataset_metadata
from .src.data_utils.monai_data_loaders import create_data_loaders_from_splits_metadata
from .src.training.training_swin_unetr import BVSegSwinUnetRTraining

device = "cuda" if cuda.is_available() else "cpu"

optimizers = {
    "AdamW": AdamW,
    "Adam": Adam, 
    "SGD": SGD,
    "RMSProp": RMSprop
}

schedulers = {
    "StepLR": StepLR,
    "ExponentialLR": ExponentialLR,
    "PolynomialLR": PolynomialLR,
    None: None
}

# define other transformations here, these were taken from
# the colab notebook by `monai` (link: https://colab.research.google.com/drive/1IqdpUPM_CoKYj6EHNb-IYaCiHvEiM08D#scrollTo=1zUgH23MFiMX) 
train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"], reader=MonAiImageReader, ensure_channel_first = True),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=0,
            a_max=2**16,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        ScaleIntensityRanged(
            keys=["label"],
            a_min=0,
            a_max=255,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Orientationd(keys=["image", "label"], axcodes="RA"),
        Spacingd(
            keys=["image", "label"],
            pixdim=5,
            mode=("bilinear", "nearest"),
        ),
        EnsureTyped(keys=["image", "label"], device=device, track_meta=False),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(96, 96),
            pos=1,
            neg=1,
            num_samples=1,
            image_key="image",
            image_threshold=0,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[0],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[1],
            prob=0.10,
        ),
        RandRotate90d(
            keys=["image", "label"],
            prob=0.10,
            max_k=3,
        ),
        RandShiftIntensityd(
            keys=["image"],
            offsets=0.10,
            prob=0.50,
        ),
    ]
)

val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"], reader=MonAiImageReader, ensure_channel_first = True),
        ScaleIntensityRanged(
            keys=["image"], a_min=0, a_max=2**16, b_min=0.0, b_max=1.0, clip=True
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Orientationd(keys=["image", "label"], axcodes="RA"),
        Spacingd(
            keys=["image", "label"],
            pixdim=5,
            mode=("bilinear", "nearest"),
        ),
        EnsureTyped(keys=["image", "label"], device=device, track_meta=True),
    ]
)

test_transforms = Compose(
    [
        LoadImaged(keys=["image"], reader=MonAiImageReader, ensure_channel_first = True), #channel_dim=None),
        ScaleIntensityRanged(
            keys=["image"], a_min=0, a_max=2**16, b_min=0.0, b_max=1.0, clip=True
        ),
        CropForegroundd(keys=["image"], source_key="image"),
        Orientationd(keys=["image"], axcodes="RA"),
        Spacingd(
            keys=["image"],
            pixdim=5,
            mode=("bilinear"),
        ),
        EnsureTyped(keys=["image"], device=device, track_meta=True),
    ]
)

if __name__ == "__main__":
    # defining arguments parser
    arg_parser = BVSegArgumentParser()
    args = arg_parser.parse_args()

    print("reading parsed arguments")
    train_data_path = args.train_data_path
    test_data_path = args.test_data_path
    K = args.K
    random_state = args.random_state
    shuffle = args.shuffle
    train_batch_size = args.train_batch_size
    validation_batch_size = args.validation_batch_size
    optimizer_id = args.optimizer_id
    scheduler_id = args.scheduler_id
    epochs = args.epochs
    patience = args.patience
    initial_learning_rate = args.initial_learning_rate
    gamma = args.gamma
    power = args.power
    sched_step_after_train = args.sched_step_after_train
    model_name = args.model_name
    dump_path = args.dump_path
    log_path = args.log_path
    relative_improvement = args.relative_improvement
    alpha = args.alpha
    stacked = args.stacked
    splits_metadata_path = args.splits_metadata_path

    # validating the arguments
    if optimizer_id not in optimizers.keys():
        raise ValueError(f"provided {optimizer_id=} is not currently supported")
    if scheduler_id not in schedulers.keys():
        raise ValueError(f"provided {scheduler_id=} is not currently supported")
    
    # retrieving arguments from id
    optimizer = optimizers[optimizer_id]
    scheduler = schedulers[scheduler_id]

    # loading the data
    print("retrieving the paths of each individual dataset")
    train_datasets_paths = get_datasets_from_data_path(
        train_data_path
    )
    test_datasets_paths = get_datasets_from_data_path(
        test_data_path
    )
    print(f"{train_data_path=}, {test_data_path=}")
    print("defining iterable folder for training and test images")
    train_iterable_folders = {
        dataset_name: Tif3DVolumeIterableFolder(dataset_path, "train") 
        for dataset_name, dataset_path in train_datasets_paths.items() if dataset_name != "kidney_3_dense"
    }
    test_iterable_folders = {
        dataset_name: Tif3DVolumeIterableFolder(dataset_path, "test")
        for dataset_name, dataset_path in test_datasets_paths.items()
    }
    # performing K-Fold split
    print("splitting the training images using K-Fold")
    train_splits_generators = {
        dataset_name: k_fold_split_iterable_folder(
            dataset_iterable_folder,
            K = K,
            random_state = random_state,
            shuffle = shuffle
        ) for dataset_name, dataset_iterable_folder in train_iterable_folders.items()
    }
    print("retrieving the file names")
    train_datasets_splits_paths = {
        dataset_name: retrieve_filenames_from_split_indexes(
            train_iterable_folders[dataset_name],
            dataset_splits
        ) for dataset_name, dataset_splits in train_splits_generators.items()
    }
    # dumping dataset metadata
    if len(os.listdir(splits_metadata_path)) == 0:
        for dataset_name, dataset_splits in train_datasets_splits_paths.items():
            dataset_metadata_path = os.path.join(
                splits_metadata_path, 
                dataset_name
            )
            os.mkdir(dataset_metadata_path)
            for split_id, splits in dataset_splits.items():
                training_paths = splits["training"]
                validation_paths = splits["validation"]
                dump_dataset_metadata(
                    dataset_metadata_path, 
                    split_id, 
                    training_paths, 
                    validation_paths
                )
    torch.backends.cudnn.benchmark = True
    # train a model for each split
    for split_to_train in range(K):
        model = SwinUNETR(
            img_size=(96, 96),
            in_channels=1,
            out_channels=1,
            feature_size=48,
            use_checkpoint=True,
        ).to(device)
        weight = torch.load("models/pretrained/model_swinvit.pt")
        model.load_from(weights=weight)
        splits_data_loaders = create_data_loaders_from_splits_metadata(
            splits_metadata_path,
            train_transforms,
            val_transforms,
            train_batch_size = train_batch_size,
            validation_batch_size = validation_batch_size
        )
        loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
        for (dataset_id, split_id, train_data_loader, validation_data_loader) in splits_data_loaders:
            if split_id == split_to_train:
                trainer = BVSegSwinUnetRTraining(
                    model,
                    train_data_loader,
                    validation_data_loader,
                    optimizer,
                    loss_function,
                    initial_learning_rate = initial_learning_rate,
                    scheduler = scheduler,
                    epochs = epochs,
                    patience = patience,
                    sched_step_after_train = sched_step_after_train,
                    model_name = model_name,
                    dump_dir = dump_path,
                    log_dir = log_path,
                    optimizer_kwargs = None,
                    scheduler_kwargs = None 
                )
                trainer.prepare()
                trainer.fit()

    


