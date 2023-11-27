f"""

"""
import os
import gc
import torch

from monai.transforms import AsDiscrete
from monai.losses import DiceCELoss
from monai.networks.nets import SwinUNETR
from monai.data import WSIReader

from torch import cuda
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.optim import Adam, AdamW, SGD, RMSprop
from torch.optim.lr_scheduler import StepLR, ExponentialLR, PolynomialLR

from .src.file_loaders.tif_iterable_folder import Tif3DVolumeIterableFolder
from .src.model_selection.k_fold_split import k_fold_split_iterable_folder
from .src.model_selection.utils import retrieve_filenames_from_split_indexes, retrieve_k_fold_groups
from .src.utils.argument_parser import BVSegArgumentParser
from .src.data_utils.monai_data_loaders import create_data_loaders_from_splits_metadata
from .src.data_utils.utils import get_volumes_fold_splits, get_datasets_from_data_path, dump_dataset_metadata
from .src.training.training_swin_unetr import BVSegSwinUnetRTraining
from .src.volumes.write_volumes import write_volumes_to_tif
from .src.feature_engineering.monai_transformations import get_monai_transformations

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
    splits_metadata_path = args.splits_metadata_path
    context_length = args.context_length
    n_samples = args.n_samples
    write_volumes = args.write_volumes
    volumes_path = args.volumes_path
    dump_metadata = args.dump_metadata
    train = args.train
    patch_size = args.patch_size

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
    # retrieving the individual groups
    train_splits_groups = {
        dataset_name: retrieve_k_fold_groups(dataset_splits)
        for dataset_name, dataset_splits in train_datasets_splits_paths.items()
    }

    if write_volumes:
        write_volumes_to_tif(
            train_splits_groups,
            context_length,
            n_samples
        )
    
    # Now we should construct the dataloader from the sampled volumes
    train_volumes = {
        dataset_name: os.path.join(volumes_path, dataset_name)
        for dataset_name in train_splits_groups.keys()
    }
    train_splits_volumes = {
        dataset_name: get_volumes_fold_splits(train_volumes_directory)
        for dataset_name, train_volumes_directory in train_volumes.items()
    }

    # Now we need to dump the volumes metadata
    if dump_metadata:
        for dataset_name, splits in train_splits_volumes.items():
            for split_id, split_dictionary in splits.items():
                training_paths = split_dictionary["training"]
                validation_paths = split_dictionary["validation"]
                metadata_dir = os.path.join(
                    splits_metadata_path,
                    dataset_name
                )
                if not os.path.exists(metadata_dir):
                    os.mkdir(metadata_dir)
                dump_dataset_metadata(
                    metadata_dir,
                    split_id,
                    training_paths, 
                    validation_paths
                )
    if train:
        # creating the data loader
        torch.backends.cudnn.benchmark = True
        train_transforms, val_transforms, test_transforms = get_monai_transformations(
            patch_size
        )
        # train a model for each split
        for split_to_train in range(K):
            model = SwinUNETR(
                img_size=(
                    patch_size, 
                    patch_size, 
                    patch_size
                ),
                in_channels=1,
                out_channels=1,
                feature_size=48,
                use_checkpoint=True,
            ).to(device)
            weight = torch.load("models/pretrained/model_swinvit.pt")
            model.load_from(weights=weight)
            model.to(device)
            optimizer = AdamW(
                model.parameters(),
                lr = 1e-4, 
                weight_decay=1e-5
            )
            splits_data_loaders = create_data_loaders_from_splits_metadata(
                splits_metadata_path,
                train_transforms,
                val_transforms,
                train_batch_size = train_batch_size,
                validation_batch_size = validation_batch_size
            )
            loss_function = DiceCELoss()
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
                        scheduler_kwargs = None,
                        split_size = patch_size 
                    )
                    #trainer.prepare()
                    trainer.fit()
