import os
import gc

import torch
from torch import cuda
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_warmup import LinearWarmup
from pytorch_memlab import profile

from monai.losses import DiceCELoss
from monai.networks.nets import SwinUNETR

from ..src.training.training_swin_unetr import BVSegSwinUnetRTraining
from ..src.data_utils.monai_data_loaders_sequential import create_data_loaders_from_splits_metadata
from ..src.feature_engineering.monai_transformations import get_monai_transformations

@profile
def debug(
        args,
        device,
    ) -> None:
    train_batch_size = args.train_batch_size
    validation_batch_size = args.validation_batch_size
    epochs = args.epochs
    patience = args.patience
    initial_learning_rate = args.initial_learning_rate
    model_name = args.model_name
    dump_path = args.dump_path
    log_path = args.log_path
    relative_improvement = args.relative_improvement
    weight_decay = args.weight_decay
    load_pre_trained = args.load_pre_trained
    splits_metadata_path = os.path.join(
        args.metadata_base_path,
        "individual_datasets"
    )
    patch_size = args.patch_size
    K = args.K
    overlap = args.overlap
    warmup_period = args.warmup_period
    data_parallel = args.data_parallel
    skip = args.skip
    feature_size = args.feature_size
    pretrained_path = args.pretrained_path
    # creating the data loader
    torch.backends.cudnn.benchmark = True
    train_transforms, val_transforms, test_transforms = get_monai_transformations(
        patch_size,
        device
    )
    # train a model for each split
    for split_to_train in range(K):
        print(f"Training on split {split_to_train}/{K}")
        print(f"Initializing model")
        model = SwinUNETR(
            img_size=(
                patch_size, 
                patch_size, 
                patch_size
            ),
            in_channels=1,
            out_channels=1,
            feature_size=feature_size,
            use_checkpoint=True,
        ).to(device)
        print("Model initialized")
        if load_pre_trained:
            print("Loading pre-trained weights")
            weight = torch.load(pretrained_path)
            model.load_from(weights=weight)
            model.to(device)
            print("Pre trained weights loaded")
        if data_parallel:
            print("Parallelizing model over multiple GPUs")
            model = torch.nn.DataParallel(
                model
            )
            #model.to(device)
            print("Model parallelized")
        splits_data_loaders = create_data_loaders_from_splits_metadata(
            split_to_train,
            splits_metadata_path,
            train_transforms,
            val_transforms,
            train_batch_size = train_batch_size,
            validation_batch_size = validation_batch_size,
            skip = skip
        )
        for (dataset_name, dataset_id, split_id, train_data_loader, validation_data_loader) in splits_data_loaders:
            print(f"Currently retrieving {dataset_name} ({dataset_id=}), {split_id=}")
            for idx, batch in enumerate(train_data_loader):
                for k, v in batch.items():
                    print(k, v.size()) 