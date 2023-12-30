import torch
from torch import cuda
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pytorch_warmup import LinearWarmup

from monai.losses import DiceCELoss
from monai.networks.nets import SwinUNETR

from ..src.training.training_swin_unetr import BVSegSwinUnetRTraining
from ..src.data_utils.monai_data_loaders_merged import create_data_loaders_from_splits_metadata
from ..src.feature_engineering.monai_transformations import get_monai_transformations


def train(
        args,
        device
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
    metadata_base_path = args.metadata_base_path
    patch_size = args.patch_size
    K = args.K
    overlap = args.overlap
    warmup_period = args.warmup_period
    load_pre_trained = args.load_pre_trained
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
        if load_pre_trained:
            weight = torch.load("models/pretrained/model_swinvit.pt")
            model.load_from(weights=weight)
            model.to(device)
        if data_parallel:
            model = torch.nn.DataParallel(
                model
            )
            model.to(device)
        optimizer = AdamW(
            model.parameters(),
            lr = initial_learning_rate, 
            weight_decay = weight_decay
        )
        scheduler = CosineAnnealingLR(
            optimizer, 
            epochs
        )
        warmup = LinearWarmup(
            optimizer, 
            warmup_period
        )
        train_data_loader, validation_data_loader = create_data_loaders_from_splits_metadata(
            split_to_train,
            metadata_base_path,
            train_transforms,
            val_transforms,
            train_batch_size = train_batch_size,
            validation_batch_size = validation_batch_size
        )
        loss_function = DiceCELoss(sigmoid = True)
        trainer = BVSegSwinUnetRTraining(
            model,
            train_data_loader,
            validation_data_loader,
            optimizer,
            loss_function,
            device,
            initial_learning_rate = initial_learning_rate,
            scheduler = scheduler,
            warmup = warmup,
            epochs = epochs,
            patience = patience,
            model_name = model_name + f"-split#{split_to_train}",
            dump_dir = dump_path,
            log_dir = log_path,
            optimizer_kwargs = None,
            scheduler_kwargs = None,
            split_size = patch_size,
            overlap = overlap
        )
        trainer.fit()
