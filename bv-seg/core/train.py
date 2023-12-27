import torch
from torch import cuda
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from monai.losses import DiceCELoss
from monai.networks.nets import SwinUNETR

from ..src.training.training_swin_unetr import BVSegSwinUnetRTraining
from ..src.data_utils.monai_data_loaders import create_data_loaders_from_splits_metadata
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
    sched_step_after_train = args.sched_step_after_train
    model_name = args.model_name
    dump_path = args.dump_path
    log_path = args.log_path
    relative_improvement = args.relative_improvement
    weight_decay = args.weight_decay
    splits_metadata_path = args.splits_metadata_path
    patch_size = args.patch_size
    K = args.K
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
            lr = initial_learning_rate, 
            weight_decay = weight_decay
        )
        scheduler = CosineAnnealingLR(
            optimizer, 
            initial_learning_rate
        )
        splits_data_loaders = create_data_loaders_from_splits_metadata(
            splits_metadata_path,
            train_transforms,
            val_transforms,
            train_batch_size = train_batch_size,
            validation_batch_size = validation_batch_size
        )
        loss_function = DiceCELoss(sigmoid = True)
        def yield_data_loaders():
            for (dataset_id, split_id, train_data_loader, validation_data_loader) in splits_data_loaders:
                if split_id == split_to_train:
                    yield train_data_loader, val_data_loaders
        data_loaders_yielder = yield_data_loaders()
        trainer = BVSegSwinUnetRTraining(
            model,
            data_loaders_yielder,
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
        trainer.fit()
