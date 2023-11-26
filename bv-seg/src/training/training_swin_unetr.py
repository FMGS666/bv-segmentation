f"""

"""
import torch
import wandb

from time import time
from pathlib import Path
from typing import Iterable, Callable, Any
from torch import nn, Tensor
from torch.optim.lr_scheduler import LRScheduler 
from torch.optim import Optimizer
from pytorch_memlab import profile
from tqdm import tqdm


from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric

from .training_base import BVSegTraining

class BVSegSwinUnetRTraining(BVSegTraining):
    def __init__(
            self,
            model: nn.Module,
            train_data_loader: Iterable,
            val_data_loader: Iterable, 
            optimizer: Optimizer,
            loss: nn.Module,
            initial_learning_rate: float | None = None,
            scheduler: LRScheduler | None = None, 
            epochs: int = 100,
            patience: int = 3, 
            sched_step_after_train: bool = False,
            model_name: str = "UNet",
            dump_dir: str | Path = "./models",
            log_dir: str | Path = "./logs",
            optimizer_kwargs: dict | None = None,
            scheduler_kwargs: dict | None = None,
            plot_style: str = "dark_background",
            patch_image: bool = True,
            split_size: int = 64,
            metric_to_monitor: str = "validation_loss",
            tollerance: float = 1e-5,
            amp: bool = False,
            gradient_clipping: float = 1.0,
            relative_improvement: bool = False,
            scale_grad: bool = True,
            verbose: bool = True  
        ) -> None:
        super(BVSegSwinUnetRTraining, self).__init__(
            model,
            train_data_loader,
            val_data_loader, 
            optimizer,
            loss,
            initial_learning_rate= initial_learning_rate,
            scheduler = scheduler, 
            epochs = epochs,
            patience = patience, 
            sched_step_after_train = sched_step_after_train,
            model_name = model_name,
            dump_dir = dump_dir,
            log_dir = log_dir,
            optimizer_kwargs = optimizer_kwargs,
            scheduler_kwargs = scheduler_kwargs,
            plot_style = plot_style,
            patch_image = patch_image,
            split_size = split_size,
            tollerance = tollerance,
            amp = amp,
            gradient_clipping = gradient_clipping,
            relative_improvement = relative_improvement,
            scale_grad = scale_grad,
            verbose = verbose
        )
        self.epoch_iterator = tqdm(
            self.train_data_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
        )
        self.post_label = AsDiscrete(to_onehot=1)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=1)
        self.dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    
    @profile
    def training_pass(
            self,
            batch: dict
        ) -> float:
        x, y = (batch["image"].cuda(), batch["label"].cuda())
        with torch.cuda.amp.autocast():
            logit_map = self.model(x)
            loss = self.loss(logit_map, y)
        self.scaler.scale(loss).backward()
        loss_value = loss.item()
        self.scaler.unscale_(self.optimizer)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        return loss_value
    
    @profile
    def validation_pass(
            self,
            batch: dict
        ) -> None:
        val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
        with torch.cuda.amp.autocast():
            val_outputs = sliding_window_inference(val_inputs, self.model.img_size, 4, self.model)
        val_labels_list = decollate_batch(val_labels)
        val_labels_convert = [
            self.post_label(val_label_tensor) for val_label_tensor in val_labels_list
        ]
        val_outputs_list = decollate_batch(val_outputs)
        val_output_convert = [
            self.post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
        ]
        self.dice_metric(y_pred=val_output_convert, y=val_labels_convert)
    

    def epoch(
            self,
            epoch: int
        ) -> Any:
        current_metrics = {
            metric_name: 0.0 for metric_name in self.history.keys()
        }
        self.model.train(
            True
        )
        epoch_iterator = tqdm(
            self.train_data_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
        )
        epoch_iterator_val = tqdm(
            self.val_data_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True
        )
        print("Performing training pass")
        for idx, batch in enumerate(epoch_iterator):
            train_loss = self.training_pass(batch)
            epoch_iterator.set_description(
                "Training (%d / %d Steps) (loss=%2.5f)"
                % (idx + 1, len(self.train_loader), train_loss)
            )
            current_metrics["train_loss"] += train_loss
        self.model.eval()
        with torch.no_grad():
            for idx, batch in enumerate(epoch_iterator_val):
                self.validation_pass(batch)
            mean_dice_val = self.dice_metric.aggregate().item()
            self.dice_metric.reset()
            current_metrics["val_loss"] += mean_dice_val
        for key, value in current_metrics.items():
            self.history[key].append(value)
    
    def fit(
            self
        ) -> float:
        for epoch in range(self.epochs):
            print(f"epoch: {epoch + 1}/{self.epochs}")
            self.epoch(epoch)
            self.dump_logs(epoch)
            self.plot_history(epoch)
            if self.track_validation_progress():
                print("Validation loss decreasing, saving the model")
                self.n_saved_models += 1
                self.n_epochs_with_no_progress = 0
                self.dump_model()
                self.make_predictions()
            else:
                self.n_epochs_with_no_progress += 1
                print(f"Validation loss not improving over the last {self.n_epochs_with_no_progress}")
                if self.n_epochs_with_no_progress > self.patience:
                    print(f"patience reached, quitting the training")
                    break
        return min(self.history["validation_loss"])

