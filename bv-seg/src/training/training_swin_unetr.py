f"""

file: {__file__}

Contents:
    * `BVSegSwinUnetRTraining`

This is the base class for training the Swin UnetR model. 
It implements the model-specific methods (namely: `training_pass`, 
`validation_pass` and `epoch`).

"""

import torch
import wandb
import os
import re
import json
import gc

from time import time
from pathlib import Path
from typing import Iterable, Callable, Any
from torch import nn, Tensor, cuda
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
        """
        Arguments: 
            * `self`
            * `model: nn.Module` -> the module to be trained
            * `train_data_loader: Iterable` -> the training data loader
            * `val_data_loader: Iterable` -> the validation data loader
            * `optimizer: Optimizer` -> the optimizer used for training
            * `loss: nn.Module` -> the loss used for back-propagating the gradients
            * `initial_learning_rate: float | None` -> the initial learning rate
            * `scheduler: LRScheduler | None` -> the scheduler for adapting the step size
            * `epochs: int` -> the maximum number of epochs which to train the model over
            * `patience: int` -> the maximum number of epochs without improvement 
                in the validation loss to wait before doing early stopping
            * `sched_step_after_train: bool` -> whether to perform the scheduler step after 
                training
            * `model_name: str` -> the name of the model
            * `dump_dir: str | Path` -> the directory where to save the 
                trained models' state dictionaries
            * `log_dir: str | Path` -> the directory where to save 
                the logs of the training session
            * `optimizer_kwargs: dict | None` -> additional key-word arguments 
                to be passed to the optimizer
            * `scheduler_kwargs: dict | None` -> additional key-word arguments 
                to be passed to the scheduler
            * `plot_style: str` -> the style for the plots
            * `split_size: int` -> the size of the patches in which the images are split
                (i.e.: the input size for the model)
            * `tollerance: float` -> the tollerance used when considering 
                validation loss improvements (i.e.: the smaller, the stricter the early
                stopping criterion)
            * `amp: bool` -> whether to use automatic mixed precision when scaling the gradient or not
            * `gradient_clipping: float | None` -> the value at which to clip gradient norms
            * `relative_improvement: bool` -> whether to consider relative improvement as stopping criterion
            * `scale_grad: bool` -> whether to perform gradient scaling after the 
                training pass and before backpropagation
            * `verbose: bool` -> the verbosity of the training 
        
        Returns: 
            * `None`
            
        """
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
        """
        Arguments: 
            * `self`
            * `batch: dict[str, Tensor]` -> batch which to perform the training pass over
        
        Returns: 
            * `float` -> the training loss
        
        Performs a forward and a backward pass on a training batch and returns the obtained training loss.
        """
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
        del logit_map, loss, x, y
        gc.collect()
        cuda.empty_cache()
        return loss_value
    
    @profile
    def validation_pass(
            self,
            batch: dict
        ) -> None:
        """
        Arguments: 
            * `self`
            * `batch: dict[str, Tensor]` -> batch which to perform the validation pass over
        
        Returns: 
            * `None` -> this function only updates the `self.dice_metric` atribute
                and does not return any object
        
        Performs a forward on a validation batch and returns the obtained validation loss.
        """
        val_inputs, val_labels = (batch["image"], batch["label"])
        with torch.cuda.amp.autocast():
            val_outputs = sliding_window_inference(val_inputs, tuple([self.split_size]*3), 1, self.model)
        val_labels_list = decollate_batch(val_labels)
        val_labels_convert = [
            self.post_label(val_label_tensor) for val_label_tensor in val_labels_list
        ]
        val_outputs_list = decollate_batch(val_outputs)
        val_output_convert = [
            self.post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
        ]
        del val_labels_list, val_labels_convert,\
            val_outputs_list, val_output_convert, batch
        gc.collect()
        self.dice_metric(y_pred=val_output_convert, y=val_labels_convert)
    
    def epoch(
            self,
            epoch: int
        ) -> Any:
        """
        Arguments: 
            * `self`
            * `epoch: int` -> the id of the current epoch
        
        Returns: 
            * `Any` -> the results of the epoch (i.e.: training and validation losses)
        
        Iterates over the batches of `self.train_data_loader` and `self.val_data_loader`
        and performs a full epoch over them by calling `self.training_pass` and `self.validation_pass`
        """
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
                % (idx + 1, len(self.train_data_loader), train_loss)
            )
            del batch
            gc.collect()
            cuda.empty_cache()
            current_metrics["train_loss"] += train_loss
        self.model.eval()
        with torch.no_grad():
            for idx, batch in enumerate(epoch_iterator_val):
                self.validation_pass(batch)
                del batch
                gc.collect()
            mean_dice_val = self.dice_metric.aggregate().item()
            self.dice_metric.reset()
            cuda.empty_cache()
            current_metrics["val_loss"] += mean_dice_val
        for key, value in current_metrics.items():
            self.history[key].append(value)
    
