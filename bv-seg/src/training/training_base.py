f"""

file: {__file__}

Contents:
    * `BVSegTraining`

This is the base class for training objects. As the training and validation
pass could slightly differ across the models, while there are still other 
components and tasks during training that are model-agnostic, we provide here a 
base class for all this methods and functions that do not depend on the model.
For the model-specific methods (namely: `training_pass`, `validation_pass` and 
`epoch`), the other classes are defined in the `bv-seg/src/training` module.
The `fit` method, given a well specified interface for the `epoch` function, 
could actually be considered model-agnostic as well. 

"""

import wandb
import torch
import os
import re
import json

from pathlib import Path
from typing import Iterable, Callable, Any
from torch import nn, Tensor
from torch.optim.lr_scheduler import LRScheduler 
from torch.optim import Optimizer
#from torch.utils.tensorboard import SummaryWriter

class BVSegTraining(object):
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
            tollerance: float = 1e-5,
            amp: bool = False,
            gradient_clipping: float | None = 1.0,
            relative_improvement: bool = False,
            scale_grad: bool = True,
            verbose: bool = True,
            decrease: bool = False
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
        self.model = model
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.optimizer = optimizer
        self.initial_learning_rate = initial_learning_rate
        self.loss = loss
        self.scheduler = scheduler
        self.epochs = epochs
        self.patience = patience
        self.sched_step_after_train = sched_step_after_train
        self.model_name = model_name
        self.dump_dir = dump_dir
        self.relative_improvement = relative_improvement
        self.scale_grad = scale_grad
        self.verbose = verbose
        if optimizer_kwargs is None :
            optimizer_kwargs = dict()
            optimizer_kwargs["lr"] = self.initial_learning_rate
        self.optimizer_kwargs = optimizer_kwargs
        self.scheduler_kwargs = scheduler_kwargs
        self.plot_style = plot_style
        self.patch = patch_image
        self.split_size = split_size
        self.tollerance = tollerance
        self.amp = amp
        self.decrease = decrease
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        self.gradient_clipping = gradient_clipping
        self.history = {
            "train_loss": [9e20],
            "validation_loss": [9e20],
        }
        self.n_epochs_with_no_progress = 0
        self.n_saved_models = 0
        self.log_dir = log_dir
        self.dump_dir = os.path.join(self.dump_dir, self.session_id)
        self.log_dir = os.path.join(self.log_dir, self.session_id)
        try:
            os.mkdir(self.dump_dir)
            os.mkdir(self.log_dir)
        except FileExistsError:
            pass
        #self.writer = SummaryWriter(self.log_dir)
        #self.writer.add_graph(self.model)
        #self.writer.close()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.save_settings()

    @property
    def session_id(
            self
        ) -> str:
        """
        Arguments: 
            * `self`
        
        Returns:
            * `str` -> the id of the training session encoded as a string
        """
        string =  f"model-{self.model_name}_optimizer-{self.optimizer.__class__}_scheduler-{self.scheduler.__class__.__name__ if self.scheduler else 'NA'}_{self.dict_as_string(self.optimizer_kwargs)}_{self.dict_as_string(self.scheduler_kwargs) if self.scheduler_kwargs else 'NA'}"
        string = re.sub('\.', '_', string)
        string = re.sub("[><)(',\s]", '', string)
        return string

    @staticmethod
    def dict_as_string(
            dictionary: dict
        ) -> str:
        """
        Arguments: 
            * `dictionary: dict` -> the dictionary to encode
        
        Returns:
            * `str` -> the dictionary encoded as a string, used for specifying the
            optimizer's and scheduler's setting in the `self.session_id` property
        """
        kv = []
        for k, v in dictionary.items():
            kv.append(f"{k}-{v}")
        return "_".join(kv)

    def save_settings(
            self
        ) -> None:
        """
        Arguments: 
            * `self` 
        
        Returns: 
            * `None`

        Saves the training session settings (i.e.: `self.optimizer_kwargs` and
        `self.scheduler_kwargs`) to a json file 
        """
        settings_file = os.path.join(
            self.dump_dir,
            "training_settings.json"
        )
        with open(settings_file, "w") as settings_file:
            settings = [
                self.optimizer_kwargs,
                self.scheduler_kwargs
            ]
            json.dump(settings, settings_file)

    def dump_logs(
            self,
            epoch: int
        ) -> None:
        """
        Arguments: 
            * `self`
            * `epoch: int` -> the current epoch which to dump the logs for
        
        Returns: 
            * `None`

        Saves the training session's history up until the current `epoch + 1`-th epoch
        """
        dump_dir = os.path.join(self.log_dir, f"epoch-{epoch}")
        os.mkdir(dump_dir)
        dump_file = os.path.join(dump_dir, "losses.json")
        with open(dump_file, "w") as log_file:
            json.dump(self.history, log_file)

    def plot_history(
            self,
            epoch
        ) -> None:
        """
        Arguments: 
            * `self`
            * `epoch: int` -> the current epoch which to dump the logs for
        
        Returns: 
            * `None`

        Saves the training session's history plots of the current `epoch + 1`-th epoch
        """
        raise NotImplementedError

    def dump_path(
            self
        ) -> str | Path:
        """
        Arguments: 
            * `self`
        
        Returns: 
            * `str | Path` -> the directory where to save the latest checkpoint
        """
        target_dir = os.path.join(self.dump_dir, f"#{self.n_saved_models}")
        return target_dir

    def track_validation_progress(
            self,
        ) -> bool:
        """
        Arguments: 
            * `self`
        
        Returns: 
            * `bool` -> whether the validation loss has improved in the last epoch
        """
        current_best_loss = min(self.history["validation_loss"][:-1])
        current_loss = self.history["validation_loss"][-1]
        if self.relative_improvement:
            return (current_best_loss - current_loss) / current_best_loss > self.tollerance if self.decrease \
                else (current_loss - current_best_loss) / current_best_loss > self.tollerance
        return (current_best_loss - current_loss) > self.tollerance if self.decrease \
            else (current_loss - current_best_loss) > self.tollerance
    
    def dump_model(
            self
        ) -> None:
        """
        Arguments: 
            * `self`
            * `epoch: int` -> the current epoch which to dump the logs for
        
        Returns: 
            * `None`

        Checkpoints the training session by saving to disk the latest 
        state dictionary for the model and the optimizer
        """
        dump_path = self.dump_path()
        os.mkdir(dump_path)
        torch.save(
            self.model.state_dict(), 
            os.path.join(dump_path, "model.pt")
        )
        torch.save(
            self.optimizer.state_dict(), 
            os.path.join(dump_path, "optimizer.pt")
        )


    def training_pass(
            self,
            batch: dict[str, Tensor]
        ) -> Any:
        """
        Arguments: 
            * `self`
            * `batch: dict[str, Tensor]` -> batch which to perform the training pass over
        
        Returns: 
            * `Any` -> the results of the training pass
        
        Performs a forward and a backward pass on a training batch and returns the obtained training loss.
        This is an abstract class and should be overridden by the model specific trainers.
        """
        raise NotImplementedError
    
    def validation_pass(
            self,
            batch: dict
        ) -> Any:
        """
        Arguments: 
            * `self`
            * `batch: dict[str, Tensor]` -> batch which to perform the validation pass over
        
        Returns: 
            * `Any` -> the results of the training pass
        
        Performs a forward on a validation batch and returns the obtained validation loss.
        This is an abstract class and should be overridden by the model specific trainers.
        """
        raise NotImplementedError
    
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
        This is an abstract class and should be overridden by the model specific trainers.
        """
        raise NotImplementedError
    
                
    def fit(
            self
        ) -> float:
        """
        Arguments: 
            * `self`
        
        Returns: 
            * `float` -> the minimum obtained validation loss
        
        Performs a full training session by iterating over the number of epochs and
        calling `self.epoch` each time. It also performs checkpointing and early stopping checks,
        as well as logging of the training session data.
        """
        for epoch in range(self.epochs):
            print(f"epoch: {epoch + 1}/{self.epochs}")
            current_metrics = self.epoch(epoch)
            self.dump_logs(epoch)
            if self.track_validation_progress():
                print("Validation loss decreasing, saving the model")
                self.n_saved_models += 1
                self.n_epochs_with_no_progress = 0
                self.dump_model()
            else:
                self.n_epochs_with_no_progress += 1
                print(f"Validation loss not improving over the last {self.n_epochs_with_no_progress}")
                if self.n_epochs_with_no_progress > self.patience:
                    print(f"patience reached, quitting the training")
                    break
        return min(self.history["validation_loss"])