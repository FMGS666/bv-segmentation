f"""

"""

import wandb

from pathlib import Path
from typing import Iterable, Callable, Any
from torch import nn, Tensor
from torch.optim.lr_scheduler import LRScheduler 
from torch.optim import Optimizer

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
            verbose: bool = True  
        ) -> None:
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
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        self.gradient_clipping = gradient_clipping
        self.history = {
            "train_loss": [9e20],
            "validation_loss": [9e20],
        }
        self.n_epochs_with_no_progress = 0
        self.n_saved_models = 0
        self.log_dir = log_dir
        self.dump_dir = os.path.join(self.dump_dir, self.session_id())
        self.log_dir = os.path.join(self.log_dir, self.session_id())
        try:
            os.mkdir(self.dump_dir)
            os.mkdir(self.log_dir)
        except FileExistsError:
            pass
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.prepare()
        self.save_settings()

    @property
    def session_id(
            self
        ) -> str:
        string =  f"model-{self.model_name}_optimizer-{self.optimizer.__class__}_scheduler-{self.scheduler.__class__.__name__ if self.scheduler else 'NA'}_{self.dict_as_string(self.optimizer_kwargs)}_{self.dict_as_string(self.scheduler_kwargs) if self.scheduler else 'NA'}"
        string = re.sub('\.', '_',string)
        string = re.sub("[><)(',\s]", '',string)
        return string

    @staticmethod
    def dict_as_string(
            dictionary: dict
        ) -> str:
        kv = []
        for k, v in dictionary.items():
            kv.append(f"{k}-{v}")
        return "_".join(kv)

    def prepare(
            self
        ) -> None: 
        self.model.to(self.device)
        self.optimizer = self.optimizer(
            self.model.parameters(),
            **self.optimizer_kwargs
        )
        if self.scheduler:
            self.scheduler = self.scheduler(
                self.optimizer,
                **self.scheduler_kwargs
            )

    def save_settings(
            self
        ) -> None:
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
        dump_dir = os.path.join(self.log_dir, f"epoch-{epoch}")
        os.mkdir(dump_dir)
        dump_file = os.path.join(dump_dir, "losses.json")
        with open(dump_file, "w") as log_file:
            json.dump(self.history, log_file)

    def plot_history(
            self,
            epoch
        ) -> None:
        raise NotImplementedError

    def dump_path(
            self
        ) -> Path:
        target_dir = os.path.join(self.dump_dir, f"#{self.n_saved_models}")
        return target_dir

    def track_validation_progress(
            self,
        ) -> bool:
        current_best_loss = min(self.history["validation_loss"][:-1])
        current_loss = self.history["validation_loss"][-1]
        if self.relative_improvement:
            return (current_best_loss - current_loss) / current_best_loss > self.tollerance
        return (current_best_loss - current_loss) > self.tollerance
    
    def dump_model(
            self
        ) -> None:
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
    
    def compute_metrics(
            self,
            predicted_mask: Tensor, 
            batch_masks: Tensor, 
            train: bool
        ) -> dict[str, float]:
        metrics_values = dict()
        for metric_name, (metric, minimize) in self.metrics.items():
            if train:
                metric_name = "train_" + metric_name
            else:
                metric_name = "validation_" + metric_name
            metric_value = metric(
                predicted_mask, 
                batch_masks
            )
            if isinstance(metric_value, Tensor):
                metric_value = metric_value.item()
            metrics_values[metric_name] = metric_value
        return metrics_values

    def training_pass(
            self,
            batch: dict
        ) -> Any:
        raise NotImplementedError
    
    def validation_pass(
            self,
            batch: dict
        ) -> Any:
        raise NotImplementedError
    
    def epoch(
            self,
            epoch: int
        ) -> Any:
        raise NotImplementedError
    
                
    def fit(
            self
        ) -> float:
        for epoch in range(self.epochs):
            print(f"epoch: {epoch + 1}/{self.epochs}")
            current_metrics = self.epoch(epoch)
            self.display_metrics(current_metrics)
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
        return min(self.history[self.metric_to_monitor])