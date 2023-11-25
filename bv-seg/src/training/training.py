f"""

"""

import wandb

from pathlib import Path
from typing import Iterable, Callable
from torch import nn, Tensor
from torch.optim.lr_scheduler import LRScheduler 
from torch.optim import Optimizer
from pytorch_memlab import profile

class BloodVesselSegTraining(object):
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
            metrics: dict[str, tuple[Callable, bool]] | None = None,
            metric_to_monitor: str = "validation_loss",
            normalize_metrics: bool = False,
            tollerance: float = 1e-5,
            amp: bool = False,
            gradient_clipping: float = 1.0,
            relative_improvement: bool = False 
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
        self.metrics = metrics
        self.metric_to_monitor = metric_to_monitor
        self.normalize_metrics = normalize_metrics
        self.relative_improvement = relative_improvement
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
        if self.metrics is not None:
            for metric_name, (_, minimize) in self.metrics.items():
                metric_name_train = "train_" + metric_name
                metric_name_validation = "validation_" + metric_name
                if minimize:
                    self.history[metric_name_train] = [9e20]
                    self.history[metric_name_validation] = [9e20]
                else:
                    self.history[metric_name_train] = [0.0]
                    self.history[metric_name_validation] = [0.0]
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

    @staticmethod
    def display_metrics(
            metrics_dictionary: dict[str, float]
        ) -> None:
        for metric_name, metric_value in metrics_dictionary.items():
            print(f"{metric_name} = {metric_value}", end = "\t")
        print("\n")

    @profile
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
        current_best_loss = min(self.history[self.metric_to_monitor][:-1])
        current_loss = self.history[self.metric_to_monitor][-1]
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

    @profile
    def training_pass(
            self,
            batch: dict
        ) -> float:
        batch_images = batch["image"].to(self.device)
        batch_masks = batch["mask"].to(self.device)
        if self.patch:
            batch_is_padding = batch["padding"].to(self.device)
            batch_images = batch_images.view(
                -1,
                1,
                self.split_size,
                self.split_size
            )
            batch_masks = batch_masks.view(
                -1,
                1,
                self.split_size,
                self.split_size
            )
            batch_is_padding = batch_is_padding.view(
                -1,
                1,
                self.split_size,
                self.split_size
            )
        with torch.autocast(self.device, enabled=self.amp):
            predicted_mask = self.model.forward(
                batch_images
            ).float()
            if self.patch:
                loss = self.loss(
                    predicted_mask, 
                    batch_masks.float()
                )
                del batch_is_padding
                gc.collect()
            else: 
                loss = self.loss(
                    predicted_mask, 
                    batch_masks
                )
        self.optimizer.zero_grad(set_to_none=True)
        self.grad_scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
        if self.scheduler and self.sched_step_after_train:
            self.scheduler.step()
        loss_value = loss.item()
        metrics_values = dict()
        if self.metrics is not None:
            metrics_values = self.compute_metrics(predicted_mask, batch_masks, True)
        metrics_values["train_loss"] = loss_value
        del batch, batch_images, loss, predicted_mask, batch_masks
        gc.collect()
        torch.cuda.empty_cache()
        return metrics_values
    
    @profile
    def validation_pass(
            self,
            batch: dict
        ) -> float:
        batch_images = batch["image"].to(self.device)
        batch_masks = batch["mask"].to(self.device)
        if self.patch:
            batch_is_padding = batch["padding"].to(self.device)
            batch_images = batch_images.view(
                -1,
                1,
                self.split_size,
                self.split_size
            )
            batch_masks = batch_masks.view(
                -1,
                1,
                self.split_size,
                self.split_size
            )
            batch_is_padding = batch_is_padding.view(
                -1,
                1, 
                self.split_size,
                self.split_size
            )
            
        predicted_mask = self.model.forward(
            batch_images
        )
        if self.patch:
            loss = self.loss(
                predicted_mask, 
                batch_masks.float()
            )
            del batch_is_padding
            gc.collect()
        else: 
            loss = self.loss(
                predicted_mask, 
                batch_masks.float()
            )
        if self.scheduler and not self.sched_step_after_train:
            self.scheduler.step()
        loss_value = loss.item()
        metrics_values = dict()
        if self.metrics is not None:
            metrics_values = self.compute_metrics(predicted_mask, batch_masks, False)
        metrics_values["validation_loss"] = loss_value
        del batch, batch_images, loss, predicted_mask, batch_masks
        gc.collect()
        torch.cuda.empty_cache()
        return metrics_values
    
    @profile
    def epoch(
            self,
            epoch: int
        ) -> tuple[float]:
        current_metrics = {
            metric_name: 0.0 for metric_name in self.history.keys()
        }
        self.model.train(
            True
        )
        print("Performing training pass")
        with tqdm(total=len(train_data_loader)) as pbar:
            for idx, batch in enumerate(train_data_loader):
                training_metrics  = self.training_pass(
                    batch
                )
                for metric_name, metric_value in training_metrics.items():
                    current_metrics[metric_name] += metric_value
                del batch, training_metrics
                gc.collect()
                torch.cuda.empty_cache()
                pbar.update(1)
        print("Performing validation pass")
        self.model.eval()
        with tqdm(total=len(val_data_loader)) as pbar:
            with torch.no_grad():
                for batch in val_data_loader:
                    validation_metrics = self.validation_pass(
                        batch
                    )
                    for metric_name, metric_value in validation_metrics.items():
                        current_metrics[metric_name] += metric_value
                    pbar.update()
                    del batch, validation_metrics
                    gc.collect()
                    torch.cuda.empty_cache()
        for metric_name, metric_value in current_metrics.items():
            if "train" in metric_name and self.normalize_metrics:
                current_metrics[metric_name] /=  sum([len(train_data_loader) for train_data_loader in self.train_data_loader])
            elif "validation" in metric_name and self.normalize_metrics:
                current_metrics[metric_name] /=  sum([len(val_data_loader) for val_data_loader in self.val_data_loader])
            self.history[metric_name].append(
                current_metrics[metric_name]
            )
        return current_metrics
    
                
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