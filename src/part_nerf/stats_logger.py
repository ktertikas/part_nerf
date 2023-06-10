"""Stats Logger, extended from https://github.com/nv-tlabs/ATISS"""

import sys
from typing import Dict

try:
    import wandb
except ImportError:
    print("Failed to import wandb package, resuming...")


class AverageAggregator:
    def __init__(self):
        self._value = 0
        self._count = 0

    @property
    def value(self):
        return self._value / self._count

    @value.setter
    def value(self, val):
        self._value += val
        self._count += 1


class StatsLogger:
    def __init__(self):
        self._values = {}
        self._loss = AverageAggregator()
        self._output_files = [sys.stdout]

    def add_output_file(self, f):
        self._output_files.append(f)

    def __getitem__(self, key):
        if key not in self._values:
            self._values[key] = AverageAggregator()
        return self._values[key]

    def clear(self):
        self._values.clear()
        self._loss = AverageAggregator()
        for f in self._output_files:
            if f.isatty():
                print(file=f, flush=True)

    def add_media(self, epoch, media, media_type: str):
        # Not supported for stats logger currently, just continue
        pass

    def print_progress(self, epoch, batch, loss, precision=".5f"):
        self._loss.value = loss
        msg = f"epoch: {epoch} - batch: {batch} - loss: {self._loss.value:{precision}}"
        for k, v in self._values.items():
            if isinstance(v, AverageAggregator):
                msg += f" - {k}: {v.value:{precision}}"
        for f in self._output_files:
            if f.isatty():
                print(msg + "\b" * len(msg), end="", flush=True, file=f)
            else:
                print(msg, flush=True, file=f)


class WandB(StatsLogger):
    """Log the metrics in weights and biases. Code adapted from
    https://github.com/angeloskath/pytorch-boilerplate/blob/main/pbp/callbacks/wandb.py

    Args:
        experiment_arguments (Dict): The dictionary containing all experiment arguments.
        model (nn.Module): The model to be tracked.
        project (str, optional): The W&B project name. Defaults to "experiment".
        name (str, optional): The W&B experiment name. Defaults to "experiment_name".
        experiment_dir (str, optional): The path to save the wandb metadata. Defaults to None.
        watch (bool, optional): Flag setting watcher on model topology for W&B. Defaults to False.
        log_frequency (int, optional): Log frequency for W&B watcher. Defaults to 10.
    """

    def __init__(
        self,
        experiment_arguments: Dict,
        model,
        project: str = "experiment",
        name: str = "experiment_name",
        experiment_dir: str = None,
        watch: bool = False,
        log_frequency: int = 10,
        start_epoch: int = 0,
    ):
        super().__init__()
        self.project = project
        self.experiment_name = name
        self.experiment_dir = experiment_dir
        self.watch = watch
        self.log_frequency = log_frequency
        self._epoch = start_epoch
        self._validation = False
        # Login to wandb
        wandb.login()

        # Init the run
        wandb.init(
            project=(self.project or None),
            name=(self.experiment_name or None),
            config=dict(experiment_arguments.items()),
            dir=experiment_dir,
        )

        if self.watch:
            wandb.watch(model, log_freq=self.log_frequency)

    def print_progress(self, epoch, batch, loss, precision=".5f"):
        super().print_progress(epoch, batch, loss, precision)

        self._validation = epoch < 0
        if not self._validation:
            self._epoch = epoch

    def add_media(self, epoch, media, media_type: str):
        self._values[media_type] = media
        self._validation = epoch < 0
        if not self._validation:
            self._epoch = epoch

    def clear(self):
        # Before clearing everything out send it to wandb
        prefix = "val_" if self._validation else ""
        values = {}
        for k, v in self._values.items():
            key = prefix + k
            if isinstance(v, AverageAggregator):
                value = v.value
            else:
                value = v
            values[key] = value
        values[prefix + "loss"] = self._loss.value
        values[prefix + "epoch"] = self._epoch
        wandb.log(values)

        super().clear()


class InferenceWandB(StatsLogger):
    def __init__(
        self,
        experiment_arguments: Dict,
        model,
        project: str = "experiment",
        name: str = "experiment_name",
        experiment_dir: str = None,
        watch: bool = False,
        log_frequency: int = 10,
        num_modes: int = 1,
    ):
        super().__init__()
        self.project = project
        self.experiment_name = name
        self.experiment_dir = experiment_dir
        self.watch = watch
        self.log_frequency = log_frequency
        self._mode = 0
        self._num_modes = num_modes
        self._modes_dict = {i: f"mode_{i}_" for i in range(self._num_modes)}
        # Login to wandb
        wandb.login()

        # Init the run
        wandb.init(
            project=(self.project or None),
            name=(self.experiment_name or None),
            config=dict(experiment_arguments.items()),
            dir=experiment_dir,
        )

        if self.watch:
            wandb.watch(model, log_freq=self.log_frequency)

    def print_progress(self, epoch, batch, loss, precision=".5f"):
        super().print_progress(epoch, batch, loss, precision)
        self._mode = epoch

    def add_media(self, epoch, media, media_type: str):
        self._values[media_type] = media
        self._mode = epoch

    def clear(self):
        # Before clearing everything out send it to wandb
        prefix = self._modes_dict[self._mode]
        values = {}
        for k, v in self._values.items():
            key = prefix + k
            if isinstance(v, AverageAggregator):
                value = v.value
            else:
                value = v
            values[key] = value
        values[prefix + "loss"] = self._loss.value
        values[prefix + "epoch"] = self._mode
        wandb.log(values)

        super().clear()
