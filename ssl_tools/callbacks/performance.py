import lightning as L
from lightning.pytorch.callbacks import Callback
import time


class PerformanceLog(Callback):
    def __init__(self):
        super().__init__()
        self.train_epoch_start_time = None
        self.fit_start_time = None

    def on_train_epoch_start(
        self, trainer: L.Trainer, module: L.LightningModule
    ):
        """Called when the train epoch begins."""
        self.train_epoch_start_time = time.time()

    def on_train_epoch_end(self, trainer: L.Trainer, module: L.LightningModule):
        """Called when the train epoch ends.

        To access all batch outputs at the end of the epoch, you can cache step outputs as an attribute of the
        :class:`lightning.pytorch.core.LightningModule` and access them in this hook:

        .. code-block:: python

            class MyLightningModule(L.LightningModule):
                def __init__(self):
                    super().__init__()
                    self.training_step_outputs = []

                def training_step(self):
                    loss = ...
                    self.training_step_outputs.append(loss)
                    return loss


            class MyCallback(L.Callback):
                def on_train_epoch_end(self, trainer, pl_module):
                    # do something with all training_step outputs, for example:
                    epoch_mean = torch.stack(pl_module.training_step_outputs).mean()
                    pl_module.log("training_epoch_mean", epoch_mean)
                    # free up the memory
                    pl_module.training_step_outputs.clear()
        """
        end = time.time()
        duration = end - self.train_epoch_start_time
        module.log(
            "train_epoch_time",
            duration,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=False,
        )
        self.train_epoch_start_time = end

    def on_fit_start(self, trainer: L.Trainer, module: L.LightningModule) -> None:
        """Called when fit begins."""
        self.fit_start_time = time.time()

    def on_fit_end(self, trainer: L.Trainer, module: L.LightningModule) -> None:
        """Called when fit ends."""
        end = time.time()
        duration = end - self.fit_start_time
        print(f"--> Overall fit time: {duration:.3f} seconds")