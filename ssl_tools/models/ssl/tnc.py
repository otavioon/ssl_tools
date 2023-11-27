import torch
import lightning as L
import numpy as np

class TNC(L.LightningModule):
    def __init__(
        self, discriminator, encoder, mc_sample_size, w, learning_rate=1e-3
    ):
        super().__init__()
        self.discriminator = discriminator
        self.encoder = encoder
        self.mc_sample_size = mc_sample_size
        self.w = w
        self.learning_rate = learning_rate
        self.loss_func = torch.nn.BCEWithLogitsLoss()

    def loss_function(self, y, y_hat):
        return self.loss_func(y, y_hat)

    def forward(self, x):
        return self.encoder(x)

    def _shared_step(self, x_t, x_p, x_n):
        batch_size, f_size, len_size = x_t.shape
        x_p = x_p.reshape((-1, f_size, len_size))
        x_n = x_n.reshape((-1, f_size, len_size))
        x_t = torch.repeat_interleave(x_t, self.mc_sample_size, axis=0)
        neighbors = torch.ones((len(x_p)), device=self.device)
        non_neighbors = torch.zeros((len(x_n)), device=self.device)

        # Encoding features
        z_t = self.forward(x_t)
        z_p = self.forward(x_p)
        z_n = self.forward(x_n)

        # Discriminate features
        d_p = self.discriminator(z_t, z_p)
        d_n = self.discriminator(z_t, z_n)

        # Compute loss
        p_loss = self.loss_function(d_p, neighbors)
        n_loss = self.loss_function(d_n, non_neighbors)
        n_loss_u = self.loss_function(d_n, neighbors)
        loss = (p_loss + self.w * n_loss_u + (1 - self.w) * n_loss) / 2
        return loss

    def training_step(self, batch, batch_idx):
        x_t, x_p, x_n = batch
        loss = self._shared_step(x_t, x_p, x_n)
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x_t, x_p, x_n = batch
        loss = self._shared_step(x_t, x_p, x_n)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        x_t, x_p, x_n = batch
        loss = self._shared_step(x_t, x_p, x_n)
        self.log("test_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def get_config(self) -> dict:
        return {
            "mc_sample_size": self.mc_sample_size,
            "w": self.w,
            "learning_rate": self.learning_rate,
        }










# from typing import Any
# import torch
# import pytorch_lightning as pl
# from torchmetrics.functional import accuracy


# class TNC(pl.LightningModule):
#     def __init__(
#         self,
#         encoder: torch.nn.Module,
#         discriminator: torch.nn.Module,
#         mc_sample_size: int = 20,
#         window_size: int = 4,
#         w: float = 0.05,
#         lr: float = 1e-3,
#         weight_decay: float = 1e-5,
#     ):
#         super().__init__()
#         self.encoder = encoder
#         self.discriminator = discriminator
#         self.mc_sample_size = mc_sample_size
#         self.window_size = window_size
#         self.w = w
#         self.learning_rate = lr
#         self.weight_decay = weight_decay
#         self.loss_func = torch.nn.BCEWithLogitsLoss()
        
#     def loss_function(self, y, y_hat):
#         return self.loss_func(y_hat, y)      

#     def training_step(self, batch, batch_idx):
#         x_t, x_p, x_n, _ = batch
#         mc_sample = x_p.shape[1]
#         batch_size, f_size, len_size = x_t.shape
#         x_p = x_p.view(-1, f_size, len_size)
#         x_n = x_n.view(-1, f_size, len_size)
#         x_t = x_t.repeat(mc_sample, 1, 1)
#         neighbors = torch.ones(len(x_p)).to(self.device)
#         non_neighbors = torch.zeros(len(x_n)).to(self.device)
#         x_t, x_p, x_n = (
#             x_t.to(self.device),
#             x_p.to(self.device),
#             x_n.to(self.device),
#         )

#         z_t = self.encoder(x_t)
#         z_p = self.encoder(x_p)
#         z_n = self.encoder(x_n)

#         d_p = self.discriminator(z_t, z_p)
#         d_n = self.discriminator(z_t, z_n)

#         p_loss = self.loss_func(d_p, neighbors)
#         n_loss = self.loss_func(d_n, non_neighbors)
#         n_loss_u = self.loss_func(d_n, neighbors)
#         loss = (p_loss + self.w * n_loss_u + (1 - self.w) * n_loss) / 2

#         self.training_step_losses.append(loss)
#         return loss

#     def on_train_epoch_end(self) -> None:
#         # do something with all training_step outputs, for example:
#         epoch_mean = torch.stack(self.training_step_losses).mean()
#         self.log(
#             "train_loss",
#             epoch_mean,
#             on_epoch=True,
#             on_step=False,
#             prog_bar=True,
#             logger=True,
#         )
#         # free up the memory
#         self.training_step_losses.clear()

#     def configure_optimizers(self):
#         learnable_parameters = list(self.discriminator.parameters()) + list(
#             self.encoder.parameters()
#         )

#         optimizer = torch.optim.Adam(
#             learnable_parameters,
#             lr=self.learning_rate,
#             weight_decay=self.weight_decay,
#         )
#         return optimizer
    
    
    


# # class TNC_Classifier(pl.LightningModule):
# #     def __init__(
# #         self,
# #         encoder: torch.nn.Module,
# #         classifier: torch.nn.Module,
# #         lr: float = 1e-3,
# #         weight_decay: float = 0.0,
# #         task_class: str = "multiclass",
# #         num_classes: int = 6,
# #     ):
# #         super().__init__()
# #         self.encoder = encoder.to(self.device)
# #         self.classifier = classifier.to(self.device)
# #         self.learning_rate = lr
# #         self.weight_decay = weight_decay
# #         self.training_step_losses = []
# #         self.validation_step_losses = []
# #         self.loss_function = torch.nn.CrossEntropyLoss()
# #         self.task_class = task_class
# #         self.num_classes = num_classes

# #     def configure_optimizers(self) -> Any:
# #         optimizer = torch.optim.Adam(
# #             self.classifier.parameters(),
# #             lr=self.learning_rate,
# #             weight_decay=self.weight_decay,
# #         )
# #         return optimizer

# #     def forward(self, x):
# #         encodings = self.encoder(x)
# #         predictions = self.classifier(encodings)
# #         return predictions

# #     def training_step(self, batch, batch_idx):
# #         x, y = batch
# #         predictions = self.forward(x)
# #         loss = self.loss_function(predictions, y.long())
# #         self.training_step_losses.append(loss)
# #         return loss

# #     def on_train_epoch_end(self) -> None:
# #         # do something with all training_step outputs, for example:
# #         epoch_mean = torch.stack(self.training_step_losses).mean()
# #         self.log(
# #             "train_loss",
# #             epoch_mean,
# #             on_epoch=True,
# #             on_step=False,
# #             prog_bar=True,
# #             logger=True,
# #         )
# #         # free up the memory
# #         self.training_step_losses.clear()

# #     def validation_step(self, batch, batch_idx):
# #         loss, acc = self._shared_eval_step(batch, batch_idx)
# #         self.validation_step_losses.append(loss)
# #         metrics = {"val_acc": acc, "val_loss": loss}
# #         self.log_dict(metrics)
# #         return metrics

# #     def test_step(self, batch, batch_idx):
# #         loss, acc = self._shared_eval_step(batch, batch_idx)
# #         metrics = {"test_acc": acc, "test_loss": loss}
# #         self.log_dict(metrics)
# #         return metrics

# #     def on_validation_epoch_end(self) -> None:
# #         # do something with all training_step outputs, for example:
# #         epoch_mean = torch.stack(self.validation_step_losses).mean()
# #         self.log(
# #             "val_loss",
# #             epoch_mean,
# #             on_epoch=True,
# #             on_step=False,
# #             prog_bar=True,
# #             logger=True,
# #         )
# #         # free up the memory
# #         self.validation_step_losses.clear()

# #     def _shared_eval_step(self, batch, batch_idx):
# #         x, y = batch
# #         predictions = self.forward(x)
# #         loss = self.loss_function(predictions, y.long())
# #         acc = accuracy(
# #             torch.argmax(predictions, dim=1),
# #             y.long(),
# #             task=self.task_class,
# #             num_classes=self.num_classes,
# #         )
# #         return loss, acc