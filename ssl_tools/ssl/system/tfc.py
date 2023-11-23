from typing import Any
import pytorch_lightning as pl
import torch
from torchmetrics.functional import accuracy


class TFC(pl.LightningModule):
    def __init__(
        self,
        time_encoder: torch.nn.Module,
        frequency_encoder: torch.nn.Module,
        time_projector: torch.nn.Module,
        frequency_projector: torch.nn.Module,
        nxtent_criterion: torch.nn.Module,
        lr: float = 1e-3,
        loss_lambda: float = 0.2,
    ):
        super().__init__()

        self.time_encoder = time_encoder
        self.time_projector = time_projector
        self.frequency_encoder = frequency_encoder
        self.frequency_projector = frequency_projector
        self.nxtent_criterion = nxtent_criterion
        self.learning_rate = lr
        self.loss_lambda = loss_lambda

    def forward(self, x_in_t, x_in_f):
        """Use Transformer"""
        x = self.time_encoder(x_in_t)
        h_time = x.reshape(x.shape[0], -1)

        """Cross-space projector"""
        z_time = self.time_projector(h_time)

        """Frequency-based contrastive encoder"""
        f = self.frequency_encoder(x_in_f)
        h_freq = f.reshape(f.shape[0], -1)

        """Cross-space projector"""
        z_freq = self.frequency_projector(h_freq)

        return h_time, z_time, h_freq, z_freq

    def configure_optimizers(self) -> Any:
        learnable_parameters = (
            list(self.time_encoder.parameters()) +
            list(self.time_projector.parameters()) +
            list(self.frequency_encoder.parameters()) +
            list(self.frequency_projector.parameters())
        )
        optimizer = torch.optim.Adam(learnable_parameters, lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        data, labels, aug1, data_f, aug1_f = batch
        
        """Producing embeddings"""
        h_t, z_t, h_f, z_f = self.forward(data, data_f)
        h_t_aug, z_t_aug, h_f_aug, z_f_aug = self.forward(aug1, aug1_f)
        
        """Calculate losses"""
        loss_time = self.nxtent_criterion(h_t, h_t_aug)
        loss_freq = self.nxtent_criterion(h_f, h_f_aug)
        loss_consistency = self.nxtent_criterion(z_t, z_f)
        loss = (self.loss_lambda * (loss_time + loss_freq)) + loss_consistency
        
        # log loss, only to appear on epoch
        self.log('time_loss', loss_time, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('freq_loss', loss_freq, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('consistency_loss', loss_consistency, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    

class TFC_classifier(pl.LightningModule):
    def __init__(
        self,
        tfc_model: torch.nn.Module,
        classifier: torch.nn.Module,
        nxtent_criterion: torch.nn.Module,
        lr: float = 1e-3,
        loss_lambda: float = 0.1,
        n_classes: int = 2,
    ):
        super().__init__()
        self.tfc_model = tfc_model
        self.classifier = classifier
        self.nxtent_criterion = nxtent_criterion
        self.learning_rate = lr
        self.n_classes = n_classes
        self.loss_lambda = loss_lambda
        self.loss_func = torch.nn.CrossEntropyLoss()

    def configure_optimizers(self) -> Any:
        learnable_parameters = list(self.tfc_model.parameters()) + list(
            self.classifier.parameters()
        )
        optimizer = torch.optim.Adam(learnable_parameters, lr=self.learning_rate)
        return optimizer

    def forward(self, x_in_t, x_in_f):
        return self.tfc_model(x_in_t, x_in_f)

    def training_step(self, batch, batch_idx):
        data, labels, aug1, data_f, aug1_f = batch

        """Producing embeddings"""
        h_t, z_t, h_f, z_f = self(data, data_f)
        h_t_aug, z_t_aug, h_f_aug, z_f_aug = self(aug1, aug1_f)

        """Add supervised loss"""
        fea_concat = torch.cat((z_t, z_f), dim=1)
        predictions = self.classifier(fea_concat)
        # fea_concat_flat = fea_concat.reshape(fea_concat.shape[0], -1)

        """Calculate losses"""
        loss_time = self.nxtent_criterion(h_t, h_t_aug)
        loss_freq = self.nxtent_criterion(h_f, h_f_aug)
        loss_consistency = self.nxtent_criterion(z_t, z_f)
        loss_p = self.loss_func(predictions, labels)
        loss = loss_p + self.loss_lambda * (loss_time + loss_freq) + loss_consistency

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "val_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return {"val_loss": loss, "val_acc": acc}

    def test_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        self.log(
            "test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "test_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return {"test_loss": loss, "test_acc": acc}

    def _shared_eval_step(self, batch, batch_idx):
        data, labels, aug1, data_f, aug1_f = batch

        """Producing embeddings"""
        h_t, z_t, h_f, z_f = self(data, data_f)
        h_t_aug, z_t_aug, h_f_aug, z_f_aug = self(aug1, aug1_f)
        
        # print(h_t.shape, z_t.shape, h_f.shape, z_f.shape, h_t_aug.shape, z_t_aug.shape, h_f_aug.shape, z_f_aug.shape)
        loss_time = self.nxtent_criterion(h_t, h_t_aug)
        loss_freq = self.nxtent_criterion(h_f, h_f_aug)
        loss_consistency = self.nxtent_criterion(z_t, z_f)

        """Add supervised loss"""
        fea_concat = torch.cat((z_t, z_f), dim=1)
        predictions = self.classifier(fea_concat)
        loss_p = self.loss_func(predictions, labels)
        
        loss = loss_p + self.loss_lambda * (loss_time + loss_freq) + loss_consistency

        acc = accuracy(
            torch.argmax(predictions, dim=1),
            labels,
            task="multiclass",
            num_classes=self.n_classes,
        )

        return loss, acc
