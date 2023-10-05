#!/usr/bin/env python
# coding: utf-8


import sys

from sklearn.exceptions import UndefinedMetricWarning
if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")
    
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)



from pathlib import Path
from typing import List, Tuple, Any

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import plotly.express as px
import plotly.graph_objects as go
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from typing import Union

from ssl_tools.transforms import *

from typing import Any
from torch.nn import TransformerEncoder, TransformerEncoderLayer


# Run parameters
num_epochs = 200
accelerator = "gpu"
num_gpus = 1
strategy = "ddp"
checkpoint_dir = Path("checkpoints/")
tfc_pretrain_checkpoint = checkpoint_dir / "tfc_pretrain_c"
tfc_classifier_checkpoint = checkpoint_dir / "tfc_classifier_c"
result_dir = Path("results/")
result_dir.mkdir(exist_ok=True, parents=True)
results_tfc_classifier = result_dir / "tfc_classifier_c.yaml"
batch_size = 60

data_path = Path("data/TFC/Epilepsy")

# NXTent Loss
jitter_ratio = 2
length_alignment = 178
drop_last = True
num_workers = 10
learning_rate = 3e-4
temperature = 0.2
use_cosine_similarity = True
is_subset = False
n_classes = 5

class TFCContrastiveDataset(Dataset):
    def __init__(
        self,
        data: torch.Tensor,
        labels: torch.Tensor = None,
        length_alignment: int = 178,
        time_transforms: Union[Transform, List[Transform]] = None,
        frequency_transforms: Union[Transform, List[Transform]] = None,
    ):
        assert len(data) == len(labels), "Data and labels must have the same length"
        
        self.data_time = data
        self.labels = labels
        self.length_alignment = length_alignment
        self.time_transforms = time_transforms or []
        self.frequency_transforms = frequency_transforms or []
        
        if not isinstance(self.time_transforms, list):
            self.time_transforms = [self.time_transforms]
        if not isinstance(self.frequency_transforms, list):
            self.frequency_transforms = [self.frequency_transforms]

        if len(self.data_time.shape) < 3:
            self.data_time = self.data_time.unsqueeze(2)

        if self.data_time.shape.index(min(self.data_time.shape)) != 1:
            self.data_time = self.data_time.permute(0, 2, 1)

        """Align the data to the same length, removing the extra features"""
        self.data_time = self.data_time[:, :1, : self.length_alignment]
        
        """Calculcate the FFT of the data and apply the transforms (if any)"""
        self.data_freq = torch.fft.fft(self.data_time).abs()
        
        # This could be done in the __getitem__ method
        # For now, we do it here to be more similar to the original implementation
        self.data_time_augmented = self.apply_transforms(self.data_time, self.time_transforms)
        self.data_freq_augmented = self.apply_transforms(self.data_freq, self.frequency_transforms)
        
    def apply_transforms(self, x: torch.Tensor, transforms: List[Transform]) -> torch.Tensor:
        for transform in transforms:
            x = transform.fit_transform(x)
        return x
        
    def __len__(self):
        return len(self.data_time)
    
    def __getitem__(self, index):
        # Time processing
        return (
            self.data_time[index].float(),
            self.labels[index],
            self.data_time_augmented[index].float(),
            self.data_freq[index].float(),
            self.data_freq_augmented[index].float(),
        )


class NTXentLoss_poly(pl.LightningModule):
    def __init__(
        self,
        batch_size,
        temperature: float = 0.2,
        use_cosine_similarity: bool = True,
    ):
        super(NTXentLoss_poly, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(
            2 * self.batch_size, -1
        )

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        """Criterion has an internal one-hot function. Here, make all positives as 1 while all negatives as 0. """
        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        CE = self.criterion(logits, labels)

        onehot_label = (
            torch.cat(
                (
                    torch.ones(2 * self.batch_size, 1),
                    torch.zeros(2 * self.batch_size, negatives.shape[-1]),
                ),
                dim=-1,
            )
            .to(self.device)
            .long()
        )
        # Add poly loss
        pt = torch.mean(onehot_label * torch.nn.functional.softmax(logits, dim=-1))

        epsilon = self.batch_size
        # loss = CE/ (2 * self.batch_size) + epsilon*(1-pt) # replace 1 by 1/self.batch_size
        loss = CE / (2 * self.batch_size) + epsilon * (1 / self.batch_size - pt)
        # loss = CE / (2 * self.batch_size)

        return loss


class TFC(pl.LightningModule):
    def __init__(
        self,
        time_encoder: nn.Module,
        frequency_encoder: nn.Module,
        time_projector: nn.Module,
        frequency_projector: nn.Module,
        nxtent_criterion: nn.Module,
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
    

from typing import Any

from torchmetrics.functional import accuracy


class SimpleClassifier(torch.nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.num_classes = num_classes
        self.fc = torch.nn.Linear(2 * 128, 64)
        self.fc2 = torch.nn.Linear(64, num_classes)

    def forward(self, x):
        emb_flatten = x.reshape(x.shape[0], -1)
        x = self.fc(emb_flatten)
        x = torch.sigmoid(x)
        y = self.fc2(x)
        return y


class TFC_classifier(pl.LightningModule):
    def __init__(
        self,
        tfc_model: torch.nn.Module,
        classifier: torch.nn.Module,
        nxtent_criterion: nn.Module,
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


def build_model():
    time_encoder = TransformerEncoder(
        TransformerEncoderLayer(
            length_alignment, dim_feedforward=2 * length_alignment, nhead=2
        ),
        num_layers=2,
    )
    frequency_encoder = TransformerEncoder(
        TransformerEncoderLayer(
            length_alignment, dim_feedforward=2 * length_alignment, nhead=2
        ),
        num_layers=2,
    )

    time_projector = nn.Sequential(
        nn.Linear(length_alignment, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Linear(256, 128),
    )
    frequency_projector = nn.Sequential(
        nn.Linear(length_alignment, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Linear(256, 128),
    )

    nxtent = NTXentLoss_poly(
        batch_size=batch_size,
        temperature=temperature,
        use_cosine_similarity=use_cosine_similarity,
    )

    tfc_model = TFC(
        time_encoder=time_encoder,
        frequency_encoder=frequency_encoder,
        time_projector=time_projector,
        frequency_projector=frequency_projector,
        nxtent_criterion=nxtent,
        lr=learning_rate,
    )
    
    tfc_model.load_state_dict(torch.load(tfc_pretrain_checkpoint / "last.ckpt")["state_dict"])
    
    classifier = SimpleClassifier(num_classes=n_classes)

    tfc_classifier = TFC_classifier(
        tfc_model=tfc_model,
        classifier=classifier,
        nxtent_criterion=nxtent,
        lr=learning_rate,
        n_classes=n_classes,
    )
    
    return tfc_classifier


def get_data():
    dataset_train = torch.load(data_path / "train.pt")
    dataset_validation = torch.load(data_path / "val.pt")
    dataset_test = torch.load(data_path / "test.pt")
    
    X_train, y_train = dataset_train["samples"], dataset_train["labels"]
    X_validation, y_validation = dataset_validation["samples"], dataset_validation["labels"]
    X_test, y_test = dataset_test["samples"], dataset_test["labels"]
    
    if is_subset:
        size = batch_size * 10
        X_train = X_train[:size]
        y_train = y_train[:size]

    train_dataset = TFCContrastiveDataset(
        data=X_train,
        labels=y_train,
        time_transforms=None,
        frequency_transforms=None,
    )

    validation_dataset = TFCContrastiveDataset(
        data=X_validation,
        labels=y_validation,
        time_transforms=None,
        frequency_transforms=None,
    )

    test_dataset = TFCContrastiveDataset(
        data=X_test,
        labels=y_test,
        time_transforms=None,
        frequency_transforms=None,
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        num_workers=num_workers
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=drop_last,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=drop_last,
        num_workers=num_workers
    )
    return train_loader, validation_loader, test_loader


def main():
    tfc_classifier = build_model()
    train_loader, validation_loader, test_loader = get_data()

    checkpoint_callback = ModelCheckpoint(
        dirpath=tfc_classifier_checkpoint,
        monitor="val_acc",
        save_last=True,
        mode="max",
    )

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator=accelerator,
        devices=num_gpus,
        callbacks=[checkpoint_callback],
        strategy=strategy
    )

    print("Fit classifier")
    trainer.fit(tfc_classifier, train_loader, validation_loader)

    print("Test classifier")
    res = trainer.test(tfc_classifier, test_loader)
    
    print(res)
    import yaml
    with results_tfc_classifier.open("w") as f:
        yaml.dump(res, f)

if __name__ == "__main__":
    main()