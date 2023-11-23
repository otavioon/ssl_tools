#!/bin/env python

import numpy as np
import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import lightning as L


class RandomDataset:
    def __init__(self, num_elements: int = 32, shape: tuple = (28, 28)):
        self.num_elements = num_elements
        self.shape = shape

    def __len__(self):
        return self.num_elements

    def __getitem__(self, idx):
        return np.random.rand(*self.shape).astype("float32"), np.random.randint(0, 10)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, x):
        return self.l1(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

    def forward(self, x):
        past = torch.zeros(
            *x.shape,
            device=x.device,
        )
        
        x = x * past
        
        return self.l1(x)


class LitAutoEncoder(L.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def main():
    # Init DataLoader from MNIST Dataset
    dataset = RandomDataset(num_elements=512)
    train_loader = DataLoader(dataset, batch_size=32)

    # Init our model
    encoder = Encoder()
    decoder = Decoder()
    autoencoder = LitAutoEncoder(encoder, decoder)

    # Initialize a trainer
    trainer = L.Trainer(
        max_epochs=3, accelerator="gpu", devices=1, strategy="auto"
    )

    # Train the model ⚡
    trainer.fit(model=autoencoder, train_dataloaders=train_loader)


if __name__ == "__main__":
    main()
