#!/usr/bin/env python3

from dataclasses import dataclass
from lightning.pytorch.cli import (
    ArgsType,
    LightningCLI,
    LightningArgumentParser,
    instantiate_class
)

from torchmetrics import Accuracy

def cli_main():
    cli = LightningCLI(run=True)
    return cli


if __name__ == "__main__":
    cli_main()