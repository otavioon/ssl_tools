#!/usr/bin/env python3

# %%
from simple1Dconv_classifier import Simple1DConvNetFineTune
import lightning as L
from evaluator import generate_embeddings, full_dataset_from_dataloader, EmbeddingEvaluator
from functools import partial
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from typing import List


from lightning.pytorch.callbacks import Callback
import torch

# %%
class PartialEmbeddingEvaluator(EmbeddingEvaluator):
    def __init__(
        self,
        experiment_name: str,
        model,
        data_module,
        trainer: L.Trainer,
        **kwargs,
    ):
        super().__init__(
            experiment_name=experiment_name,
            registered_model_name=None,
            **kwargs,
        )
        self.model = model
        self.data_module = data_module
        self.trainer = trainer
        self.experiment_name = experiment_name

    def run(self):
        return self.run_task(self.model, self.data_module, self.trainer)


class PartialEmbeddingEvaluatorCallback(Callback):
    def __init__(
        self,
        experiment_name: str,
        frequency: int = 1,
        **partal_embedding_evaluator_kwargs,
    ):
        self.experiment_name = experiment_name
        self.frequency = frequency
        self.partal_embedding_evaluator_kwargs = (
            partal_embedding_evaluator_kwargs
        )

    def on_validation_end(self, trainer: L.Trainer, pl_module):
        if trainer.sanity_checking:
            return
        
        if trainer.current_epoch == 0:
            return
        
        if ((trainer.current_epoch+1) % self.frequency) == 0:
            # with torch.no_grad():
                print("Running PartialEmbeddingEvaluator....")
                evaluator = PartialEmbeddingEvaluator(
                    experiment_name=self.experiment_name,
                    model=pl_module,
                    data_module=trainer.datamodule,
                    trainer=trainer,
                    add_epoch_info=True,
                    **self.partal_embedding_evaluator_kwargs,
                )
                evaluator.run()
                print("PartialEmbeddingEvaluator finished.")


class Simple1DConvNetFineTune2(Simple1DConvNetFineTune):
    def get_callbacks(self) -> List[Callback]:
        callbacks = super().get_callbacks()

        evaluator_callback = PartialEmbeddingEvaluatorCallback(
            experiment_name=self.experiment_name,
            frequency=1,
        )
        callbacks.append(evaluator_callback)
        print("******* Added PartialEmbeddingEvaluatorCallback *******")

        return callbacks

# %%
experiment = Simple1DConvNetFineTune2(
    data="/workspaces/hiaac-m4/ssl_tools/data/standartized_balanced/KuHar",
    experiment_name="test",
    model_name="conv1d_conss",
    accelerator="gpu",
    devices=1,
    batch_size=256,
    max_epochs=5,
    patience=10,
    checkpoint_monitor_metric="val_loss",
    num_workers=16,
    registered_model_name="simple1Dconv",
    registered_model_tags={
        "model": "simple1Dconv",
        "trained_on": "MotionSense",
        "stage": "train"
    },
    model_tags={
        "model": "simple1Dconv",
        "trained_on": "MotionSense",
        "finetune_on": "KuHar",
        "stage": "finetune",
        "test": True
    },
    limit_train_batches=1,
    limit_val_batches=1,
)

# %%
experiment.run()


