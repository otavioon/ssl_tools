from pathlib import Path
from typing import List, Union

import pandas as pd
import lightning as L
from torch.utils.data import DataLoader
from ssl_tools.data.datasets import MultiModalDataframeDataset


class CovidUserAnomalyDataModule(L.LightningDataModule):
    def __init__(
        self,
        # Dataset params
        data_path: Path,
        participants: Union[str, int, List[Union[str, int]]] = None,
        feature_column_prefix: str = "RHR",
        target_column: str = "anomaly",
        participant_column: str = "participant_id",
        include_recovered_in_test: bool = False,
        reshape: tuple = None,
        train_transforms: List[callable] = None,
        # Dataloader params
        batch_size: int = 32,
        num_workers: int = 1,
        validation_split: float = 0.2,
        dataset_transforms: List[callable] = None,
        shuffle_train: bool = True,
        discard_last_batch: bool = True,
        balance: bool = False,
    ):
        super().__init__()
        self.data_path = data_path
        self.participants = participants
        self.feature_column_prefix = feature_column_prefix
        self.target_column = target_column
        self.participant_column = participant_column
        self.include_recovered_in_test = include_recovered_in_test
        self.reshape = reshape
        self.train_transforms = train_transforms
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.validation_split = validation_split
        self.dataset_transforms = dataset_transforms
        self.shuffle_train = shuffle_train
        self.discard_last_batch = discard_last_batch
        self.balance = balance

    def setup(self, stage: str):
        data = pd.read_csv(self.data_path)

        assert (
            self.participant_column in data.columns
            and self.target_column in data.columns
        ), f"Columns {self.participant_column} and {self.target_column} must be in the dataframe"
        # assert self.validation_split > 0 and self.validation_split < 1, (
        #     "validation_split must be between 0 and 1"
        # )

        participant_ids = sorted(data[self.participant_column].unique())
        selected_participant_ids = []

        # Filter only selected participants
        if self.participants is None:
            selected_participant_ids = participant_ids
        else:
            if isinstance(self.participants, str) or isinstance(
                self.participants, int
            ):
                self.participants = [self.participants]
            for participant in self.participants:
                if isinstance(participant, int):
                    if participant < 0 or participant >= len(participant_ids):
                        raise ValueError(
                            f"participant_id must be between 0 and {len(participant_ids)}, not {participant}"
                        )
                    selected_participant_ids.append(
                        participant_ids[participant]
                    )
                elif isinstance(participant, str):
                    if participant not in participant_ids:
                        raise ValueError(
                            f"participant_id {participant} is not in the dataset"
                        )
                    selected_participant_ids.append(participant)
                else:
                    raise ValueError(
                        f"participant_id must be either int or str, not {type(participant)}"
                    )
        # Filter only selected participants
        data = data[
            data[self.participant_column].isin(selected_participant_ids)
        ]

        # Switch between train and test
        if stage == "fit":
            # Filter only baseline data
            # TODO add this
            # data = data[data["baseline"] == True]

            # train_test_split
            train_data = data.sample(frac=1 - self.validation_split)
            val_data = data.drop(train_data.index)

            self.train_dataset = MultiModalDataframeDataset(
                train_data,
                feature_column_prefix=self.feature_column_prefix,
                target_column=self.target_column,
                reshape=self.reshape,
                transforms=self.train_transforms,
                name=f"{self.participants}_train",
                dataset_transforms=self.dataset_transforms,
                balance=self.balance,
            )

            self.validation_dataset = MultiModalDataframeDataset(
                val_data,
                feature_column_prefix=self.feature_column_prefix,
                target_column=self.target_column,
                reshape=self.reshape,
                transforms=self.train_transforms,
                name=f"{self.participants}_validation",
            )

        else:
            # elif stage == "test" or stage == "predict":
            # Filter only non-baseline data
            data = data[data["baseline"] == False]
            if not self.include_recovered_in_test:
                data = data[data["label"] != "recovered"]
            self.test_dataset = MultiModalDataframeDataset(
                data,
                feature_column_prefix=self.feature_column_prefix,
                target_column=self.target_column,
                reshape=self.reshape,
                transforms=None,
                name=f"{self.participants}_{stage}",
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle_train,
            drop_last=self.discard_last_batch,
        )

    def val_dataloader(self):
        return DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=self.discard_last_batch,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=self.discard_last_batch,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=self.discard_last_batch,
        )
