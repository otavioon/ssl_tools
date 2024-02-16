import lightning as L
from ssl_tools.experiments import LightningTest
from ssl_tools.models.nets.simple import SimpleClassificationNet
import torch
import torchmetrics
import pandas as pd
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
import plotly.graph_objects as go

from functools import partial
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


def get_split_dataloader(
    stage: str, data_module: L.LightningDataModule
) -> DataLoader:
    if stage == "train":
        data_module.setup("fit")
        return data_module.train_dataloader()
    elif stage == "validation":
        data_module.setup("fit")
        return data_module.val_dataloader()
    elif stage == "test":
        data_module.setup("test")
        return data_module.test_dataloader()
    else:
        raise ValueError(f"Invalid stage: {stage}")


def full_dataset_from_dataloader(dataloader: DataLoader):
    return dataloader.dataset[:]


def get_full_data_split(
    data_module: L.LightningDataModule,
    stage: str,
):
    dataloader = get_split_dataloader(stage, data_module)
    return full_dataset_from_dataloader(dataloader)


def generate_embeddings(
    model: SimpleClassificationNet,
    dataloader: DataLoader,
    trainer: L.Trainer,
):
    old_fc = model.fc
    model.fc = torch.nn.Identity()
    embeddings = trainer.predict(model, dataloader)
    embeddings = torch.cat(embeddings)
    model.fc = old_fc
    return embeddings


class EvaluatorBase(LightningTest):
    def __init__(
        self,
        results_file: str = "results.csv",
        confusion_matrix_file: str = "confusion_matrix.csv",
        confusion_matrix_image_file: str = "confusion_matrix.png",
        tsne_plot_file: str = "tsne_embeddings.png",
        embedding_file: str = "embeddings.csv",
        predictions_file: str = "predictions.csv",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.results_file = results_file
        self.confusion_matrix_file = confusion_matrix_file
        self.confusion_matrix_image_file = confusion_matrix_image_file
        self.tsne_plot_file = tsne_plot_file
        self.embedding_file = embedding_file
        self.predictions_file = predictions_file
        self._sklearn_models = {
            "random_forest-100": partial(
                RandomForestClassifier, n_estimators=100, random_state=42
            ),
            "svm": partial(SVC, random_state=42),
            "knn-5": partial(KNeighborsClassifier, n_neighbors=5),
        }

    def _compute_embeddings(self, model, data_module, trainer):
        old_fc = model.fc
        model.fc = torch.nn.Identity()
        embeddings = trainer.predict(model, datamodule=data_module)
        embeddings = torch.cat(embeddings)
        embeddings = embeddings.reshape(embeddings.shape[0], -1)
        model.fc = old_fc
        return embeddings

    def _compute_classification_metrics(
        self, y_hat_logits: torch.Tensor, y: torch.Tensor, n_classes: int
    ) -> pd.DataFrame:
        results = {
            "accuracy": [
                torchmetrics.functional.accuracy(
                    y_hat_logits, y, num_classes=n_classes, task="multiclass"
                ).item()
            ],
            "f1": [
                torchmetrics.functional.f1_score(
                    y_hat_logits, y, num_classes=n_classes, task="multiclass"
                ).item()
            ],
            # "roc_auc": [
            #     torchmetrics.functional.auroc(
            #         y_hat_logits, y, num_classes=n_classes, task="multiclass"
            #     ).item()
            # ],
        }
        return pd.DataFrame(results)

    def _plot_confusion_matrix(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
        n_classes: int,
        cm_file: str,
        cm_image_file: str,
    ):
        cm = torchmetrics.functional.confusion_matrix(
            y_hat, y, num_classes=n_classes, normalize="true", task="multiclass"
        )

        pd.DataFrame(cm).to_csv(cm_file, index=False)
        print(f"Confusion matrix saved to {cm_file}")

        classes = [str(c) for c in sorted(y_hat.unique().tolist())]
        fig = go.Figure(data=go.Heatmap(z=cm, x=classes, y=classes))
        fig.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted",
            yaxis_title="True",
            legend=dict(title="Classes"),
            showlegend=True,
        )
        fig.write_image(
            cm_image_file,
            width=1.5 * 600,
            height=1.5 * 600,
            scale=1,
        )
        print(f"Confusion matrix image saved to {cm_image_file}")

        return cm

    def _plot_tnse_embeddings(
        self,
        embeddings: torch.Tensor,
        y: torch.Tensor,
        y_hat: torch.Tensor,
        n_components: int = 2,
        tsne_plot_file: str = "tsne_embeddings.png",
    ):
        tsne = TSNE(n_components=n_components)
        embeddings_tsne = tsne.fit_transform(embeddings)

        # Colorize embeddings based on y
        colors = y

        # Create a list to store marker symbols
        markers = []

        # Iterate over y and y_hat to determine marker symbols
        for i in range(len(y)):
            if y[i] == y_hat[i]:
                markers.append("circle")
            else:
                markers.append("cross")

        # Create a scatter plot
        fig = go.Figure()
        markers = np.array(markers)

        # Add markers to the scatter plot
        unique_labels = torch.unique(y)
        for label in unique_labels:
            mask = (y == label).squeeze()
            fig.add_trace(
                go.Scatter(
                    x=embeddings_tsne[mask, 0],
                    y=embeddings_tsne[mask, 1],
                    mode="markers",
                    name=f"Class {label.item()}",
                    marker=dict(color=label.item(), symbol=markers[mask]),
                )
            )

        fig.update_layout(
            title="T-SNE Embeddings",
            xaxis_title="Component 1",
            yaxis_title="Component 2",
            legend=dict(title="Classes"),
        )

        fig.write_image(
            tsne_plot_file,
            width=1.5 * 600,
            height=1.5 * 600,
            scale=1,
        )
        print(f"T-SNE plot saved to {tsne_plot_file}")
        return embeddings_tsne

    def predict(self, model, dataloader, trainer):
        y_hat = trainer.predict(model, dataloader)
        y_hat = torch.cat(y_hat)
        return y_hat

    def evaluate_model_performance(self, model, data_module, trainer):
        for stage in ["validation", "test"]:
            # ------------ Generate required data ------------
            # Get dataloader
            dataloader = get_split_dataloader(stage, data_module)
            # Get labels
            _, y = full_dataset_from_dataloader(dataloader)
            y = torch.LongTensor(y)
            # Predict
            y_hat_logits = self.predict(model, dataloader, trainer)
            y_hat = torch.argmax(y_hat_logits, dim=1)
            # Get embeddings
            embeddings = generate_embeddings(model, dataloader, trainer)
            embeddings = embeddings.reshape(embeddings.shape[0], -1)
            # Get number of classes
            n_classes = len(torch.unique(y))

            # ------------ Evaluation ------------
            # Create split folder
            split_folder = self.experiment_dir / f"split_{stage}"
            split_folder.mkdir(parents=True, exist_ok=True)

            # Generate predictions CSV
            predictions = pd.DataFrame(
                {
                    "y": y.numpy().reshape(-1),
                    "y_hat": y_hat.numpy().reshape(-1),
                }
            )
            predictions.to_csv(
                split_folder / self.predictions_file, index=False
            )
            print(f"Predictions saved to {split_folder/self.predictions_file}")

            # Classification metrics
            clasification_results = self._compute_classification_metrics(
                y_hat_logits, y, n_classes
            )
            clasification_results.to_csv(
                split_folder / self.results_file, index=False
            )
            print(f"Results saved to {split_folder/self.results_file}")

            # Confusion matrix
            self._plot_confusion_matrix(
                y_hat,
                y,
                n_classes,
                split_folder / self.confusion_matrix_file,
                split_folder / self.confusion_matrix_image_file,
            )

            # TSNE plot
            self._plot_tnse_embeddings(
                embeddings,
                y,
                y_hat,
                n_components=2,
                tsne_plot_file=split_folder / self.tsne_plot_file,
            )


    def evaluate_embeddings(self, model, data_module, trainer):
        train_loader = get_split_dataloader("train", data_module)
        val_loader = get_split_dataloader("validation", data_module)
        test_loader = get_split_dataloader("test", data_module)

        # Generate embeddings
        X_train_emb = generate_embeddings(model, train_loader, trainer)
        X_train_emb = X_train_emb.reshape(X_train_emb.shape[0], -1)
        
        X_val_emb = generate_embeddings(model, val_loader, trainer)
        X_val_emb = X_val_emb.reshape(X_val_emb.shape[0], -1)
        
        X_test_emb = generate_embeddings(model, test_loader, trainer)
        X_test_emb = X_test_emb.reshape(X_test_emb.shape[0], -1)

        # Get data
        X_train, y_train = full_dataset_from_dataloader(train_loader)
        X_val, y_val = full_dataset_from_dataloader(val_loader)
        X_test, y_test = full_dataset_from_dataloader(test_loader)
        
        # Get number of classes
        n_classes = len(np.unique(y_train))

        
        # Train using sklearn models 
        for model_name, model_cls in self._sklearn_models.items():
            model = model_cls()
            print(f"Training {model_name} on train....")
            model.fit(X_train_emb, y_train)
            
            ######### Evaluate on validation
            y_hat_val = model.predict(X_val_emb)
            # Create split folder
            split_folder = self.experiment_dir / f"sklearn_{model_name}_train_on_train_test_on_validation"
            split_folder.mkdir(parents=True, exist_ok=True)

            # Generate predictions CSV
            predictions = pd.DataFrame(
                {
                    "y": y_val.reshape(-1),
                    "y_hat": y_hat_val.reshape(-1),
                }
            )
            predictions.to_csv(
                split_folder / self.predictions_file, index=False
            )
            print(f"Predictions saved to {split_folder/self.predictions_file}")

            # Classification metrics
            clasification_results = self._compute_classification_metrics(
                torch.LongTensor(y_hat_val), torch.LongTensor(y_val), n_classes
            )
            clasification_results.to_csv(
                split_folder / self.results_file, index=False
            )
            print(f"Results saved to {split_folder/self.results_file}")

            # Confusion matrix
            self._plot_confusion_matrix(
                torch.LongTensor(y_hat_val),
                torch.LongTensor(y_val),
                n_classes,
                split_folder / self.confusion_matrix_file,
                split_folder / self.confusion_matrix_image_file,
            )
            
            
            ############# Evaluate on test
            y_hat_test = model.predict(X_test_emb)
            
            split_folder = self.experiment_dir / f"sklearn_{model_name}_train_on_train_test_on_test"
            split_folder.mkdir(parents=True, exist_ok=True)
            
            # Generate predictions CSV
            predictions = pd.DataFrame(
                {
                    "y": y_test.reshape(-1),
                    "y_hat": y_hat_test.reshape(-1),
                }
            )
            
            predictions.to_csv(
                split_folder / self.predictions_file, index=False
            )
            
            print(f"Predictions saved to {split_folder/self.predictions_file}")
            
            # Classification metrics
            clasification_results = self._compute_classification_metrics(
                torch.LongTensor(y_hat_test), torch.LongTensor(y_test), n_classes
            )
            
            clasification_results.to_csv(
                split_folder / self.results_file, index=False
            )
            
            print(f"Results saved to {split_folder/self.results_file}")
            
            # Confusion matrix
            self._plot_confusion_matrix(
                torch.LongTensor(y_hat_test),
                torch.LongTensor(y_test),
                n_classes,
                split_folder / self.confusion_matrix_file,
                split_folder / self.confusion_matrix_image_file,
            )
            
                
        # Concatenate train and validation
        X_train_val_emb = torch.cat([X_train_emb, X_val_emb])
        y_train_val = np.concatenate([y_train, y_val])
        
        for model_name, model_cls in self._sklearn_models.items():
            model = model_cls()
            print(f"Training {model_name} on train+val....")
            model.fit(X_train_val_emb, y_train_val)
            y_hat_test = model.predict(X_test_emb)
            
            split_folder = self.experiment_dir / f"sklearn_{model_name}_train_on_train+val_test_on_test"
            split_folder.mkdir(parents=True, exist_ok=True)
            
            # Generate predictions CSV
            predictions = pd.DataFrame(
                {
                    "y": y_test.reshape(-1),
                    "y_hat": y_hat_test.reshape(-1),
                }
            )
            
            predictions.to_csv(
                split_folder / self.predictions_file, index=False
            )
            
            print(f"Predictions saved to {split_folder/self.predictions_file}")
            
            # Classification metrics
            clasification_results = self._compute_classification_metrics(
                torch.LongTensor(y_hat_test), torch.LongTensor(y_test), n_classes
            )
            
            clasification_results.to_csv(
                split_folder / self.results_file, index=False
            )
            
            print(f"Results saved to {split_folder/self.results_file}")
            
            # Confusion matrix
            self._plot_confusion_matrix(
                torch.LongTensor(y_hat_test),
                torch.LongTensor(y_test),
                n_classes,
                split_folder / self.confusion_matrix_file,
                split_folder / self.confusion_matrix_image_file,
            )
        
        
            
            
    def run_model(
        self,
        model: SimpleClassificationNet,
        data_module: L.LightningDataModule,
        trainer: L.Trainer,
    ):
        self.evaluate_model_performance(model, data_module, trainer)
        self.evaluate_embeddings(model, data_module, trainer)
        
        # self.evaluate_embeddings(model, data_module, trainer)

        # ---- Predictions ----
        # for stage in ["validation", "test"]:
        #     dataloader = self.get_split_dataloader(stage, data_module)

        # # Classification predictions
        # y_hat = trainer.predict(model, datamodule=data_module)
        # y_hat_logits = torch.cat(y_hat)
        # y_hat = torch.argmax(y_hat_logits, dim=1)

        # # Labels
        # y = list(y for x, y in trainer.predict_dataloaders)
        # y = torch.cat(y)
        # n_classes = len(torch.unique(y))

        # # Embedding
        # old_fc = model.fc
        # model.fc = torch.nn.Identity()
        # embeddings = trainer.predict(model, datamodule=data_module)
        # embeddings = torch.cat(embeddings)
        # embeddings = embeddings.reshape(embeddings.shape[0], -1)
        # model.fc = old_fc

        # print(
        #     f"-******Shape of y_hat: {y_hat.shape}, y: {y.shape}, embeddings: {embeddings.shape}"
        # )

        # # ---- Predictions ----

        # predictions = pd.DataFrame(
        #     {
        #         "y": y.numpy().reshape(-1),
        #         "y_hat": y_hat.numpy().reshape(-1),
        #     }
        # )
        # predictions.to_csv(self.predictions_file, index=False)
        # print(f"Predictions saved to {self.predictions_file}")

        # # Embeddings
        # embeddings = pd.DataFrame(embeddings.numpy())
        # embeddings.to_csv(self.embedding_file, index=False)
        # print(f"Embeddings saved to {self.embedding_file}")

        # # ---- Results ----
        # results = dict()
        # results["performance"] = self._compute_classification_metrics(
        #     y_hat_logits, y, n_classes
        # )
        # results["confusion_matrix"] = self._plot_confusion_matrix(
        #     y_hat, y, n_classes
        # )
        # results["tsne_embeddings"] = self._plot_tnse_embeddings(
        #     embeddings, y, y_hat, n_components=2
        # )
        # return results


class PredictionHeadClassifier(SimpleClassificationNet):
    def __init__(self, prediction_head: torch.nn.Module, num_classes: int = 6):
        super().__init__(
            backbone=torch.nn.Identity(),
            fc=prediction_head,
            flatten=True,
            loss_fn=torch.nn.CrossEntropyLoss(),
            val_metrics={
                "acc": torchmetrics.Accuracy(
                    task="multiclass", num_classes=num_classes
                )
            },
            test_metrics={
                "acc": torchmetrics.Accuracy(
                    task="multiclass", num_classes=num_classes
                )
            },
        )
