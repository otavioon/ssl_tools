#!/usr/bin/env python

from pathlib import Path
from jsonargparse import CLI
import pandas as pd
import plotly.graph_objects as go
import yaml


class PlotMetrics:
    def loss(
        self,
        experiment_dir: str,
        metrics_file: str = "metrics.csv",
        hyperparams_file: str = "hparams.yaml",
    ):
        experiment_dir = Path(experiment_dir)
        metrics = pd.read_csv(experiment_dir / metrics_file)

        train_loss_df = metrics[["epoch", "train_loss"]].dropna()
        train_loss_df["epoch"] = train_loss_df["epoch"].astype(int)
        train_loss_df["train_loss"] = train_loss_df["train_loss"].astype(float)

        val_loss_df = metrics[["epoch", "val_loss"]].dropna()
        val_loss_df["epoch"] = val_loss_df["epoch"].astype(int)
        val_loss_df["val_loss"] = val_loss_df["val_loss"].astype(float)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=train_loss_df["epoch"],
                y=train_loss_df["train_loss"],
                name="train loss",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=val_loss_df["epoch"],
                y=val_loss_df["val_loss"],
                name="val loss",
            )
        )
        
        fig.update_layout(title="Loss", xaxis_title="Epoch", yaxis_title="Loss")
        
        output_file = experiment_dir / "loss.png"
        fig.write_image(str(output_file))
        print(f"Figure saved to {output_file}")
        
    def accuracy(
        self,
        root_experiment_dir: str,
        results_file: str = "results.csv",
        hyperparams_file: str = "hparams.yaml",
        metric: str = "test_acc",
        train_dataset: str = "train",
    ):
        root_experiment_dir = Path(root_experiment_dir)
        
        fig = go.Figure()
       
        for path in sorted(root_experiment_dir.rglob(results_file)):
            experiment_dir = path.parent
            results = pd.read_csv(path)
            hyperparams = yaml.safe_load(
                (experiment_dir / hyperparams_file).read_text()
            )
            
            dataset_name = hyperparams["data"].split("/")[-1]
            accuracy = results[metric].values[-1]
            
            fig.add_trace(
                go.Bar(
                    x=[dataset_name],
                    y=[accuracy],
                    name=dataset_name
                )
            )
            
        fig.update_layout(title=f"Trained on {train_dataset}", xaxis_title="Dataset", yaxis_title=metric)
            
        output_file = root_experiment_dir / f"{train_dataset}_{metric}.png"
        fig.write_image(str(output_file))
        print(f"Figure saved to {output_file}")


def main():
    CLI(PlotMetrics, as_positional=False)


if __name__ == "__main__":
    main()
