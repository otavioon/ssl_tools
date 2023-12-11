#!/usr/bin/env python

from pathlib import Path
from jsonargparse import CLI
import pandas as pd
import plotly.graph_objects as go
import yaml
from collections.abc import Iterable
from typing import List


class PlotMetrics:
    """Class for plotting metrics from a training/predict run."""

    def epoch_loss(
        self,
        experiment_dir: str,
        losses: List[str] = ("train_loss", "val_loss"),
        metrics_file: str = "metrics.csv",
        title: str = "Loss",
    ):
        """Plot the loss over epochs.

        Parameters
        ----------
        experiment_dir : str
            The folder containing the metrics file.
        losses : List[str], optional
            The list of loss names to plot, by default ("train_loss",
            "val_loss")
        metrics_file : str, optional
            Name of the metrics_file. The file must be a csv file inside the
            ``experiment_dir``.
        title : str, optional
            Title of the plot.
        """
        experiment_dir = Path(experiment_dir)
        metrics = pd.read_csv(experiment_dir / metrics_file)
        if not isinstance(losses, Iterable):
            losses = [losses]

        fig = go.Figure()

        # Iterate over losses and plot them as a scatter plot in the same figure
        for loss_name in losses:
            loss_df = metrics[["epoch", loss_name]].dropna()
            loss_df["epoch"] = loss_df["epoch"].astype(int)
            loss_df[loss_name] = loss_df[loss_name].astype(float)

            fig.add_trace(
                go.Scatter(
                    x=loss_df["epoch"],
                    y=loss_df[loss_name],
                    name=loss_name,
                )
            )
        fig.update_layout(title=title, xaxis_title="Epoch", yaxis_title="Loss")

        output_file = experiment_dir / f"loss.png"
        fig.write_image(str(output_file))
        print(f"Figure saved to {output_file}")

    def accuracy(
        self,
        root_experiment_dir: str,
        results_file: str = "results.csv",
        hyperparams_file: str = "hparams.yaml",
        metric: str = "test_acc",
        title: str = "Results",
    ):
        """Plot the accuracy of a multiple test runs in a single figure. This
        is useful for comparing the performance of a model trained on different
        datasets. The folder structure should be as follows:
        ```
        root_experiment_dir
        ├── experiment_1
        │   ├── hparams.yaml
        │   ├── results.csv
        │   └── ...
        ├── experiment_2
        │   ├── hparams.yaml
        │   ├── results.csv
        │   └── ...
        └── ...
        ```

        Parameters
        ----------
        root_experiment_dir : str
            The root folder containing the experiments.
        results_file : str, optional
            The name of the results file. It must be a csv file inside the
            ``root_experiment_dir``.
        hyperparams_file : str, optional
            The name of the hyperparamters file. It must be a csv file inside the
            ``root_experiment_dir``.
        metric : str, optional
            Name of the metric to plot.
        title : str, optional
            Title of the plot.
        """
        root_experiment_dir = Path(root_experiment_dir)

        fig = go.Figure()

        # Iterate over experiments and plot them as a bar plot in the same 
        # figure. Each experiment is a different bar.
        for path in sorted(root_experiment_dir.rglob(results_file)):
            experiment_dir = path.parent
            results = pd.read_csv(path)
            hyperparams = yaml.safe_load(
                (experiment_dir / hyperparams_file).read_text()
            )

            dataset_name = hyperparams["data"].split("/")[-1]
            accuracy = results[metric].values[-1]

            fig.add_trace(
                go.Bar(x=[dataset_name], y=[accuracy], name=dataset_name)
            )

        fig.update_layout(
            title=title, xaxis_title="Dataset", yaxis_title=metric
        )

        output_file = root_experiment_dir / f"{title}_{metric}.png"
        fig.write_image(str(output_file))
        print(f"Figure saved to {output_file}")


def main():
    CLI(PlotMetrics, as_positional=False)


if __name__ == "__main__":
    main()
