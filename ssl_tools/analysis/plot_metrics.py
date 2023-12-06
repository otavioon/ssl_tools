#!/usr/bin/env python

from pathlib import Path
from jsonargparse import CLI
import pandas as pd
import plotly.graph_objects as go
import yaml


class PlotMetrics:
    def __init__(
        self,
        experiment_dir: str,
        metrics_file: str = "metrics.csv",
        hyperparams_file: str = "hparams.yaml",
        results_file: str = "results.csv",
    ):
        self.experiment_dir = Path(experiment_dir)
        self.metrics_file = metrics_file
        self.hyperparams_file = hyperparams_file
        self.results_file = results_file        

    def __call__(self):
        metrics = pd.read_csv(self.experiment_dir / self.metrics_file)
        hyperparams = yaml.safe_load((self.experiment_dir / self.hyperparams_file).read_text())
        # results = pd.read_csv(self.experiment_dir / self.results_file)
        
        train_loss_df = metrics[["epoch", "train_loss"]].dropna()
        train_loss_df["epoch"] = train_loss_df["epoch"].astype(int)
        train_loss_df["train_loss"] = train_loss_df["train_loss"].astype(float)
        
        val_loss_df = metrics[["epoch", "val_loss"]].dropna()
        val_loss_df["epoch"] = val_loss_df["epoch"].astype(int)
        val_loss_df["val_loss"] = val_loss_df["val_loss"].astype(float)
                
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train_loss_df["epoch"], y=train_loss_df["train_loss"], name="train loss"))
        fig.add_trace(go.Scatter(x=val_loss_df["epoch"], y=val_loss_df["val_loss"], name="val loss"))
        fig.update_layout(title="Loss", xaxis_title="Epoch", yaxis_title="Loss")
        # fig.show()
        # save
        fig.write_image(str(self.experiment_dir / "loss.png"))
        
        print(f"Figure saved to {self.experiment_dir / 'loss.png'}")
        
        
def main():
    CLI(PlotMetrics, as_positional=False)()
    
if __name__ == "__main__":
    main()