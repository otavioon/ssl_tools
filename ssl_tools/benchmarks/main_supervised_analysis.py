#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import numpy as np
from jsonargparse import CLI
import tqdm


def analysis(
    results_dir: str,
    query: str = None,
    output_dir: str = None,
    result_file: str = "results.csv",
    print_results: bool = True,
):
    results_dir = Path(results_dir)
    result_files = list(results_dir.rglob("metrics.csv"))
    results = []

    if len(result_files) == 0:
        print("No results found")
        return
    else:
        print(f"Found {len(result_files)} results files")

    for f in tqdm.tqdm(
        result_files, total=len(result_files), desc="Parsing files"
    ):
        model, data_module, train_dataset, test_dataset = f.parents[
            1
        ].name.split("-")

        train_dataset = train_dataset.replace("train_on_", "")
        test_dataset = test_dataset.replace("test_on_", "")

        df = pd.read_csv(f)

        # Allow unfinished runs
        if "test_accuracy" not in df.columns:
            acc = 0.0
        else:
            acc = float(df["test_accuracy"].iloc[-1])

        results.append(
            {
                "model": model,
                "data_module": data_module,
                "train_dataset": train_dataset,
                "test_dataset": test_dataset,
                "accuracy": acc,
            }
        )

    results = pd.DataFrame(results).sort_values("accuracy", ascending=False)
    final_results = []
    datasets = sorted(results["test_dataset"].unique().tolist())

    if query is not None:
        results = results.query(query)

    if output_dir is None:
        output_dir = results_dir
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

    for model_name, model_df in results.groupby("model"):
        r = dict()
        r["model"] = model_name
        for dataset in datasets:
            r[dataset] = model_df[model_df["test_dataset"] == dataset]["accuracy"]
            if len(r[dataset]) == 0:
                r[dataset] = 0.0
            else:
                r[dataset] = r[dataset].iloc[0]

        r["mean"] = np.mean(
            [r[dataset] for dataset in datasets if r[dataset] != 0.0]
        )
        final_results.append(r)

    final_results = pd.DataFrame(final_results).sort_values("model")
    if print_results:
        print(final_results.to_markdown())

    final_results.to_csv(output_dir / result_file, index=False)
    print(f"Results saved to {output_dir / result_file}")


if __name__ == "__main__":
    CLI(analysis)
