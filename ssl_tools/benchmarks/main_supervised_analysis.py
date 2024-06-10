#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import numpy as np
from jsonargparse import CLI
import tqdm
import shutil


def analysis(
    results_dir: str,
    query: str = None,
    output_dir: str = None,
    result_file: str = "results.csv",
    print_results: bool = True,
    remove_on_error: bool = False,
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
        try:
            
            model, data_module, train_dataset, test_dataset = f.parents[
                1
            ].name.split("-")

            train_dataset = train_dataset.replace("train_on_", "")
            test_dataset = test_dataset.replace("test_on_", "")

            df = pd.read_csv(f)

            # Allow unfinished runs
            if "test_accuracy" not in df.columns:
                last_acc = 0.0
            else:
                last_acc = float(max(df["test_accuracy"].dropna()))

            test_acc_train_loss = np.max(np.array(
                [max(df[c].dropna()) for c in df.columns if "test_accuracy@train_loss" in c]
            ))
            val_acc_train_loss = np.max(np.array(
                [max(df[c].dropna()) for c in df.columns if "test_accuracy@val_loss" in c]
            ))

            results.append(
                {
                    "model": model,
                    "data_module": data_module,
                    "train_dataset": train_dataset,
                    "test_dataset": test_dataset,
                    "accuracy_last_epoch": last_acc,
                    "accuracy_best_train_loss": test_acc_train_loss,
                    "accuracy_best_val_loss": val_acc_train_loss,
                }
            )
        except Exception as e:
            print(f"Error parsing {f}: {e}")
            if remove_on_error:
                print(f"Removing {f.parents[1]}")
                shutil.rmtree(f.parents[1])
            continue

    results = pd.DataFrame(results).sort_values("accuracy_last_epoch", ascending=False)
    datasets = sorted(results["test_dataset"].unique().tolist())

    if query is not None:
        results = results.query(query)

    if output_dir is None:
        output_dir = results_dir
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

    for metric in ["accuracy_last_epoch", "accuracy_best_train_loss", "accuracy_best_val_loss"]:
        final_results = []
        for model_name, model_df in results.groupby("model"):
            r = dict()
            r["model"] = model_name
            for dataset in datasets:
                r[dataset] = model_df[model_df["test_dataset"] == dataset][metric]
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
            print(f"{'*' * 10} {metric} {'*' * 10}")
            print(final_results.to_markdown())

        output_file = output_dir / f"{metric}_{result_file}"
        final_results.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")
        print()


if __name__ == "__main__":
    CLI(analysis)
