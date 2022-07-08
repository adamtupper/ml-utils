"""Download all files and metrics from a Weights & Biases project.

Usage:
    wandb-dl -e ENTITY -p PROJECT -o OUTPUT_DIRECTORY
"""
import argparse
import json
import os
import sys

import pandas as pd
import wandb

NUM_SAMPLES = (
    2147483647  # Maximum number of samples for system metrics, default is the maximum
)


def parse_args(args):
    """Parse command line parameters.

    :param args: command line parameters as list of strings (for example
        ``["--help"]``).
    :type args: list

    :return: command line parameters namespace.
    """
    parser = argparse.ArgumentParser(description="Download a Weights & Biases project.")

    parser.add_argument(
        "--entity",
        "-e",
        default=False,
        help="name of the entity (user)",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--project", "-p", help="the name of the project.", type=str, required=True
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        help="the directory to export the data to",
        type=str,
        required=True,
    )

    return parser.parse_args(args)


def main(args):
    """Download all files and metrics from the Weights & Biases project.

    :param args: command line parameters as list of strings (for example
        ``["--help"]``).
    :type args: list
    """
    args = parse_args(args)
    api = wandb.Api()
    runs = api.runs(os.path.join(args.entity, args.project))

    for run in runs:
        root_dir = os.path.join(args.output_dir, run.id)

        if not os.path.exists(root_dir):
            os.makedirs(root_dir)

        # Download run metrics
        history = run.scan_history()
        metrics_df = pd.DataFrame(history)
        metrics_df.to_csv(os.path.join(root_dir, "metrics.csv"))

        # Download system metrics
        history = run.history(samples=NUM_SAMPLES, stream="events")
        system_metrics_df = pd.DataFrame(history)
        system_metrics_df.to_csv(os.path.join(root_dir, "system_metrics.csv"))

        # Download run configs
        # .config contains the hyperparameters, we remove special values that start with _
        config = {k: v for k, v in run.config.items() if not k.startswith("_")}
        config_df = pd.DataFrame([config])
        config_df.to_csv(os.path.join(root_dir, "config.csv"))

        # Download run summaries
        summary_df = pd.DataFrame([run.summary._json_dict])
        summary_df.to_csv(os.path.join(root_dir, "summary.csv"))

        # Download artifacts
        for artifact in run.logged_artifacts():
            artifact.download(root=os.path.join(root_dir, "artifacts", artifact.name))
            metadata_path = os.path.join(
                root_dir, "artifacts", artifact.name, "metadata.json"
            )
            with open(metadata_path, "w") as f:
                json.dump(artifact.metadata, f)

        # Download files
        for file in run.files():
            file.download(root=root_dir, replace=True)


def run():
    """Calls :func:`main` passing the CLI arguments extracted from :obj:`sys.argv`
    This function can be used as entry point to create console scripts.
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
