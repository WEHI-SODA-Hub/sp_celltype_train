#!/usr/bin/env python3

import pandas as pd
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    cohen_kappa_score,
    matthews_corrcoef,
    log_loss,
    classification_report,
    confusion_matrix,
)

import os, sys
import json
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns
import getpass
import toml, json
import tabulate


def prepare_plot(y_test: pd.DataFrame, y_pred: pd.DataFrame, decoder: dict) -> Figure:
    # create figure and subplots
    fig, axs = plt.subplots(1, 2, figsize=(20, 10), dpi=75, sharey=True)

    # create raw confusion matrix
    phenotypes = y_test.iloc[:, 0].unique()
    phenotypes_string = [decoder.get(str(x)) for x in phenotypes]
    cm = confusion_matrix(y_test.iloc[:, 0], y_pred.iloc[:, 0])

    # plot raw confusion matrix
    plt.sca(axs[0])
    sns.heatmap(cm, annot=True, fmt="g")
    plt.xticks(phenotypes + 0.5, phenotypes_string, rotation=45)
    plt.yticks(phenotypes + 0.5, phenotypes_string, rotation=45)
    plt.ylabel("True phenotypes", fontweight="bold", fontsize=16)
    plt.xlabel("Predicted phenotypes", fontweight="bold", fontsize=16)
    plt.title(
        "Confusion matrix for predicted phenotypes (absolute)",
        fontweight="bold",
        fontsize=20,
    )

    # create normalized confusion matrix
    cm_norm = cm / cm.astype(float).sum(axis=1)

    # plot normalized confusion matrix
    plt.sca(axs[1])
    sns.heatmap(cm_norm, annot=True, fmt=".3f", vmin=0.0, vmax=1.0)
    plt.xticks(phenotypes + 0.5, phenotypes_string, rotation=45)
    plt.yticks(phenotypes + 0.5, phenotypes_string, rotation=45)
    # plt.ylabel("True phenotypes", fontweight="bold")
    plt.xlabel("Predicted phenotypes", fontweight="bold", fontsize=16)
    plt.title(
        "Confusion matrix for predicted phenotypes (normalised)",
        fontweight="bold",
        fontsize=20,
    )

    fig.tight_layout()

    return fig


def prepare_classification_report(
    y_test: pd.DataFrame, y_pred: pd.DataFrame, decoder: dict, debug: bool
) -> pd.DataFrame:
    classification_report_ = classification_report(
        y_test, y_pred.iloc[:, 0], digits=5, output_dict=True
    )
    classification_report_df = pd.DataFrame(classification_report_).rename(
        columns=decoder
    )

    if not debug:
        classification_report_df = classification_report_df.rename(columns=decoder)
        summary_bold = {
            "accuracy": "**accuracy**",
            "macro avg": "**macro avg**",
            "weighted avg": "**weighted avg**",
        }
        classification_report_df = classification_report_df.rename(columns=summary_bold)

    return classification_report_df.T


def calculate_overall_scores(
    y_test: pd.DataFrame, y_pred: pd.DataFrame, debug: bool
) -> pd.DataFrame:
    """Calculates various scores based on predicted vs test values

    Args:
        y_test: A dataframe with the values to test against
        y_pred: A dataframe with the predicted values to test
        debug: True if in debug

    Returns:
        A markdown table string with the calculated scores
    """
    # overall accuracy
    accuracy = accuracy_score(y_test, y_pred.iloc[:, 0])

    # balanced accuracy
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred.iloc[:, 0])

    # f1 score macro averaged
    f1_macro = f1_score(y_test, y_pred.iloc[:, 0], average="macro")

    # precision score macro averaged
    precision_macro = precision_score(y_test, y_pred.iloc[:, 0], average="macro")

    # recall score macro averaged
    recall_macro = recall_score(y_test, y_pred.iloc[:, 0], average="macro")

    # cohen kappa score
    cohen_kappa = cohen_kappa_score(y_test, y_pred.iloc[:, 0])

    # matthews correlation
    matthews_correlation = matthews_corrcoef(y_test, y_pred.iloc[:, 0])

    # log loss or cross entropy
    log_loss_ = log_loss(
        y_test, y_pred.iloc[:, 1 : (len(y_test.iloc[:, 0].unique()) + 1)]
    )

    if debug:
        return pd.DataFrame(
            {
                "accuracy": accuracy,
                "balanced_accuracy": balanced_accuracy,
                "f1_macro": f1_macro,
                "precision_macro": precision_macro,
                "recall_macro": recall_macro,
                "cohen_kappa": cohen_kappa,
                "matthews_correlation": matthews_correlation,
                "log_loss": log_loss_,
            },
            index=[0],
        ).T
    else:
        return pd.DataFrame(
            {
                "accuracy": accuracy,
                "balanced_accuracy": balanced_accuracy,
                "f1_macro": f1_macro,
                "precision_macro": precision_macro,
                "recall_macro": recall_macro,
                "cohen_kappa": cohen_kappa,
                "matthews_correlation": matthews_correlation,
                "log_loss": log_loss_,
            },
            index=["Score"],
        ).T


def process_input(
    label: str, input_path: str, decoder_path: str, output_path: str, debug: bool
) -> tuple:
    """Produces a markdown report for evaluating MIBI training schemes

    Args:
        label: A label used to distinguish data inputs
        input_path: The path at which the outputs from training is located
        decoder_path: The path where the JSON decoder is located
        output_path: The output directory which output images are stored

    Returns:
        A tuple consisting of the path to the output image, the markdown table with
        the classification table, and the markdown table containing accuracy scores.
    """
    # reading data
    y_test = pd.read_csv(os.path.join(input_path, "y_test.csv"))
    y_pred = pd.read_csv(os.path.join(input_path, "y_predicted.csv"))

    with open(decoder_path) as json_file:
        decoder = json.load(json_file)

    fig = prepare_plot(y_test, y_pred, decoder)

    output_image_path = os.path.join(
        output_path, f"cell_type_confusion_matrices_{label}.png"
    )

    fig.savefig(output_image_path)

    classification_report_df = prepare_classification_report(
        y_test, y_pred, decoder, debug
    )

    overall_score_df = calculate_overall_scores(y_test, y_pred, debug)

    if debug:
        classification_report_df.to_csv(
            os.path.join(output_path, f"{label}_classification_report.csv")
        )
        overall_score_df.to_csv(os.path.join(output_path, f"{label}_overall_score.csv"))

    return (
        output_image_path,
        classification_report_df.to_markdown(),
        overall_score_df.to_markdown(),
    )


def print_header() -> None:
    """prints the header for the final markdown file"""
    print(
        f"""---
title: MIBI Assess Predictions Report
author: {getpass.getuser()}
date: now
format:
  html:
    toc: true
    toc-location: left
    code-fold: true
    page-layout: full
    embed-resources: true
---

"""
    )


def print_section(
    input_path: str, decoder_path: str, output_path: str, debug: bool
) -> None:
    label = os.path.basename(input_path.rstrip("/"))
    """prints markdown section related to the supplied results
    
    Args:
        input_path: path to the directory containing training outputs
        decoder_path: path to the JSON decoder file
        output_path: path to the directory to store generated images"""

    print(
        f"""## {label}

**Input/output Folder:**

```
{os.path.realpath(input_path)}
```

**Decoder:**

```
{os.path.realpath(decoder_path)}
```
"""
    )

    img_path, classification_report_txt, overall_scores_txt = process_input(
        label, input_path, decoder_path, output_path, debug
    )

    print(
        f"""
![Confusion matrices for {label}]({img_path}){{fig-alt="Confusion matrices for {label} data. Left: absolute, right: normalised."}}
"""
    )

    print("::: {#tbl-panel layout-ncol=2}")
    print(classification_report_txt + "\n\n: Per cell type scores {{#tbl-first}}")
    print()
    print(overall_scores_txt + "\n\n: Overall scores {{#tbl-second}}")
    print()
    print(f"Scoring tables for {label}\n:::")


def main(
    decoder_path: str,
    options_toml: str,
    input_dirs: list,
    output_path: str,
    debug: bool,
) -> None:
    """Driver function

    Args:
        input_path: path to the directory containing training outputs
        options_toml: path to toml with preprocess options. used to print the poly combinations.
        decoder_path: path to the JSON decoder file
        output_path: path to the directory to store generated images"""

    print_header()

    try:
        options = toml.load(options_toml)
    except FileNotFoundError:
        print(f"Options TOML file not found at {options_toml}", file=sys.stderr)
        sys.exit(2)

    try:
        combinations = options["preprocess_options"]["combinations"]
    except KeyError:
        print(
            "Looks like preprocess_options.combionations isn't present!",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"""
## Poly Preprocess combinations

These are the combinations you've provided to be used with the `poly` preprocess scheme.

{tabulate.tabulate(combinations, combinations.keys(), tablefmt="github")}

"""
    )

    for d in input_dirs:
        print_section(d, decoder_path, output_path, debug)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        "MIBI-assess-report-generator",
    )

    parser.add_argument(
        "--decoder", "-j", help="Path where the decoder JSON file is.", required=True
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        help="Directory to store scoring results and heatmaps.",
        required=True,
    )
    parser.add_argument(
        "--options-toml",
        "-x",
        help="TOML file containing training preprocess poly combinations to include in the report.",
        required=True,
    )
    parser.add_argument(
        "INPUT_DIRECTORIES",
        nargs="+",
        help="Directories containing MIBI training results.",
    )
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Turns on debug mode, which writes results to csv for validation.",
    )

    args = parser.parse_args()

    main(
        args.decoder,
        args.options_toml,
        args.INPUT_DIRECTORIES,
        args.output_dir,
        args.debug,
    )
