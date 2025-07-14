#!/usr/bin/env python3

import pandas as pd, numpy as np
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

import os, sys, pathlib
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
    #cm_norm = cm / cm.astype(float).sum(axis=1)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)


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
        classification_report_df,
        overall_score_df,
    )

PER_CELL_STATS_COLUMNS = ("precision", "recall")
OVERALL_STATS_COLUMNS = ("**accuracy**", "**macro avg**", "**weighted avg**")

def _process_report_dfs(fpath, df_dict):

    mask = ~df_dict["classification"].index.isin(OVERALL_STATS_COLUMNS)
    cell_stats_df = df_dict["classification"].T.loc[
        PER_CELL_STATS_COLUMNS, 
        mask
    ]

    s = cell_stats_df.unstack()

    fname_segmented = pathlib.Path(fpath).stem.split("-")
    bcv_iters = fname_segmented[-1]
    bscheme = fname_segmented[-2]
    ppscheme = fname_segmented[-3]
    s.name = " ".join((ppscheme, bscheme, bcv_iters))
    s.index.names = ("Cell type", "Statistic")

    for stat in df_dict["overall"].index:
        s.loc[("overall", stat)] = df_dict["overall"].loc[stat, 0]

    return s
def _highlight_max(s, props=""):
    return np.where(s==np.nanmax(s.values), props, None)

class mibi_train_reporter:

    def __init__(self, combinations, decoder_path, output_dir=".", debug=False):

        self.decoder_path = decoder_path
        self.output_path = output_dir
        self.debug = debug

        self.sections = []

        self.scores_dfs = {}

        self.aggregated_stats_header = f"""---
title: MIBI Assess Predictions Statistics Report"
author: {getpass.getuser()}
date: now
format:
  html:
    page-layout: full
    embed-resources: true
---
"""

        self.header = f"""---
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

## Poly Preprocess combinations

These are the combinations you've provided to be used with the `poly` preprocess scheme.

{tabulate.tabulate(combinations, combinations.keys(), tablefmt="github")}"""
        
        self.section_body = """## {label}

**Input/output Folder:**

```
{input_path}
```

**Decoder:**

```
{decoder_path}
```

![Confusion matrices for {label}]({img_path}){{fig-alt="Confusion matrices for {label} data. Left: absolute, right: normalised."}}

::: {{#tbl-panel layout-ncol=2}}
{classification_report_table}

: Per cell type scores {{#tbl-first}}

{overall_scores_table}

: Overall scores {{#tbl-second}}

Scoring tables for {label}
:::
"""

    def add_section(self, input_path):

        label = os.path.basename(input_path.rstrip("/"))

        img_path, classification_report_df, overall_scores_df = process_input(
            label, input_path, self.decoder_path, self.output_path, self.debug
        )

        self.scores_dfs[input_path] = {
            "classification": classification_report_df,
            "overall": overall_scores_df
        }

        mapping = {
            "label": label,
            "input_path": os.path.realpath(input_path),
            "decoder_path": os.path.realpath(self.decoder_path),
            "img_path": img_path,
            "classification_report_table": classification_report_df.to_markdown(),
            "overall_scores_table": overall_scores_df.to_markdown()
        }

        self.sections.append(self.section_body.format_map(mapping))

    def _aggregate_classification_scores(self):

        cell_stats_df = pd.concat(
            [_process_report_dfs(fpath, df_dict) for fpath, df_dict in self.scores_dfs.items()], 
            axis=1
        )

        style = cell_stats_df.style.set_table_styles([
            {'selector': 'th', 'props': [
                ('background-color', '#f0f0f0'),  # Light gray header background
                ('border', '1px solid #ddd'),    # Light gray borders
                ('font-family', 'sans-serif'),
                ('font-weight', 'bold')
            ]},
            {'selector': 'td', 'props': [
                ('border', '1px solid #ddd'),    # Light gray borders
                ('font-family', 'sans-serif')
            ]}
        ]).set_table_attributes('style="border-collapse: collapse"')  # Ensure table collapse

        print(
            "# Aggregated statistics", 
            style.format(precision=4).to_html(),
            "Aggregated overall and per cell-type prediction statistics.",
            sep="\n\n"
        )

    def print_report(self):

        print(self.header, sep="\n\n")

        for s in self.sections:
            print(s, sep="\n\n")

        self._aggregate_classification_scores()

    def print_aggregated_statistics_report(self):

        print(self.aggregated_stats_header, sep="\n\n")

        self._aggregate_classification_scores()


def main(
    decoder_path: str,
    options_toml: str,
    input_dirs: list,
    output_path: str,
    debug: bool,
    only_aggregated_stats: bool,
) -> None:
    """Driver function

    Args:
        input_path: path to the directory containing training outputs
        options_toml: path to toml with preprocess options. used to print the poly combinations.
        decoder_path: path to the JSON decoder file
        output_path: path to the directory to store generated images"""

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

    reporter = mibi_train_reporter(combinations, decoder_path, output_path, debug)

    for d in input_dirs:
        reporter.add_section(d)

    if only_aggregated_stats:
        reporter.print_aggregated_statistics_report()
    else:
        reporter.print_report()


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
    parser.add_argument(
        "--only-aggregated-stats",
        "-a",
        action="store_true",
        help="Only prints the aggregate prediction statistics table (as well as relevant quarto headers)."
    )

    args = parser.parse_args()

    main(
        args.decoder,
        args.options_toml,
        args.INPUT_DIRECTORIES,
        args.output_dir,
        args.debug,
        args.only_aggregated_stats
    )
