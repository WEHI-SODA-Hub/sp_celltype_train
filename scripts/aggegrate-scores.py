#!/usr/bin/env python3

import pandas as pd
import glob, os, getpass

def transform_csv(fname):

    df = pd.read_csv(fname, index_col=0).T

    # fname will always be <prefix>-<preprocessing scheme>-<balance scheme>-<BayesCV iterations>_classification_report.csv (or _overall_score.csv)
    sections = fname.split("-")
    df.insert(0, "BayesCV iterations", sections[-1].split("_")[0])
    df.insert(0, "Balance scheme", sections[-2])
    df.insert(0, "Preprocessing scheme", sections[-3])

    df.columns

    return df

def process_csvs(input_dir):

    overall_score_files = glob.iglob(os.path.join(input_dir, '*_overall_score.csv'))
    classification_report_files = glob.iglob(os.path.join(input_dir, '*_overall_score.csv'))

    overall_score_df = pd.concat(
        [transform_csv(f) for f in overall_score_files], 
        axis=0
    )
    classification_report_df = pd.concat(
        [transform_csv(f) for f in classification_report_files],
        axis=0
    )

    print(f"""---
title: MIBI aggregated scores
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

# Classification Scores

{classification_report_df.to_markdown(index=False, tablefmt="simple")}

# Overall scores

{overall_score_df.to_markdown(index=False, tablefmt="simple")}

""")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        "MIBI-aggregate-scores", 
        description="CLI utility to group produce a summary qmd report")

    parser.add_argument(
        "input_dir",
        help="Directory containing the collection of *_classification_report.csv and *_overall_score.csv CSV files",
        default="."
    )

    args = parser.parse_args()

    process_csvs(args.input_dir)