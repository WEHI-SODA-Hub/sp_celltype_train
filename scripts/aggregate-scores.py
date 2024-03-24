#!/usr/bin/env python3

import pandas as pd, numpy as np
import glob, os, getpass

CELL_STATS = ("precision", "recall")
CELL_OVERALL_STATS = ("accuracy", "macro avg", "weighted avg")

def process_classification_report(fpath):

    df = pd.read_csv(fpath, index_col=0).T.loc[CELL_STATS, :]

    mask = ~df.columns.isin(CELL_OVERALL_STATS)
    s = df.loc[CELL_STATS, mask].unstack()
    
    sections = fpath.split("-")
    bcv_iters = sections[-1].split("_")[0]
    bscheme = sections[-2]
    ppscheme = sections[-3]
    s.name = " ".join((ppscheme, bscheme, bcv_iters))

    return s

def highlight_max(s, props=""):
    return np.where(s==np.nanmax(s.values), props, None)
    
def print_aggregated_classification(csvs):

    aggregated_df = pd.concat(
        [process_classification_report(f) for f in csvs],
        axis=1
    )
    
    s = aggregated_df.style.apply(highlight_max, props="color:green", axis=1)
    print("""---
title: MIBI training aggregated scores
format:
  html:
    number-sections: true
    embed-resources: true
    theme: cosmo
    page-layout: full
    toc: true
    toc-location: left
    fontsize: "8"
---\n\n""", s.format(precision=4).to_html())

def print_aggregated_overrall(csvs):

    print("overall: ", csvs)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        "MIBI-aggregate-scores", 
        description="CLI utility to group produce a summary qmd report")
    
    subparsers = parser.add_subparsers(
        title="score/report type", 
        description="Type of scores/report to aggregate"
    )

    cparser = subparsers.add_parser("classification-report")
    cparser.add_argument("CSVs", nargs='+')
    cparser.set_defaults(func=print_aggregated_classification)

    oparser = subparsers.add_parser("overall-scores")
    oparser.add_argument("CSVs", nargs='+')
    oparser.set_defaults(func=print_aggregated_overrall)

    args = parser.parse_args()

    args.func(args.CSVs)

    # process_csvs(args.input_dir)