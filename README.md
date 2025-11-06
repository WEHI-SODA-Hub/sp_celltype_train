# SODA-Classify-Train-Model Pipeline

This Nextflow pipeline is a sub-pipeline in the SODA-Classify suite for Spatial Proteomics. It is used to train an XGBoost model
on preprocessed output QuPath data or cell marker measurements in tabular format. This README contains WEHI-specific as well as general usage 
instructions.

## Introduction

* This pipeline uses BayesCV to tune the training of an XGBoost model. The parameters tuned are:
  * "eta": Real(1e-8, 1, "log-uniform"),
  * "reg_alpha": Real(1e-8, 1.0, "log-uniform"),
  * "reg_lambda": Real(1e-8, 1000, "log-uniform"),
  * "max_depth": Integer(0, 50, "uniform"),
  * "n_estimators": Integer(10, 300, "uniform"),
  * "learning_rate": Real(1e-8, 1.0, "log-uniform"),
  * "min_child_weight": Integer(0, 10, "uniform"),
  * "max_delta_step": Integer(1, 100, "uniform"),
  * "subsample": Real(1e-8, 1.0, "uniform"),
  * "colsample_bytree": Real(1e-8, 1.0, "uniform"),
  * "colsample_bylevel": (1e-8, 1.0, "uniform"),
  * "gamma": Real(1e-8, 1.0, "log-uniform"),
  * "min_child_weight": Integer(0, 5, "uniform")

* The preprocessing scheme (i.e., how the data is preprocessed before being fed into XGBoost). Can be:
	* `null`: no pre-processing - train the model on the data as-is.
	* `logp1`: apply apply the `ln(x+1)` transformation, where `x` is the input data.
	* `poly`: extract polynomial and interaction features to be trained on.
* The balance scheme. See [Imbalanced learn documentation](https://imbalanced-learn.org/stable/user_guide.html) for more information. Can be combination of: `ENN`, `TOMEK`, `ADASYN`, `SMOTEENN`, `SMOTE`, `SMOTETOMEK`, and/or `RUS`.
* The number of BayesCV iterations to test.

## Usage

Parameters:

```
  Usage:  nextflow run main.nf
  		--run_name <name>
  		--input <preprocessed.csv>
		--label_file <cell_type_labels.csv>
		--output_path <dir>
		--preprocess_scheme logp1,poly
		--balance_schemes TOMEK,ADASYN,SMOTEENN,SMOTE
		--bayescv_iterations 1,5,10,20
		--classifier Xgboost
		--options_toml <options.toml>
		--decoder <decoder.json>

  Required Arguments:
  --run_name			Run name used to label output files.
  --input				Preprocessed input data file from QuPath (.csv or .parquet).
  --label_file			File containing cell type labels (.csv or .parquet).
  --output_path			Path to directory to store output files.
  --preprocess_schemes	The schemes to use to transform the input data.
  --balance_schemes		The schemes to use to balance the input data.
  --bayescv_iterations	Numbers of parameter settings that are sampled in Bayes Search CV.
  --options_toml		TOML file containing preprocessing scheme and model classifier options.
  --decoder             JSON file containing the decoder for the predicted cell types.
  --classifier			Classifier to train. Can be XGBoost or XGBoost-gpu.

  Note: The pipeline automatically detects whether input files are CSV or Parquet format.
        Prediction output files (y_predicted, y_test, y_train) are always saved as CSV.
```

If you wish to use the GPU-accelerated version on WEHI's HPC, run the pipeline with the additional 
flag `-profile wehi_gpu`.

If you feel comfortable with the command line, you can run the preprocessing Python script directly.
However, the script does not run combinations of parameters in parallel, just a single combination.

```
$ conda env create -f envs/environment.yml   # or envs/environment-gpu.yml
$ conda activate xgboost-cell-classification # or xgboost-cell-classification-gpu
$ python scripts/train_classifier.py --help
usage: MIBI-train [-h] [--name NAME] [--input INPUT] [--labels LABELS]
                  [--output-path OUTPUT_PATH]
                  [--preprocess-scheme {null,logp1,poly}]
                  [--balance-scheme {ENN,TOMEK,ADASYN,SMOTEENN,SMOTE,SMOTETOMEK,RUS}]
                  [--bayescv-iterations BAYESCV_ITERATIONS]
                  [--options OPTIONS]
                  [--classifier {Xgboost,Xgboost-gpu}]
                  [--save-preprocessed] [--config CONFIG]

This script is for training the XGBoost classifier on labelled cell data.

optional arguments:
  -h, --help            show this help message and exit
  --name NAME, -n NAME  Run name used to label output files.
  --input INPUT, -i INPUT
                        Preprocessed input data file from QuPath (.csv or .parquet).
  --labels LABELS, -l LABELS
                        File containing cell type labels (.csv or .parquet).
  --output-path OUTPUT_PATH, -o OUTPUT_PATH
                        Path to directory to store output files.
  --preprocess-scheme {null,logp1,poly}, -s {null,logp1,poly}
                        The scheme to use to transform the input data.
  --balance-scheme {ENN,TOMEK,ADASYN,SMOTEENN,SMOTE,SMOTETOMEK,RUS}, -b {ENN,TOMEK,ADASYN,SMOTEENN,SMOTE,SMOTETOMEK,RUS}
                        The scheme to use to balance the input data.
  --bayescv-iterations BAYESCV_ITERATIONS, -t BAYESCV_ITERATIONS
                        Number of parameter settings that are sampled in
                        Bayes Search CV. Trades off runtime vs quality of
                        the solution.
  --options OPTIONS, -x OPTIONS
                        Path to TOML file containing preprocessing scheme
                        and model classifier options.
  --classifier {Xgboost,Xgboost-gpu}, -c {Xgboost,Xgboost-gpu}
                        Classifier to train.
  --save-preprocessed, -S
                        Saves the preprocessed data.
  --config CONFIG, -j CONFIG
                        Path to JSON config file.

Note: The script automatically detects input file format (.csv or .parquet) by extension.
      Prediction outputs are always saved as CSV for compatibility with reporting tools.
```

The script used to assess the results usage:

```
$ python scripts/create-qmd.py --help
usage: MIBI-assess-report-generator [-h] --decoder DECODER --output-dir OUTPUT_DIR --options-toml OPTIONS_TOML [--debug] INPUT_DIRECTORIES [INPUT_DIRECTORIES ...]

positional arguments:
  INPUT_DIRECTORIES     Directories containing MIBI training results.

optional arguments:
  -h, --help            show this help message and exit
  --decoder DECODER, -j DECODER
                        Path where the decoder JSON file is.
  --output-dir OUTPUT_DIR, -o OUTPUT_DIR
                        Directory to store scoring results and heatmaps.
  --options-toml OPTIONS_TOML, -x OPTIONS_TOML
                        TOML file containing training preprocess poly combinations to include in the report.
  --debug, -d           Turns on debug mode, which writes results to csv for validation.
```

### Example usage

After preprocessing the test data as per [the preprocessing example](https://github.com/BioimageAnalysisCoreWEHI/MIBI-preprocess-data/blob/main/README.md#example-usage),
To train/hyperparameter tune the model:

```
# Using CSV input (default preprocessing output)
nextflow run main.nf \
    --run_name test-train \
    --input /tmp/mibi-test-run-output/test_preprocessed_input_data.csv \
    --label_file /tmp/mibi-test-run-output/test_cell_type_labels.csv \
    --output_path /tmp/mibi-test-run-output/ \
    --preprocess_schemes null,poly \
    --balance_schemes TOMEK \
    --bayescv_iterations 1,3 \
    --options_toml options-example.toml \
    --decoder /tmp/mibi-test-run-output/test_decoder.json \
    -profile wehi_gpu # For running on WEHI's GPUs, which is much faster than CPUs.

# Using Parquet input (if preprocessing output was in Parquet format)
nextflow run main.nf \
    --run_name test-train-parquet \
    --input /tmp/mibi-test-run-output/test_preprocessed_input_data.parquet \
    --label_file /tmp/mibi-test-run-output/test_cell_type_labels.parquet \
    --output_path /tmp/mibi-test-run-output/ \
    --preprocess_schemes null,poly \
    --balance_schemes TOMEK \
    --bayescv_iterations 1,3 \
    --options_toml options-example.toml \
    --decoder /tmp/mibi-test-run-output/test_decoder.json \
    -profile wehi_gpu
```

This will reuse the output folder used in the preprocessing example.

**Note:** The pipeline automatically detects the input file format based on the file extension (`.csv` or `.parquet`). Parquet input files provide faster loading times for large datasets.

### Training options file

As part of the training, an options TOML must be supplied. This file exposes some of the more
detailed model training configuration options. An example of a working options TOML is in
`options-example.toml`. When training a model to predict cell type, only the
`[preprocess_options.combinations]` need to be modified.

#### Predicting functional markers

Ensure that your data has been preprocessed using the `MIBI-preprocess-data-FM` pipeline. When
supplying the options TOML for training, select `options-example-FM.toml` instead. This TOML file
differs by:

* changing `OBJECTIVE_FUNC` from `multi:softprob` to `binary:logistic`,
* changing `SCORING` from `balanced_accuracy` to `roc_auc`, and
* removing any `preprocess_options.combinations`.

## Pipeline Output

The pipeline will produce a collection of HTML reports which contain scoring and accuracy
information for the combinations of hyperparameters considered. Use the reports to choose which
model to use moving forward. The reports will be copied to where you specify as `--output_path`. If
using the Python script directly, the report will be printed to the terminal in markdown format.

The models themselves are stored as pickle files and the directory containing the models will be
recorded in the report. The model will be saved as `bayes_cv_model.sav` e.g., 
`<process directory>/<name>-logp1-ADASYN-1/bayes_cv_model.sav`.

## Credits 

The core functionality of the MIBI pipeline was developed by Kenta Yotoke (@yokotenka) under the supervision of Claire Marceaux 
(@ClaireMarceaux). The pipeline was adapted to Nextflow by Edward Yang (@edoyango) and maintained by Michael Mckay (@mikemcka) and Michael Milton (@multimeric).

## Citation

TBC
