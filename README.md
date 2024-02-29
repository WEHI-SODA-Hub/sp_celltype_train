# MIBI Train Model Pipeline

This Nextflow pipeline is a sub-pipeline in the MIBI suite. It is used to train an XGBoost model
on preprocessed QuPath data. This README contains WEHI-specific as well as general usage 
instructions.

## Introduction

This pipeline uses BayesCV to tune the training of an XGBoost model. The parameters tuned are:

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
  --input				Preprocessed input data file from QuPath.
  --label_file			File containing cell type labels.
  --output_path			Path to directory to store output files.
  --preprocess_schemes	The schemes to use to transform the input data.
  --balance_schemes		The schemes to use to balance the input data.
  --bayescv_iterations	Numbers of parameter settings that are sampled in Bayes Search CV.
  --options_toml		TOML file containing preprocessing scheme and model classifier options.
  --decoder             JSON file containing the decoder for the predicted cell types.
  --classifier			Classifier to train. Can be XGBoost or XGBoost-gpu.
```

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
                        Preprocessed input data file from QuPath.
  --labels LABELS, -l LABELS
                        File containing cell type labels.
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

## Pipeline Output

The pipeline will produce a collection of HTML reports which contain scoring and accuracy
information for the combinations of hyperparameters considered. Use the reports to choose which
model to use moving forward. The reports will be copied to where you specify as `--output_path`. If
using the Python script directly, the report will be printed to the terminal in markdown format.

The models themselves are stored as pickle files and the directory containing the models will be
recorded in the report. The model will be saved as `final_model.sav` e.g., 
`<process directory>/<name>-logp1-ADASYN-1/final_model.sav`.

## Credits 

The core functionality of the MIBI pipeline was developed by Kenta Yotoke (@yokotenka) under the supervision of Claire Marceaux 
(@ClaireMarceaux). The pipeline was adapted to Nextflow by Edward Yang (@edoyango).

## Citation

TBC