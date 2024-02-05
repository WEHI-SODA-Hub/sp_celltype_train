"""
Author: YOKOTE Kenta
Aim: To run the XGBoost classifier on labelled cell data on SLURM
    Takes 8 inputs from STDIN:
        1. run_name: name of the run. The outputs will be saved to a folder
                     which has this as the name
        2. input_folder: 
        3. input_file: name of the file located in input_folder
        4. labels_file: name of the file containing labels 
        5. output_folder: 
        6. classifier_scheme: the type of classifier to use
        7. model_options: 
"""

import sys
import json
import pandas as pd
import pickle
from classifier_initilaliser import ClassifierInitialiser
from preprocess.data_transformer import DataTransformer
import os


def train(
    run_name: str,
    input_file: str,
    labels_file: str,
    output_folder: str,
    preprocess_scheme: str,
    preprocess_options: str,
    model_options: str,
    classifier_scheme: str,
    save_preprocessed_data: str,
):
    # Make output directory if not already made
    full_output_directory = os.path.join(output_folder, run_name)
    if not os.path.exists(full_output_directory):
        os.makedirs(full_output_directory)

    # Read the data
    print("INFO: Loading data")
    print("INFO: Measurement file: {}".format(input_file))
    X = pd.read_csv(input_file)
    y = pd.read_csv(labels_file)

    # Preprocess
    print("INFO: Preprocessing")
    data_transformer = DataTransformer()
    X = data_transformer.transform_data(
        X, transform_scheme=preprocess_scheme, args=preprocess_options
    )

    if save_preprocessed_data:
        X.to_csv(
            os.path.join(full_output_directory, "preprocessed_data.csv"), index=False
        )

    # Tune the hyperparameters
    print("INFO: Tuning hyperparameters")
    classifier_applier = ClassifierInitialiser()
    classifier_applier.tune_hyper_parameter(X, y, classifier_scheme, model_options)

    # get predictions
    print("INFO: Save the predictions")
    classifier_applier.get_prediction_df().to_csv(
        os.path.join(full_output_directory, "y_predicted.csv"), index=False
    )
    classifier_applier.y_train_orig.to_csv(
        os.path.join(full_output_directory, "y_train.csv"), index=False
    )
    classifier_applier.y_test.to_csv(
        os.path.join(full_output_directory, "y_test.csv"), index=False
    )

    # Save the outputs
    print("INFO: Save the cross validation results")
    filename_bayes = os.path.join(full_output_directory, "bayes_cv_model.sav")
    pickle.dump(classifier_applier.bayes_cv_tuner, open(filename_bayes, "wb"))

    # Sav ethe final best model
    print("INFO: Save final model")
    model = classifier_applier.get_final_classifier()
    filename = os.path.join(full_output_directory, "final_model.sav")
    pickle.dump(model, open(filename, "wb"))
    print("INFO: Finished")


def check_arg(arg, long_flag, short_flag):
    """Function to be called after arguments are parsed. Checks whether the right flags are passed
    when --config/-j haven't been."""

    if arg:
        return arg
    else:
        print(f"{long_flag}/{short_flag} must be supplied if not using --config/-j.")
        sys.exit(1)


if __name__ == "__main__":
    import argparse, toml

    parser = argparse.ArgumentParser(
        prog="MIBI-train",
        description="""This script is for training the XGBoost classifier on labelled cell data.""",
    )

    parser.add_argument("--name", "-n", help="Run name used to label output files.")
    parser.add_argument(
        "--input", "-i", help="Preprocessed input data file from QuPath."
    )
    parser.add_argument("--labels", "-l", help="File containing cell type labels.")
    parser.add_argument(
        "--output-path", "-o", help="Path to directory to store output files."
    )
    parser.add_argument(
        "--preprocess-scheme",
        "-s",
        help="The scheme to use to transform the input data.",
        choices=["logp1", "poly"],
    )
    parser.add_argument(
        "--balance-scheme",
        "-b",
        help="The scheme to use to balance the input data.",
        choices=["ENN", "TOMEK", "ADASYN", "SMOTEENN", "SMOTE"],
    )
    parser.add_argument(
        "--bayescv-iterations",
        "-t",
        help="Number of parameter settings that are sampled in Bayes Search CV. Trades off runtime vs quality of the solution.",
        type=int,
    )
    parser.add_argument(
        "--options",
        "-x",
        help="Path to TOML file containing preprocessing scheme and model classifier options.",
    )
    parser.add_argument(
        "--classifier", "-c", help="Classifier to train.", choices=["Xgboost", "Xgboost-gpu"]
    )
    parser.add_argument(
        "--save-preprocessed",
        "-S",
        action="store_true",
        help="Saves the preprocessed data.",
    )
    parser.add_argument("--config", "-j", help="Path to JSON config file.")

    args = parser.parse_args()

    # Run options
    run_options_file = args.config

    if run_options_file:
        try:
            with open(run_options_file) as json_file:
                run_options = json.load(json_file)
        except FileNotFoundError:
            print(f"File {run_options_file} could not be loaded.", file=sys.stderr)
            sys.exit(2)
        except:
            print(
                f"There was a problem with loading the JSON config file, {run_options_file}",
                file=sys.stderr,
            )
            sys.exit(1)

        # Run name
        run_name = run_options["RUN_NAME"]
        # Get the input variables
        input_file = run_options["INPUT_FILE"]
        # Get the labels file
        labels_file = run_options["LABELS_FILE"]
        # Get the output folder
        output_folder = run_options["OUTPUT_FOLDER"]
        # Preprocess
        preprocess_scheme = run_options["PREPROCESS_SCHEME"]
        preprocess_options = run_options["PREPROCESS_OPTIONS"]
        # model options
        model_options = run_options["MODEL_OPTIONS"]
        # classifier
        classifier_scheme = run_options["CLASSIFIER"]
        # Save preprocessed data
        try:
            save_preprocessed_data = run_options["SAVE_PREPROCESSED"]
        except KeyError:
            save_preprocessed_data = False

    else:
        run_name = check_arg(args.name, "--name", "-n")
        input_file = check_arg(args.input, "--input", "-i")
        labels_file = check_arg(args.labels, "--labels", "-l")
        output_folder = check_arg(args.output_path, "--output-path", "-o")
        preprocess_scheme = check_arg(
            args.preprocess_scheme, "--preprocess-scheme", "-s"
        )
        balance_scheme = check_arg(args.balance_scheme, "--balance-scheme", "-b")
        bayescv_iterations = check_arg(
            args.bayescv_iterations, "--bayescv-iterations", "-t"
        )
        preprocess_model_options = check_arg(args.options, "--options", "-x")
        classifier_scheme = check_arg(args.classifier, "--classifier", "-c")
        save_preprocessed_data = args.save_preprocessed

        # load options toml
        try:
            options_toml = toml.load(preprocess_model_options)
        except FileNotFoundError:
            print(f"Options TOML file not found at {preprocess_model_options}")
            sys.exit(2)

        preprocess_options = options_toml["preprocess_options"]
        model_options = options_toml["model_options"]

        # if balance_scheme passed by command line, add back into model_options dict
        model_options["BALANCE_SCHEME"] = balance_scheme
        model_options["BAYESCV_OPTIONS"]["ITERATIONS"] = bayescv_iterations

    train(
        run_name,
        input_file,
        labels_file,
        output_folder,
        preprocess_scheme,
        preprocess_options,
        model_options,
        classifier_scheme,
        save_preprocessed_data,
    )
