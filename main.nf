#!/usr/bin/env nextflow

/// To use DSL-2
nextflow.enable.dsl=2

// Import subworkflows to be run in the workflow
include { TRAIN } from './modules/train'
include { REPORT } from './modules/report'

/// Print a header 
log.info """\

=======================================================================================
MIBI Training Pipeline - nf 
=======================================================================================

Created by Clair Marceux, WEHI

Find documentation and more info @ https://github.com/BioimageAnalysisCoreWEHI/MIBI-train-model/

Cite this pipeline @ INSERT DOI

Log issues @ https://github.com/BioimageAnalysisCoreWEHI/MIBI-train-model/

=======================================================================================
Workflow run parameters 
=======================================================================================
run_name          : ${params.run_name}
input	          : ${params.input}
label_file        : ${params.label_file}
output_path       : ${params.output_path}
preprocess_schemes: ${params.preprocess_schemes}
balance_schemes   : ${params.balance_schemes}
bayescv_iterations: ${params.bayescv_iterations}
options_toml      : ${params.options_toml}
classifier        : ${params.classifier}
decoder           : ${params.decoder}
workDir           : ${workflow.workDir}
=======================================================================================

"""

/// Help function 
def helpMessage() {
    log.info"""
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

  Required Arguments:
  --run_name			Run name used to label output files.

  --input				Preprocessed input data file from QuPath.
  --label_file			File containing cell type labels.
  --output_path			Path to directory to store output files.

  --preprocess_schemes	The schemes to use to transform the input data.
  --balance_schemes		The schemes to use to balance the input data.

  --bayescv_iterations	Numbers of parameter settings that are sampled in Bayes Search CV.

  --options_toml		TOML file containing preprocessing scheme and model classifier options.
  --classifier			Classifier to train.

""".stripIndent()
}

workflow {

	// Show help message if --help is run or if any required params are not provided at runtime
	if ( params.help ||
	     params.input == "" ||
		 params.run_name == "" ||
		 params.label_file == "" ||
		 params.output_path == "" ||
		 params.preprocess_schemes == "" ||
		 params.balance_schemes == "" ||
		 params.bayescv_iterations == "" ||
		 params.options_toml == "" ||
		 params.decoder == "" ||
		 params.classifier == ""){
   
		// Invoke the help function above and exit
		helpMessage()
		exit 1

	// if none of the above are a problem, then run the workflow
	} else {
		
		// Define input channels 
		input = Channel.fromPath(file("${params.input}"), checkIfExists: true)
		label_file = Channel.fromPath(file("${params.label_file}"), checkIfExists: true)
		options_toml = Channel.fromPath(file("${params.options_toml}"), checkIfExists: true)
		preprocess_schemes = Channel.from(params.preprocess_schemes.split(","))
		balance_schemes = Channel.from(params.balance_schemes.split(","))
		bayescv_iterations = Channel.from(params.bayescv_iterations.split(","))

		// Run training process
		results_ch = TRAIN(input, label_file, options_toml, preprocess_schemes, balance_schemes, bayescv_iterations)

		decoder = Channel.fromPath(file("${params.decoder}"), checkIfExists: true)

		report_ch = REPORT(results_ch.collect(), decoder, options_toml)

	}}

	workflow.onComplete {
	summary = """
=======================================================================================
Workflow execution summary
=======================================================================================

Duration    : ${workflow.duration}
Success     : ${workflow.success}
workDir     : ${workflow.workDir}
Exit status : ${workflow.exitStatus}
output_path : ${params.output_path}

=======================================================================================
	"""
	println summary

}
