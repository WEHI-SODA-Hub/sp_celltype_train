#!/bin/env nextflow 

// Enable DSL-2 syntax
nextflow.enable.dsl=2

process TRAIN {	

	memory "100 GB"
	cpus 56
	conda "${projectDir}/envs/environment.yml"
	time "${params.walltime}"
	label "train"

	input:
	path script
	path input
	path label_file
	path options_toml
	each preprocess_scheme
	each balance_scheme
	each bayescv_iteration

	output:
	path("${params.run_name}-${preprocess_scheme}-${balance_scheme}-${bayescv_iteration}")
	
	script:
	"""
	python3 -B -u ${script} \\
		--name ${params.run_name}-${preprocess_scheme}-${balance_scheme}-${bayescv_iteration} \\
		--input ${input} \\
		--labels ${label_file} \\
		--output-path . \\
		--preprocess-scheme ${preprocess_scheme} \\
		--balance-scheme ${balance_scheme} \\
		--bayescv-iterations ${bayescv_iteration} \\
		--options ${options_toml} \\
		--classifier ${params.classifier}
	""".stripIndent()
}
