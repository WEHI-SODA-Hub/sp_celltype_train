#!/bin/env nextflow 

// Enable DSL-2 syntax
nextflow.enable.dsl=2

// Define the process
process processOne {	
	cpus "${params.cpus}"
	debug = true //turn to false to stop printing command stdout to screen
	memory "1 GB"

	// See: https://www.nextflow.io/docs/latest/process.html#inputs
	// each input needs to be placed on a new line
	input:
	path cohort_ch
	each preprocess_scheme
	each balance_scheme
	each bayescv_iteration

	// See: https://www.nextflow.io/docs/latest/process.html#outputs
	// each new output needs to be placed on a new line
	output:
	stdout
	
	// this is an example of some code to run in the code block 
	script:
	"""
	echo ${preprocess_scheme} ${balance_scheme} ${bayescv_iteration}
	"""
}