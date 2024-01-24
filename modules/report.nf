#!/bin/env nextflow 

// Enable DSL-2 syntax
nextflow.enable.dsl=2

process REPORT {

    memory "4 GB"
    cpus 1
    conda "${projectDir}/environment.yml"
    time "30min"
    label "report"
    publishDir "${params.output_path}", mode: 'copy'

    input:
    path input_data
    path decoder
    path toml

    output:
    path "assess.html"

    script:
    """
    python3 ${projectDir}/src/create-qmd.py \\
        --decoder ${decoder} \\
        --output-dir . \\
        --options-toml ${toml} \\
        -d \\
        ${input_data} > assess.qmd
    
    quarto render assess.qmd --to html
    """

}