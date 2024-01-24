#!/bin/env nextflow 

// Enable DSL-2 syntax
nextflow.enable.dsl=2

process REPORT {

    memory "4 GB"
    cpus 1
    conda "quarto tabulate pandas=1.4.4 scikit-learn=1.1.1 seaborn=0.11.2 matplotlib=3.5.3"
    time "30min"
    label "report"
    publishDir "${params.output_path}", mode: 'copy'

    input:
    path input_data
    path decoder

    output:
    path "assess.html"

    script:
    """
    python3 ${projectDir}/src/create-qmd.py \\
        --decoder ${decoder} \\
        --output-dir . \\
        -d \\
        ${input_data} > assess.qmd
    
    quarto render assess.qmd --to html
    """

}