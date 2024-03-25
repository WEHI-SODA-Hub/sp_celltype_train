#!/bin/env nextflow 

// Enable DSL-2 syntax
nextflow.enable.dsl=2

process REPORT {

    memory "4 GB"
    cpus 1
    conda "${projectDir}/envs/environment.yml"
    time "30min"
    label "report"
    publishDir "${params.output_path}", mode: 'copy'

    input:
    path input_data
    path decoder
    path toml

    output:
    path "*.html"

    shell:
    '''
    # collate each field
    unique_preprocessing_schemes=$(for diri in *-*-*-*; do echo ${diri} | rev | cut -d '-' -f 3 | rev; done | sort -u)
    unique_balance_schemes=$(for diri in *-*-*-*; do echo ${diri} | rev | cut -d '-' -f 2 | rev; done | sort -u)
    unique_bayescv_iterations=$(for diri in *-*-*-*; do echo ${diri} | rev | cut -d '-' -f 1 | rev; done | sort -u)

    # create reports for balance schemes
    for pp_scheme in $unique_preprocessing_schemes;
    do
        for b_iter in $unique_bayescv_iterations;
        do
            reportname="!{params.run_name}-ALL_BALANCE_SCHEMES-${pp_scheme}-${b_iter}-assess"
            python3 !{projectDir}/scripts/create-qmd.py \\
                --decoder !{decoder} \\
                --output-dir . \\
                --options-toml !{toml} \\
                -d \\
                $(ls -d !{params.run_name}-${pp_scheme}-*-${b_iter}/) >> "$reportname.qmd"
    
                quarto render "$reportname.qmd" --to html
        done
    done

    # create reports for preprocessing schemes
    for b_scheme in $unique_balance_schemes;
    do
        for b_iter in $unique_bayescv_iterations;
        do
            reportname="!{params.run_name}-ALL_PREPROCESSING_SCHEMES-${b_scheme}-${b_iter}-assess"
            python3 !{projectDir}/scripts/create-qmd.py \\
                --decoder !{decoder} \\
                --output-dir . \\
                --options-toml !{toml} \\
                -d \\
                $(ls -d !{params.run_name}-*-${b_scheme}-${b_iter}/) >> "$reportname.qmd"
    
                quarto render "$reportname.qmd" --to html
        done
    done

    # create reports for bayescv iterations
    for b_scheme in $unique_balance_schemes;
    do
        for pp_scheme in $unique_preprocessing_schemes;
        do
            reportname="!{params.run_name}-ALL_BAYESCV_ITERATIONS-${b_scheme}-${pp_scheme}-assess"
            python3 !{projectDir}/scripts/create-qmd.py \\
                --decoder !{decoder} \\
                --output-dir . \\
                --options-toml !{toml} \\
                -d \\
                $(ls -d !{params.run_name}-${pp_scheme}-${b_scheme}-*/) >> "$reportname.qmd"
    
                quarto render "$reportname.qmd" --to html
        done
    done

    python3 !{projectDir}/scripts/create-qmd.py \\
        --decoder !{decoder} \\
        --output-dir . \\
        --options-toml !{toml} \\
        -d \\
        --only-aggregated-stats \\
        $(ls -d */) > !{params.run_name}-aggregated-scores.qmd

    quarto render !{params.run_name}-aggregated-scores.qmd --to html

    '''

}
