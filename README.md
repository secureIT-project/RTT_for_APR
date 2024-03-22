
[![source under MIT licence](https://img.shields.io/badge/source%20license-MIT-green)](LICENSE.txt)
[![data under CC BY 4.0 license](https://img.shields.io/badge/data%20license-CC%20BY%204.0-green)](https://creativecommons.org/licenses/by/4.0/)
<a href="https://doi.org/10.5281/zenodo.10500593">
  <img align="right" src="https://zenodo.org/badge/DOI/10.5281/zenodo.10500593.svg" alt="DOI: 10.5281/zenodo.10500593">
 </a>
 
# Replication Package for _"Assessing the Latent Automated Program Repair Capabilities of Large Language Models using Round-Trip Translation"_

This repository contains the replication package for the paper "Assessing the Latent Automated Program Repair Capabilities of Large Language Models using Round-Trip Translation" by Fernando Vallecillos Ruiz, Anastasiia Grishina, Max Hort and Leon Moonen, which is currently under review.

An earlier version was deposited on arXiv with DOI: [10.48550/arXiv.2401.07994](https://doi.org/10.48550/arXiv.2401.07994) with title "A Novel Approach for Automated Program Repair using Round-Trip Translation with Large Language Models".

The replication package is archived on Zenodo with DOI: [10.5281/zenodo.10500593](https://doi.org/10.5281/zenodo.10500593). It is maintained on GitHub at <https://github.com/secureIT-project/RTT_for_APR>. 

The source code is distributed under the MIT license, and except for 3rd party datasets that come with their own license, all documentation, data, models and results in this repository are distributed under the CC BY 4.0 license. 


## Organization

The replication package is organized as follows:

- clm-apr
    - plbart: code to generate patches with PLBART models.
    - codet5: code to generate patches with CodeT5 models.
    - transcoder: code to generate patches with the TransCoder model.
    - incoder: code to generate patches with InCoder models.
    - santacoder: code to generate patches with the SantaCoder model.
    - starcoder: code to generate patches with the StarCoderBase model.
    - quixbugs: code to validate patches generated for the QuixBugs benchmark.
    - defects4j: code to validate patches generated for any of the Defects4J benchmarks.
    - humaneval: code to validate patches generated for the HumanEval-Java benchmark.
- humaneval-java: the HumanEval-Java benchmark proposed by Jiang et al. 2023
- jasper: a Java tool to parse Java programs needed to preprocess input.
- model: folder to download the language models.
- analysis\_wandb: data from WandB and Jupyter notebook to create graphs.
- tmp\_benchmarks: folder for temporary files used in patch validation. 
       The folder may contain pairs of `paralell' folders src and src_org for
       each benchmark, used to replace buggy code with candidate patches.


## Replication

### Prerequisites

- Python version: 3.8--3.10.
- [Git LFS](https://git-lfs.com/) is required for model downloading.

#### Weight and Biases (WandB)
1.  Create an account on  [Weights and Biases](https://wandb.ai/)
2.  Install the  [Weights and Biases](https://docs.wandb.ai/ref/python)  library
3.  Run  `wandb login`  and follow the instructions

#### Set up OpenAI access

OpenAI account is needed with access to  `gpt-3.5-turbo`  and `gpt-4` . The  `OPENAI_API_KEY`  environment variable should be set to your OpenAI API access token.

### Dependencies

- [Defects4J](https://github.com/rjust/defects4j)
  To generate inputs for the Defects4J datasets or to validate them, you need 
  to have installed [their tool](https://github.com/rjust/defects4j).
- Java 8
- Apache Maven


### Setup

We recommend the use of the setup script: 

    setup.sh

which performs the following:

 1. Creates a virtual environment for Python and activate it.
 2. Install the packages in `requirements.txt`.
 3. Compiles Jasper.
 4. Downloads parsers.
 5. Check if the Defects4J installation is correct.


### Download models

The following bash script contains the code to download all of the models used: 

```
models/download_models.sh
```

We recommend downloading only the models you are going to use due to their size

```
cd models
chmod +x download_models.sh
./download_models.sh
```

To run one specific model, for example, PLBART (C\#), use the following commands:

```
cd models
git lfs install
git clone https://huggingface.co/uclanlp/plbart-java-cs
git clone https://huggingface.co/uclanlp/plbart-cs-java
cd ../..
```


### Step 1: Preprocessing and Prompting:

Each script in each `clm-apr/[model]` folder connects one or more models with 
one dataset. These scripts follow the template: [benchmark]\_[model]\_[technique].py. 
The scripts first create an `[model]_input.json` file with the preprocessed
input. Then generate outputs based on that file with one or more models. 
For example:

```
cd clm-apr/plbart
python quixbugs_plbart_round.py         # Generates input for QuixBugs and generate patches using Java<->C# RTT.
python quixbugs_plbart_round_nl.py      # Generates input for QuixBugs and generate patches using Java<->NL RTT.
```

Optionally, use argument `--device_map cpu` if you wish to run the script on 
CPU, for example:

```
python quixbugs_plbart_round.py --device_map cpu
```

Otherwise, the script will be run on all available CUDA GPU's. 

We have commented the generation of inputs in the scripts. Users are free to 
uncomment this method and try for themselves. It is easily recognizable by 
their name template `[model]_[benchmark]_input()`. In the previous case:

```
quixbugs_plbart_input()
```


### Step 2 and 3: Round Trip Translation and Postprocessing

These steps are also included in the [benchmark]\_[model]\_[technique].py 
script mentioned above. They are modularized in the method recognizable by 
their name template [model]_[benchmark]_output(). 
For example:

```
quixbugs_incoder_output()
```

This method:

 1. Reads the input json file.
 2. Generates outputs through the LLM.
 3. Postprocess the output (extract the patch, clean up extra token, etc.).
 4. Creates [model]\_output\_[technique]\_[extra].json.

The last 3 steps are repeated according to the number of runs set to performed 
(10 in our experiments). Each run will produce a different file with the seed 
used in its generation. For example, `quixbugs\_plbart\_round.py` and 
`quixbugs\_plbart\_round_nl.py` scripts create:

```
clm-apr/quixbugs/plbart_results/run_0/plbart_java_cs_java_output_round_csharp_batch.json
clm-apr/quixbugs/plbart_results/run_0/plbart_java_nl_java_output_round_nl_batch.json
```


### Step 4: Evaluation of RTT Results:

The last step evaluates the generated outputs against the test-suites of each 
benchmark. This script reads the previous outputs files and generates a new one 
with the results of the test for one model. Furthermore, it connects with the 
_WandB_ tool to calculate metrics and send them to analyze.

Following the previous examples, to validate the results previously obtained, 
we execute the following:

```
cd clm-apr/quixbugs
python validate_quixbugs_parallel.py
```

Given the included JSON, this script would create:

```
clm-apr/quixbugs/plbart_results/run_0/plbart_java_cs_java_validate_round_csharp_batch.json
```

We have disabled _WandB_ in the script to allow users to try the script first. 
However, it can be easily activated by changing the parameter `mode="disabled"` 
to `mode="online"`.
We have set the variable `total_runs = 1`, as well as `input_file` and `output_file`
to the results included. They should be modified accordingly to validate more runs
or to validate other files/models.


### Included Results

We include two CSV files obtained through WandB. 

```
'data_cleaned_grouped.csv': Aggregated metrics of the 25 outputs for all runs.
'full_data_all_runs.csv':  All metrics for all outputs on all runs.
```

## Citation

If you build on this data or code, please cite this work by referring to the paper:

```
@misc{ruiz2024:rtt:arxiv,
   title = {A Novel Approach for Automated Program Repair using Round-Trip Translation with Large Language Models},
   author = {Vallecillos Ruiz, Fernando and Anastasiia Grishina and Max Hort and Leon Moonen},
   year = {2024},
   month = jan,
   number = {arXiv:2401.07994},
   eprint = {2401.07994},
   primaryclass = {cs},
   publisher = {{arXiv}}
}
```

## Changelog
- v0.1 - initial replication package corresponding to v1 of arXiv deposit: includes raw data, code, and example outputs. 


## Acknowledgement

The work included in this repository was supported by the Research Council of Norway through the secureIT project (IKTPLUSS #288787). Max Hort is supported through the ERCIM ‘Alain Bensoussan’ Fellowship Programme. The empirical evaluation was performed on the Experimental Infrastructure for Exploration of Exascale Computing (eX3), financially supported by the Research Council of Norway under contract #270053.


## References

Jiang, N.; Liu, K.; Lutellier, T.; and Tan, L. 2023. Impact of Code Language 
Models on Automated Program Repair. In 45th International Conference on 
Software Engineering (ICSE), 1430–1442. IEEE. ISBN 978-1-66545-701-9.
