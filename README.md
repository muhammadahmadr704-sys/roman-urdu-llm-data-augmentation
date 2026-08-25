
Project Title
Large Language Model Based Data Augmentation for Hate Speech Detection in Roman Urdu

##Project Overview

This repo contains the source code, datasets, synthetic data, quality control outputs, model artefacts and final evaluation results produced for the dissertation project.
The project investigates whether large language models can be used to generate useful synthetic Roman Urdu training data for fine-grained hate speech detection. The work uses the RUHSOLD dataset and focuses on the minority classes Religious Hate, Sexism and Profane. Mistral-7B-Instruct-v0.2 was adapted using QLoRA for synthetic-data generation, while XLM-R was used as the final downstream classifier. A multi stage quality control pipeline was developed to compare unfiltered synthetic augmentation with quality-controlled augmentation.

Content Warning

This project contains examples of abusive, offensive, sexist, religiously hateful and profane language. Some files contain authentic social media posts and some contain model-generated harmful content. These materials are included solely for academic research and evaluation purposes.

Directory Structure

```text

├── README.md
├── 01_data/
├── 02_notebooks/
├── 03_generator_adaptation/
├── 04_generation/
├── 05_quality_control/
├── 06_final_synthetic_datasets/
├── 07_results/
└── 08_exploratory_material/
```

Dataset and Label Mapping

The project uses the fine-grained RUHSOLD Roman Urdu hate-speech dataset.
```text
0 = Abusive or Offensive Language
1 = Normal / Non-abusive Language
2 = Religious Hate
3 = Sexism
4 = Profane
```
The final data files used in the experiments are located in `01_data/`:

```text
RUHSOLD_train.tsv

RUHSOLD_validation.tsv

RUHSOLD_test.tsv

label_definitions.txt

```
The validation split used in this project was reconstructed from the training data using stratified sampling because the validation file obtained from the original repository duplicated the training data. The official test set was kept separate and used only for final evaluation.

Software and Hardware Requirements

The project was implemented in Python using Jupyter notebooks. Main libraries include:
```text
PyTorch
Transformers
Datasets
PEFT
BitsAndBytes
Sentence-Transformers
FAISS
scikit-learn
pandas
NumPy
Optuna
Matplotlib
```

The experiments were conducted in the Queen Mary University of London JupyterHub environment using an NVIDIA A100 GPU. A CUDA capable GPU is strongly recommended for reproducing the LLM adaptation, synthetic generation and repeated classifier training experiments.

Required Pre-trained Models

Important public models used by the notebooks include:
```text
mistralai/Mistral-7B-Instruct-v0.2

FacebookAI/xlm-roberta-base

google-bert/bert-base-multilingual-cased

intfloat/multilingual-e5-large

sentence-transformers/LaBSE
```
Additional models explored during development are documented in the corresponding notebooks. Internet access and sufficient Hugging Face cache space may be required when reproducing the experiments from scratch.
Recommended Execution Order
The notebooks in `02_notebooks/` are numbered to reflect the main workflow.
`01_dataset_preparation.ipynb` – prepares the RUHSOLD data and validation split.
`02_dataset_analysis.ipynb` – performs exploratory dataset analysis.
`03_mbert_classifier_selection.ipynb` – evaluates mBERT.
`04_xlmr_classifier_selection.ipynb` – evaluates XLM-R.
`05_mmbert_classifier_selection.ipynb` – evaluates mmBERT.
`06_embedding_model_selection.ipynb` – compares multilingual embedding models.
`07_mistral_qlora_adaptation.ipynb` – adapts Mistral using QLoRA.
`08_synthetic_generation.ipynb` – generates synthetic Roman Urdu samples.
`09_prometheus_validation.ipynb` – evaluates Prometheus as a possible LLM judge.
`10_quality_control_pipeline.ipynb` – applies the final QC pipeline.
`11_unfiltered_augmentation.ipynb` – constructs the matched unfiltered condition.
`12_qc_filtered_augmentation.ipynb` – trains/evaluates the QC-filtered augmented condition.
`13_original_xlmr.ipynb` – trains/evaluates the original-data XLM-R baseline.
`14_mistral_zero_shot_classification.ipynb` – evaluates the generation-adapted Mistral model as a classifier.
`15_final_test_results.ipynb` – aggregates and analyses final held-out test results.
Generator Adaptation
The directory `03_generator_adaptation/` contains the data and configuration used for Mistral adaptation:
```text
cleaned_ruhsold_train.csv
train_sft.jsonl
eval_sft.jsonl
experiment_config.json
```
The final QLoRA adapter is included in:
```text
03_generator_adaptation/final_mistral_adapter/
```
This adapter should be loaded on top of `mistralai/Mistral-7B-Instruct-v0.2`; it is not a standalone full model.
Synthetic Generation
The directory `04_generation/` contains:
`curated_demonstration_pools/` – high-precision class-specific demonstrations used in prompting.
`pilot_generation/` – pilot generations used during development.
`round1_raw_generation/` – first full generation round.
`round2_raw_generation/` – second full generation round.
Quality-Control Pipeline
The directory `05_quality_control/` contains the intermediate and final QC artefacts.
Prometheus and human validation
`05_quality_control/prometheus_and_human_validation/` contains human annotation files, Prometheus outputs, diagnostic results, parsing failures and supporting notebooks/spreadsheets. These files document the evaluation that led to Prometheus being excluded from the final QC pipeline.
Final QC stages
The final QC procedure consists of:
Basic validity and formatting checks.
XLM-R target-label verification.
Semantic similarity filtering against authentic same-class RUHSOLD examples using multilingual-E5 and FAISS.
Synthetic-sample diversity filtering.
`round1_qc/` and `round2_qc/` contain audit, passed and rejected files from the individual stages.
Authoritative Final Synthetic Datasets
The exact synthetic datasets used in the final downstream comparison are located in:
```text
06_final_synthetic_datasets/
```
The two authoritative files are:
```text
qc_filtered_1x_synthetic_training_set.csv
unfiltered_1x_synthetic_training_set.csv
```
These should be used when reproducing the final augmentation experiments.
Final XLM-R Results
The authoritative final downstream results are located in:
```text
07_results/xlmr_final_corrected_comparison/
```
This directory contains three experimental conditions:
```text
original_xlm_r/
unfiltered_1x/
qc_filtered_1x/
```
Each condition contains per seed test predictions and classification reports for random seeds 42, 43 and 44.
Combined result files include:
```text
all_xlmr_test_runs.csv
all_xlmr_per_class_test_results.csv
xlmr_test_condition_summary.csv
xlmr_per_class_test_summary.csv
```
These files contain the final corrected XLM-R results reported in the dissertation.
Mistral Classification Results
Direct classification results for the generation-adapted Mistral model are stored in:
```text
07_results/mistral_zero_shot_classification/
```
This folder contains validation predictions, test predictions, overall test summaries and per-class results. The model was adapted for generation rather than classification, so this comparison should be interpreted in that context.
Exploratory Material
The directory `08_exploratory_material/` contains development stage experiments that informed the final methodology but are not part of the authoritative final evaluation. These include base Mistral prompting, Qwen experiments and generator evaluation material.
Executable Software
A standalone executable file is not provided because this project is implemented as a sequence of Python Jupyter notebooks for dataset preparation, model training, LLM adaptation, synthetic-data generation, quality control and evaluation.
To run the project, an examiner should:
Create a Python environment with the required libraries.
Use a CUDA capable GPU for computationally intensive stages.
Download the required public pre-trained models from Hugging Face.
Open the notebooks in `02_notebooks/`.
Update file paths where necessary for the local environment.
Execute the notebooks in the numbered order shown above.
Use the supplied final Mistral adapter for generation experiments, or retrain it with Notebook 07.
Use the supplied final synthetic datasets and result files when inspecting the final evaluation without retraining all models.
Included and Excluded Model Files
The final Mistral QLoRA adapter is included because it is a central artefact of the synthetic-generation pipeline and is relatively compact compared with full model checkpoints.
The full base Mistral model is not included because it is publicly available from Hugging Face.
The trained XLM-R checkpoint weights are not included. The final experiment contains multiple XLM-R models across three conditions and three random seeds, and each checkpoint is approximately 1.1 GB. Including all checkpoints would substantially increase the supporting-material size. Instead, the submission includes the source notebooks, final datasets, random seeds, per-seed predictions, classification reports and summary results needed to inspect and reproduce the evaluation.
Reproducibility Notes
The final downstream comparison uses random seeds:
```text
42
43
44
```
The same held out RUHSOLD test set is used across the original, unfiltered and quality controlled conditions. The unfiltered and quality controlled conditions contain the same number of synthetic examples for each target class so that the comparison focuses on the effect of quality control rather than data quantity.
Some notebook paths reflect the original Queen Mary JupyterHub project directory and may need to be changed when the project is run elsewhere. CUDA versions, package versions and model download versions may also affect exact reproduction. The supplied predictions and final result summaries therefore provide the authoritative record of the experiments reported in the dissertation.
Storage and Distribution
Generated harmful language data should be handled responsibly and should not be redistributed outside legitimate academic or research contexts without appropriate consideration of ethical and data protection issues.
Project Context
This supporting material accompanies an MSc dissertation submitted to the School of Electronic Engineering and Computer Science, Queen Mary University of London.
