# Adversarial Jailbreak Detection for Large Language Models

A two-stage jailbreak prompt detection system that combines a FAISS semantic cache with an adversarially trained transformer classifier, designed to screen prompts before they reach a downstream LLM interface.

<img width="2604" height="1632" alt="image" src="https://github.com/user-attachments/assets/6509c903-0fd6-4b0c-9dd9-f00dbc69be09" />

## Project Overview

Large language models remain vulnerable to jailbreak and prompt-injection attacks. This project builds a practical detection pipeline with:

1. **Stage 1: semantic cache** using FAISS to flag known jailbreak patterns with very low latency.
2. **Stage 2: neural classifier** using a fine-tuned transformer to classify prompts that miss the cache.
3. **Adversarial loop** using prompt mutation to generate harder jailbreak variants and improve robustness over multiple rounds.

The current Deliverable 3 system combines:

- a processed prompt dataset in [results/collected_prompts.csv](/Users/mrunal/Documents/Projects/ADL/Jailbreak Detection/results/collected_prompts.csv)
- a fine-tuned DistilBERT classifier with saved experiment checkpoints
- a two-stage detector in [src/detector.py](/Users/mrunal/Documents/Projects/ADL/Jailbreak Detection/src/detector.py)
- adversarial mutation and training utilities in [src/mutator.py](/Users/mrunal/Documents/Projects/ADL/Jailbreak Detection/src/mutator.py) and [src/train_loop.py](/Users/mrunal/Documents/Projects/ADL/Jailbreak Detection/src/train_loop.py)
- a refined Flask demo UI in [ui/app.py](/Users/mrunal/Documents/Projects/ADL/Jailbreak Detection/ui/app.py)
- Deliverable 3 analysis artifacts in [results/deliverable3](/Users/mrunal/Documents/Projects/ADL/Jailbreak Detection/results/deliverable3)

## Main Page Overview

### System Architecture

The project uses an offline training pipeline and a runtime detection pipeline. During inference, prompts are checked against the cache first and then passed to the neural classifier if needed.

![System architecture](docs/system_architecture.png)

### Interface Preview

The refined Deliverable 3 UI explains the chosen operating mode, shows whether the result came from the cache or the model, and presents the output in a reviewer-friendly format.

![Cache-hit interface example](docs/ui_cache_hit_example.png)

## Deliverable 3 Highlights

- The deployment default is the calibrated baseline classifier rather than the adversarial branch, because the adversarial/cache-backed run over-flagged too many benign mutated prompts.
- The repo includes explicit `balanced` and `strict` operating modes:
  - `balanced` uses threshold `0.51` and gives the best clean-test F1.
  - `strict` uses threshold `0.85` and reduces wrapper-triggered false positives on mutated prompts.
- The UI was upgraded from a bare form into a presentation-ready dashboard with:
  - operating-mode selection
  - example prompts
  - detector status
  - stage, threshold, latency, similarity, and recommendation output
- A lightweight Deliverable 3 analysis script now generates:
  - threshold sweeps
  - operating-mode summaries
  - per-source and per-category breakdowns
  - false-positive examples for report writing and reflection

## Repository Layout

```text
Jailbreak Detection/
├── docs/
│   ├── system_architecture.png
│   ├── ui_cache_hit_example.png
│   ├── deliverable3_architecture.md
│   ├── Description.pdf
│   └── Jailbreaking_Detection-Blueprint.pdf
├── notebooks/
│   ├── setup.ipynb
│   ├── adversarial_training_from_splits.ipynb
│   ├── deliverable3_extended_evaluation.ipynb
│   ├── visualize_distilbert_experiment_results.ipynb
│   ├── visualize_distilbert_training_old.ipynb
│   └── model training/distilbert_detector_train_old.py
├── results/
│   ├── collected_prompts.csv
│   ├── deliverable3/
│   ├── distilbert_experiment/
│   ├── ui_cache/
│   └── screenshots and demo outputs
├── src/
│   ├── detector.py
│   ├── mutator.py
│   ├── train_loop.py
│   └── helper/
│       ├── deliverable3_analysis.py
│       ├── distilbert_eval_only.py
│       ├── distilbert_experiment.py
│       └── mutator_test.py
├── ui/
│   └── app.py
├── requirements.txt
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/jailbreak-detection.git
cd jailbreak-detection
```

### 2. Create a Python environment

You can use either `venv` or Conda.

Using `venv`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

Using Conda:

```bash
conda create -n jailbreak-detector python=3.10 -y
conda activate jailbreak-detector
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Apple Silicon notes

If you are on a Mac with Apple Silicon, `faiss-cpu` can be tricky. If the pip wheel fails or conflicts with Conda FAISS, use this order instead:

```bash
conda create -n jailbreak-detector python=3.10 -y
conda activate jailbreak-detector
python -m pip install --upgrade pip
conda install -c conda-forge faiss=1.10.0 libfaiss=1.10.0 -y
pip install -r requirements.txt
```

Verify key packages after install:

```bash
python -c "import faiss; print('faiss ok:', faiss.__version__)"
python -c "import torch; print('mps available:', torch.backends.mps.is_available())"
python -c "import torch, transformers, numpy, sklearn, flask, nltk, huggingface_hub; print('ok')"
```

### 4. Google Colab option

Open [notebooks/setup.ipynb](/Users/mrunal/Documents/Projects/ADL/Jailbreak Detection/notebooks/setup.ipynb) in Colab for a quick environment check and free GPU access:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mrun9/jailbreak-detection/blob/main/notebooks/setup.ipynb)

### 5. Notes

- The Deliverable 3 analysis script uses only the Python standard library, so it can still run in lightweight environments even when the full ML stack is unavailable.
- The UI now supports persisted cache reuse, so only the first warm-up run needs to build the seeded FAISS cache from scratch.

## How to Run

### Run the setup notebook

```bash
jupyter notebook notebooks/setup.ipynb
```

This notebook can be used to:

- verify your environment and installed packages
- inspect the dataset
- run basic exploratory data analysis
- produce summary statistics and plots saved to `results/`

### Reuse or rerun the main experiment

To rerun the baseline-versus-adversarial experiment from the saved dataset:

```bash
python3 src/helper/distilbert_experiment.py
```

If the checkpoints and splits already exist and you only want to regenerate the evaluation CSVs:

```bash
python3 src/helper/distilbert_eval_only.py --force-eval
```

### Generate the Deliverable 3 analysis artifacts

```bash
python3 src/helper/deliverable3_analysis.py
```

This writes:

- [results/deliverable3/deliverable3_summary.json](/Users/mrunal/Documents/Projects/ADL/Jailbreak Detection/results/deliverable3/deliverable3_summary.json)
- [results/deliverable3/operating_modes.csv](/Users/mrunal/Documents/Projects/ADL/Jailbreak Detection/results/deliverable3/operating_modes.csv)
- [results/deliverable3/threshold_sweep.csv](/Users/mrunal/Documents/Projects/ADL/Jailbreak Detection/results/deliverable3/threshold_sweep.csv)
- [results/deliverable3/source_breakdown.csv](/Users/mrunal/Documents/Projects/ADL/Jailbreak Detection/results/deliverable3/source_breakdown.csv)
- [results/deliverable3/category_breakdown.csv](/Users/mrunal/Documents/Projects/ADL/Jailbreak Detection/results/deliverable3/category_breakdown.csv)

### Generate the Deliverable 3 report

```bash
python3 reports/generate_deliverable3_report.py
```

Outputs:

- [reports/deliverable3_report.md](/Users/mrunal/Documents/Projects/ADL/Jailbreak Detection/reports/deliverable3_report.md)
- [reports/deliverable3_report.pdf](/Users/mrunal/Documents/Projects/ADL/Jailbreak Detection/reports/deliverable3_report.pdf)

### Launch the refined interface

```bash
python3 ui/app.py
```

Then open `http://localhost:5000`.

The UI:

- auto-discovers the repo-shipped baseline checkpoint
- seeds the cache from known jailbreak prompts in the processed CSV
- reuses the saved UI cache on later runs when the dataset has not changed
- exposes `Balanced` and `Strict` operating modes
- reports detector stage, threshold, similarity, confidence, latency, and recommendation

### Run standalone checks

Run the mutator:

```bash
python3 src/mutator.py
```

Run the detector:

```bash
python3 src/detector.py
```

## Dataset

The final processed runtime dataset is stored in [results/collected_prompts.csv](/Users/mrunal/Documents/Projects/ADL/Jailbreak Detection/results/collected_prompts.csv).

**Sources**

1. `walledai/AdvBench`
2. `allenai/wildjailbreak`
3. `TrustAIRLab/in-the-wild-jailbreak-prompts`
4. `aurora-m/redteam`
5. `JailbreakBench/JBB-Behaviors`
6. `deepset/prompt-injections`

Source references:

1. https://huggingface.co/datasets/walledai/AdvBench/viewer/default/train?f%5Btarget%5D%5Bmin%5D=130&f%5Btarget%5D%5Bmax%5D=144&f%5Btarget%5D%5Btransform%5D=length
2. https://huggingface.co/datasets/allenai/wildjailbreak/viewer/eval
3. https://huggingface.co/datasets/TrustAIRLab/in-the-wild-jailbreak-prompts/viewer/jailbreak_2023_12_25
4. https://huggingface.co/datasets/aurora-m/redteam/viewer/default/train?p=60
5. https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors/viewer/behaviors/benign?row=3
6. https://huggingface.co/datasets/deepset/prompt-injections/viewer/default/train?f%5Blabel%5D%5Bmin%5D=1&f%5Blabel%5D%5Bimax%5D=1

Dataset details:

- **Type:** English text prompts
- **Labels:** binary, where `1` means jailbreak and `0` means benign
- **Access:** compiled into the local processed CSV used by the detector and evaluation pipeline

## Updated Performance Summary

The Deliverable 3 recommendation is to deploy the calibrated baseline (`Model A`) and keep the adversarial branch as an experimental comparison.

| Configuration | Dataset | Threshold | Accuracy | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Baseline default | Clean test | 0.50 | 0.9561 | 0.9458 | 0.9509 | 0.9484 |
| Balanced mode | Clean test | 0.51 | 0.9565 | 0.9476 | 0.9500 | 0.9488 |
| Balanced mode | Mutated data | 0.51 | 0.7196 | 0.6540 | 0.8916 | 0.7545 |
| Strict mode | Mutated data | 0.85 | 0.7501 | 0.7097 | 0.8173 | 0.7597 |
| Adversarial default | Mutated data | 0.50 | 0.4929 | 0.4880 | 1.0000 | 0.6559 |

Important interpretation:

- `Balanced` is the best default because it slightly improves clean-test F1 without changing the user experience much.
- `Strict` is the best demo mode when false positives are more expensive than missed detections.
- The adversarial run still matters because it exposed where the pipeline was unstable, but it is not the final recommended deployment checkpoint.

## Interface and Evidence

Relevant Deliverable 3 artifacts:

- Refined interface: [ui/app.py](/Users/mrunal/Documents/Projects/ADL/Jailbreak Detection/ui/app.py)
- Extended evaluation notebook: [notebooks/deliverable3_extended_evaluation.ipynb](/Users/mrunal/Documents/Projects/ADL/Jailbreak Detection/notebooks/deliverable3_extended_evaluation.ipynb)
- Report PDF: [reports/deliverable3_report.pdf](/Users/mrunal/Documents/Projects/ADL/Jailbreak Detection/reports/deliverable3_report.pdf)
- Existing screenshots and demo captures: [results](/Users/mrunal/Documents/Projects/ADL/Jailbreak Detection/results)

## Known Issues and Warnings

- The local machine used for grading may not have the full ML stack installed. Training and live detector inference require `torch`, `transformers`, `flask`, and related packages from `requirements.txt`.
- The adversarial/cache-backed run is intentionally not the default because it produces too many false positives on mutated benign prompts.
- Some benign prompt families remain hard for the model, especially wrapper-heavy prompts from the TrustAIRLab regular source and several JBB benign categories.
- The cache currently stores raw matched prompts for explainability. That is useful for demos, but a real deployment should minimize or redact stored content.
- The repository currently ships a refined Flask application for demonstration, not a packaged browser extension.

## Contact

**Mrunal Mohan Vibhute**  
Applied Deep Learning  
Email: [mvibhute@ufl.edu](mailto:mvibhute@ufl.edu)
