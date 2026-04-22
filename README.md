# Adversarial Jailbreak Detection for Large Language Models

This project detects jailbreak and prompt-injection attempts before they reach a downstream LLM. The current Deliverable 3 system combines:

- a processed prompt dataset in [results/collected_prompts.csv](results/collected_prompts.csv)
- a fine-tuned DistilBERT classifier with saved experiment checkpoints
- a two-stage detector in [src/detector.py](src/detector.py)
- adversarial mutation and training utilities in [src/mutator.py](src/mutator.py) and [src/train_loop.py](src/train_loop.py)
- a refined Flask demo UI in [ui/app.py](ui/app.py)
- Deliverable 3 analysis artifacts in [results/deliverable3](results/deliverable3)

## Main Page Overview

### System Architecture

The project uses an offline training pipeline and a runtime detection pipeline. During inference, prompts are checked against the cache first and then passed to the neural classifier if needed.

![System architecture](docs/system_architecture.png)

### Interface Preview

The refined Deliverable 3 UI explains the chosen operating mode, shows whether the answer came from the cache or the model, and presents the result in a more reviewer-friendly way.

![Cache-hit interface example](docs/ui_cache_hit_example.png)

## Deliverable 3 Highlights

- The deployment default is now the calibrated baseline classifier rather than the adversarial branch. Extended evaluation showed that the adversarial/cache-backed run over-flagged too many benign mutated prompts.
- The repo now includes explicit `balanced` and `strict` operating modes:
  - `balanced` uses threshold `0.51` and gives the best clean-test F1.
  - `strict` uses threshold `0.85` and reduces wrapper-triggered false positives on mutated prompts.
- The UI was upgraded from a bare form into a presentation-ready dashboard with:
  - operating-mode selection
  - example prompts
  - detector status
  - stage, threshold, latency, and recommendation output
- A lightweight Deliverable 3 analysis script now generates:
  - threshold sweeps
  - operating-mode summaries
  - per-source and per-category breakdowns
  - false-positive examples for report writing and reflection

## Repository Layout

```text
Jailbreak Detection/
├── docs/
├── notebooks/
│   ├── adversarial_training_from_splits.ipynb
│   ├── deliverable3_extended_evaluation.ipynb
│   └── ...
├── reports/
│   ├── generate_deliverable3_report.py
│   ├── deliverable3_report.md
│   ├── deliverable3_report.pdf
│   └── ...
├── results/
│   ├── collected_prompts.csv
│   ├── deliverable3/
│   └── distilbert_experiment/
├── src/
│   ├── helper/
│   │   ├── deliverable3_analysis.py
│   │   ├── distilbert_eval_only.py
│   │   └── distilbert_experiment.py
│   ├── detector.py
│   ├── mutator.py
│   └── train_loop.py
├── ui/
│   └── app.py
└── requirements.txt
```

## Installation

Create and activate a Python environment, then install the project dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

Notes:

- `faiss-cpu` can be tricky on Apple Silicon. If the pip wheel fails, install FAISS through `conda-forge` first and then run `pip install -r requirements.txt`.
- The Deliverable 3 analysis script uses only the Python standard library, so it can still run in lightweight environments even when the full training stack is unavailable.

## How to Run the Refined Pipeline

### 1. Reuse or rerun the main experiment

To rerun the baseline-versus-adversarial experiment from the saved dataset:

```bash
python3 src/helper/distilbert_experiment.py
```

If the checkpoints and splits already exist and you only want to regenerate the four evaluation CSVs:

```bash
python3 src/helper/distilbert_eval_only.py --force-eval
```

### 2. Generate the Deliverable 3 analysis artifacts

```bash
python3 src/helper/deliverable3_analysis.py
```

This writes:

- [results/deliverable3/deliverable3_summary.json](results/deliverable3/deliverable3_summary.json)
- [results/deliverable3/operating_modes.csv](results/deliverable3/operating_modes.csv)
- [results/deliverable3/threshold_sweep.csv](results/deliverable3/threshold_sweep.csv)
- [results/deliverable3/source_breakdown.csv](results/deliverable3/source_breakdown.csv)
- [results/deliverable3/category_breakdown.csv](results/deliverable3/category_breakdown.csv)

### 3. Generate the Deliverable 3 report

```bash
python3 reports/generate_deliverable3_report.py
```

Outputs:

- [reports/deliverable3_report.md](reports/deliverable3_report.md)
- [reports/deliverable3_report.pdf](reports/deliverable3_report.pdf)

### 4. Launch the refined interface

```bash
python3 ui/app.py
```

Then open `http://localhost:5000`.

The UI now:

- auto-discovers the repo-shipped baseline checkpoint
- seeds the cache from known jailbreak prompts in the processed CSV
- exposes `Balanced` and `Strict` operating modes
- reports detector stage, threshold, similarity, confidence, latency, and recommendation

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

- Refined interface: [ui/app.py](ui/app.py)
- Extended evaluation notebook: [notebooks/deliverable3_extended_evaluation.ipynb](notebooks/deliverable3_extended_evaluation.ipynb)
- Report PDF: [reports/deliverable3_report.pdf](reports/deliverable3_report.pdf)
- Existing screenshots and demo captures: [results](results)

## Known Issues and Warnings

- The local machine used for grading may not have the full ML stack installed. Training and live detector inference require `torch`, `transformers`, `flask`, and related packages from `requirements.txt`.
- The adversarial/cache-backed run is intentionally not the default because it produces too many false positives on mutated benign prompts.
- Some benign prompt families remain hard for the model, especially wrapper-heavy prompts from the TrustAIRLab regular source and several JBB benign categories.
- The cache currently stores raw matched prompts for explainability. That is useful for demos, but a real deployment should minimize or redact stored content.
- The repository contains a refined Flask app, not a packaged browser extension.

## Contact

Mrunal Mohan Vibhute  
Applied Deep Learning  
Email: [mvibhute@ufl.edu](mailto:mvibhute@ufl.edu)
