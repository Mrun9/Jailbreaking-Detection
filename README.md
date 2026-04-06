# Adversarial Jailbreak Detection for Large Language Models

A two-stage jailbreak prompt detection system combining a FAISS semantic cache
with an adversarially trained transformer classifier, deployed as a browser
extension that intercepts prompts before they reach any LLM interface.

---

## Project Overview

Large language models remain vulnerable to jailbreak attacks — carefully crafted
prompts that bypass built-in safety mechanisms. This project builds a detection
pipeline that:

1. **Stage 1 (Fast path):** Uses a FAISS semantic cache to flag known jailbreak
   patterns in sub-millisecond latency.
2. **Stage 2 (Slow path):** Routes cache misses through a fine-tuned transformer
   classifier (benchmarking DistilBERT, ModernBERT, DeBERTa-v3-small).
3. **Adversarial loop:** A mutator module generates attack variants each round to
   harden the classifier against unseen jailbreak patterns.

---

## Repository Structure

```
jailbreak-detection/
├── data/                   # Raw and processed dataset files
│   └── sample/             # Small sample for quick verification
├── notebooks/
│   ├── setup.ipynb         # Environment check + dataset loading + EDA
│   └── training.ipynb      # Model fine-tuning (to be added)
├── src/
│   ├── mutator.py          # 6 mutation strategies + context injector
│   ├── detector.py         # FAISS cache + neural inference pipeline
│   └── train_loop.py       # Adversarial training orchestration
├── ui/
│   └── app.py              # Flask server (placeholder)
├── results/                # EDA plots, evaluation outputs
├── docs/
│   └── architecture.png    # System architecture diagram
├── requirements.txt
└── README.md
```

---

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/jailbreak-detection.git
cd jailbreak-detection
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. (Recommended) Use Google Colab with GPU

Open `notebooks/setup.ipynb` directly in Colab for free GPU access:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/jailbreak-detection/blob/main/notebooks/setup.ipynb)

---

## How to Run

### Run the setup notebook

```bash
jupyter notebook notebooks/setup.ipynb
```

This notebook will:
- Verify your environment and installed packages
- Load the dataset from HuggingFace
- Run basic exploratory data analysis
- Produce summary statistics and plots saved to `results/`

### Run the mutator (standalone test)

```bash
python src/mutator.py
```

### Run the detector (standalone test)

```bash
python src/detector.py
```

---

## Dataset

**Source:** [TrustAIRLab/in-the-wild-jailbreak-prompts](https://huggingface.co/datasets/TrustAIRLab/in-the-wild-jailbreak-prompts)

- **Type:** Text (English prompts)
- **Labels:** Binary — `1` = jailbreak, `0` = benign
- **Size:** ~3,800 labeled prompts (collected through December 2023)
- **Access:** Loaded directly via HuggingFace `datasets` library — no manual
  download required.

```python
from datasets import load_dataset
ds = load_dataset("TrustAIRLab/in-the-wild-jailbreak-prompts", "regular_2023_12_25")
```

---

## Author

**Mrunal Mohan Vibhute**
Applied Deep Learning — KU Leuven
Contact: [your email here]
