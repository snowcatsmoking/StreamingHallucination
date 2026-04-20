# Streaming Hallucination Detection in Long Chain-of-Thought Reasoning

This repository contains the code and dataset for the paper:

> **Streaming Hallucination Detection in Long Chain-of-Thought Reasoning**
> Haolang Lu, Minghui Pan, Ripeng Li, Guoshun Nan, Jialin Zhuang, Zijie Zhao, Zhongxiang Sun, Kun Wang, Yang Liu
> arXiv:2601.02170

## Overview

We model hallucination in long CoT reasoning as an evolving latent state rather than a one-off erroneous event. Our framework derives two complementary signals at each reasoning step:

- **Step-level hallucination** (`c_t^step`): a local alarm estimated from the step's token hidden states via a lightweight linear probe.
- **Prefix-level hallucination** (`c_t^prefix`): a global state tracking whether the entire reasoning prefix has been contaminated, trained with an anchor loss and a quadratic alarm synchronisation loss guided by the step-level teacher.

## Repository Structure

```
.
├── dataset/                   # CHAINED dataset pipeline
│   ├── generate.py            # Generate CoT responses from an LLM
│   ├── merge.py               # Combine splits / inject hallucination labels
│   ├── tag.py                 # Annotate per-step hallucination labels via a judge LLM
│   ├── validate.py            # Check label completeness and logical consistency
│   └── README.md              # Dataset details
│
├── tag_validation/            # Annotation quality validation
│   ├── analyze_tags.py        # Internal logical consistency checks
│   ├── compare_datasets.py    # Agreement rate vs. GPT-4o re-annotation
│   └── advanced_validation.py # Statistical validation (chi-square, correlations)
│
├── src/                       # Core method
│   ├── process.py             # Extract per-step token hidden states from a transformer layer
│   ├── extract_features.py    # Compute 20480-d step feature vectors from JSONL
│   ├── train_teacher.py       # Train step-level (teacher) probe with LBFGS
│   ├── train.py               # Train prefix-level probe with anchor + sync loss
│   └── visualize.py           # Plot prediction curves along CoT trajectories
│
└── pyproject.toml             # Dependencies (use with uv)
```

## Dataset

The CHAINED dataset is available on [Hugging Face](https://huggingface.co/datasets/JlinZ/CHAINED). Download it via:

```bash
python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download(repo_id="JlinZ/CHAINED", repo_type="dataset", local_dir="dataset/data")
EOF
```

See [`dataset/DATASET.md`](dataset/DATASET.md) for the full construction pipeline.

## Installation

```bash
uv sync
```

Or with pip:

```bash
pip install -e .
```

## Usage

### Step 1 — Extract hidden states

Run a forward pass over the dataset and save per-step token hidden states from a specific transformer layer:

```bash
python src/process.py \
    --input   dataset/data/train/Llama_train.json \
    --output  processed/Llama_train.jsonl \
    --model   meta-llama/Llama-3.1-8B-Instruct \
    --layer   16 \
    --device  cuda:0
```

### Step 2 — Extract features

Compute the 20480-d feature vectors (5 aggregation strategies × 4096-d):

```bash
python src/extract_features.py \
    --input  processed/Llama_train.jsonl processed/Llama_test.jsonl \
    --output features/Llama_train.pt     features/Llama_test.pt
```

### Step 3 — Train teacher probe

Train the step-level teacher probe (LBFGS on `step_time_exp` features):

```bash
python src/train_teacher.py \
    --train  features/Llama_train.pt \
    --test   features/Llama_test.pt \
    --output models/Llama_teacher.pth
```

### Step 4 — Train prefix probe

Train the prefix-level probe with anchor + sync loss:

```bash
python src/train.py \
    --train   features/Llama_train.pt \
    --test    features/Llama_test.pt \
    --teacher models/Llama_teacher.pth \
    --output  models/Llama_prefix_probe.pth \
    --baseline-output models/Llama_baseline.pth
```

### Step 5 — Visualize

Plot prediction trajectories for representative CoT sequences:

```bash
python src/visualize.py \
    --test       features/Llama_test.pt \
    --teacher    models/Llama_teacher.pth \
    --baseline   models/Llama_baseline.pth \
    --prefix     models/Llama_prefix_probe.pth \
    --output-dir results/viz
```

## License

The code in this repository is released under the [MIT License](LICENSE).

The CHAINED dataset is derived from [MuSiQue](https://github.com/StonyBrookNLP/musique) (CC BY 4.0) and [BIG-Bench Hard](https://github.com/suzgunmirac/BIG-Bench-Hard) (MIT). Accordingly, the dataset is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). If you use the dataset, please cite both this work and the original sources.

## Citation

```bibtex
@article{lu2026streaming,
  title={Streaming Hallucination Detection in Long Chain-of-Thought Reasoning},
  author={Lu, Haolang and Pan, Minghui and Li, Ripeng and Nan, Guoshun and Zhuang, Jialin and Zhao, Zijie and Sun, Zhongxiang and Wang, Kun and Liu, Yang},
  journal={arXiv preprint arXiv:2601.02170},
  year={2026}
}
```
