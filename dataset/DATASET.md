# CHAINED Dataset

The CHAINED dataset is available on [Hugging Face](https://huggingface.co/datasets/JlinZ/CHAINED).  
Local copies are stored under `data/` organized by split:

```
data/
├── train/   Llama_train.json  Qwen_train.json  deepseek_train.json
├── val/     Llama_val.json    Qwen_val.json    deepseek_val.json
└── test/    Llama_test.json   Qwen_test.json   deepseek_test.json
```

Each entry contains a multi-hop question, the model's chain-of-thought response parsed into numbered steps, and per-step hallucination labels (`step_hallucination`, `cumulative_hallucination`).

## Pipeline

The dataset was constructed in four stages:

| Script | Stage |
|---|---|
| `generate.py` | Run a local LLM on MuSiQue / BBH questions to produce CoT responses |
| `merge.py merge` | Combine subsets, filter empty answers, re-index IDs |
| `tag.py` | Call a judge LLM to annotate hallucination labels per step |
| `merge.py inject` | Inject labels into the data file, drop tagging failures |

After tagging, run `validate.py <file>` to check tag completeness and internal consistency.
