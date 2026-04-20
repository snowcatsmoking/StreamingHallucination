"""
Run a forward pass over the CHAINED dataset and extract per-step token-level
hidden states from a specified transformer layer.

For each entry the script:
  1. Reconstructs the original prompt + response text via the model's chat template.
  2. Tokenizes the full text and runs a forward pass with output_hidden_states=True.
  3. Locates each reasoning step's character span inside the response, maps it to
     token indices, and stores the raw hidden states for those tokens.
  4. Writes the result as JSONL (one CoT entry per line) for downstream feature
     extraction (extract_features.py).

The output field added to every step is:
    "token_hidden_states": List[List[float]]  # shape [num_tokens, hidden_dim]

Usage:
    python process.py \\
        --input   dataset/data/train/Llama_train.json \\
        --output  processed/Llama_train.jsonl \\
        --model   meta-llama/Llama-3.1-8B-Instruct \\
        --layer   16 \\
        --device  cuda:0 \\
        --batch-size 8
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt construction  (must match generate.py exactly so the response text
# can be located inside the tokenized full sequence)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are an expert in reading comprehension, multi-step reasoning."
    "Your job is to produce an **extremely detailed, rigorous, step-by-step Chain-of-Thought (CoT)** to solve complex questions."
    "Your reasoning must consist of **many short, atomic steps** that **incrementally transform facts from the context into the final answer**."
    "The steps must be **logically connected, non-redundant, and explicit about evidence use**."
    "**Crucially, your thinking process must not contain repeated, cyclical, or excessively similar steps.**"
    "Always conclude with 'Final Answer: ' followed by your final, clear answer."
)


def _build_user_content(example: dict[str, Any]) -> str:
    query = example.get("query", "")
    context = example.get("context", "").strip()
    if not context:
        return query
    return (
        "TASK: Read the context carefully and answer the question based on the information provided.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"{query}\n\n"
        "INSTRUCTIONS:\n"
        "1. Produce a numbered list of reasoning steps (1., 2., 3., ...).\n"
        "2. Start by detailing your step-by-step reasoning (Chain-of-Thought). "
        "**Your goal is to generate as many unique, non-repetitive steps as possible "
        "to fully decompose the problem**\n"
        "3. Each step should be one short sentence performing ONE micro-inference (atomic).\n"
        "4. Cite or paraphrase concrete evidence from the context as you go; avoid vague phrasing.\n"
        "5. Do not skip intermediate hops; unfold implicit links explicitly.\n"
        "6. Ensure every step is unique and directly advances the solution. "
        "**If you find yourself stuck in a loop or repeating steps, STOP the reasoning "
        "immediately and proceed directly to the final answer.**\n"
        "7. Conclude your entire response with the final answer, and it must be prefixed "
        "by the exact string: 'Final Answer: '.\n"
        "8. STOP GENERATING any further text or reasoning immediately after the 'Final Answer:' line."
    )


def _build_full_text(example: dict[str, Any], tokenizer) -> tuple[str, int]:
    """
    Apply the chat template to reconstruct [prompt + response] as a single string.
    Returns (full_text, response_start_char_idx).
    """
    model_response = example["response"]
    messages_full = [
        {"role": "system",    "content": _SYSTEM_PROMPT},
        {"role": "user",      "content": _build_user_content(example)},
        {"role": "assistant", "content": model_response},
    ]
    full_text = tokenizer.apply_chat_template(messages_full, tokenize=False)

    # Locate where the response starts (DeepSeek templates sometimes drop it)
    response_start = full_text.find(model_response)
    if response_start == -1:
        # Fallback: manually append response after prompt
        messages_prompt = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": _build_user_content(example)},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages_prompt, tokenize=False, add_generation_prompt=True
        )
        full_text = prompt_text + model_response
        response_start = len(prompt_text)

    return full_text, response_start


# ---------------------------------------------------------------------------
# Token-level hidden state extraction
# ---------------------------------------------------------------------------

def _extract_hidden_states(
    model,
    tokenizer,
    full_text: str,
    target_layer: int,
    device: str,
    max_length: int = 4096,
) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    """
    Tokenize full_text, run one forward pass, return:
      hidden_states : [seq_len, hidden_dim]  (target layer, on CPU)
      offset_mapping: list of (char_start, char_end) per token
    """
    encoding = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors="pt",
        return_offsets_mapping=True,
    )
    input_ids     = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    offset_mapping = encoding["offset_mapping"].squeeze(0).tolist()

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

    hidden_states = outputs.hidden_states[target_layer].squeeze(0).cpu()
    return hidden_states, offset_mapping


def _map_steps_to_hidden_states(
    steps: list[dict],
    full_text: str,
    hidden_states: torch.Tensor,
    offset_mapping: list[tuple[int, int]],
    response_start: int,
) -> list[dict]:
    """
    For each step, find its character span in full_text using forward search,
    map to token indices, and attach token_hidden_states.
    Steps that cannot be located are dropped.
    """
    updated = []
    cur_char = max(0, response_start)

    for step in steps:
        text = step.get("text", "")

        # Forward search with strip fallback
        idx = full_text.find(text, cur_char)
        if idx == -1:
            stripped = text.strip()
            if stripped:
                idx = full_text.find(stripped, cur_char)
                if idx != -1:
                    text = stripped
        if idx == -1:
            continue

        start_char = idx
        end_char   = start_char + len(text)

        token_indices = [
            i for i, (s, e) in enumerate(offset_mapping)
            if s >= start_char and e <= end_char
        ]
        if not token_indices:
            cur_char = end_char
            continue

        start_tok = token_indices[0]
        end_tok   = min(token_indices[-1] + 1, hidden_states.shape[0])

        step_hidden = hidden_states[start_tok:end_tok]  # [L, H]

        new_step = dict(step)
        new_step["token_hidden_states"] = step_hidden.tolist()
        updated.append(new_step)
        cur_char = end_char

    return updated


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def process_batch(
    batch: list[dict[str, Any]],
    model,
    tokenizer,
    target_layer: int,
    device: str,
    max_length: int,
) -> list[dict[str, Any]]:
    results = []
    for example in batch:
        try:
            full_text, response_start = _build_full_text(example, tokenizer)
            hidden_states, offset_mapping = _extract_hidden_states(
                model, tokenizer, full_text, target_layer, device, max_length
            )
            updated_steps = _map_steps_to_hidden_states(
                example.get("steps", []),
                full_text,
                hidden_states,
                offset_mapping,
                response_start,
            )
            if not updated_steps:
                continue

            out = {k: v for k, v in example.items() if k != "steps"}
            out["steps"] = updated_steps
            results.append(out)
        except Exception as e:
            logger.warning(f"Failed id={example.get('id')}: {e}")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract per-step token hidden states from a transformer layer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input",      required=True, help="Input JSON file (CHAINED dataset split)")
    parser.add_argument("--output",     required=True, help="Output JSONL file")
    parser.add_argument("--model",      required=True, help="HuggingFace model path or repo ID")
    parser.add_argument("--layer",      type=int, required=True,
                        help="Layer index to extract hidden states from (0-indexed)")
    parser.add_argument("--device",     default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Process only the first N samples (for testing)")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # File log
    log_path = output_path.with_suffix(".log")
    logger.addHandler(logging.FileHandler(log_path, encoding="utf-8"))

    # Load data
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input not found: {input_path}")
        sys.exit(1)
    with open(input_path, encoding="utf-8") as f:
        data: list[dict] = json.load(f)
    if args.num_samples:
        data = data[: args.num_samples]
    logger.info(f"Loaded {len(data)} examples from {input_path.name}")

    # Load model
    logger.info(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=args.device
    )
    model.eval()
    model.config.use_cache = False

    num_layers = model.config.num_hidden_layers
    if not (0 <= args.layer <= num_layers):
        logger.error(f"--layer {args.layer} out of range [0, {num_layers}]")
        sys.exit(1)
    logger.info(f"Extracting layer {args.layer} / {num_layers}  |  batch={args.batch_size}")

    # Process
    t0 = time.time()
    total_written = 0
    num_batches = (len(data) + args.batch_size - 1) // args.batch_size

    with open(output_path, "w", encoding="utf-8") as out_f:
        for b in tqdm(range(num_batches), desc="Batches"):
            batch = data[b * args.batch_size : (b + 1) * args.batch_size]
            results = process_batch(batch, model, tokenizer, args.layer, args.device, args.max_length)
            for entry in results:
                out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            total_written += len(results)

    elapsed = time.time() - t0
    logger.info(f"Done. {total_written}/{len(data)} examples written to {output_path}")
    logger.info(f"Elapsed: {elapsed:.0f}s  ({total_written / elapsed:.2f} ex/s)")
    logger.info(f"Log: {log_path}")


if __name__ == "__main__":
    main()
