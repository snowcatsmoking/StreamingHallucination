"""
Generate CoT responses for CHAINED dataset construction.

Loads a source benchmark (MuSiQue or BBH), runs each question through a local
HuggingFace causal-LM model multiple times, parses the numbered reasoning steps,
and saves the results for downstream hallucination tagging.

Supports resuming interrupted runs: if the output file already exists, processing
continues from where it left off.

Usage:
    python generate.py \
        --input   /path/to/source.json \
        --output  /path/to/output.json \
        --model   /path/to/model_or_hf_id \
        --subset  MuSiQue \
        --runs    3 \
        --device  cuda:0
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a hyper-rigorous, expert logician and microscopic reasoner. "
    "Your task is to produce an **EXHAUSTIVE and HYPER-DETAILED Chain-of-Thought (CoT)**. "
    "**DECOMPOSE the problem into the SMALLEST POSSIBLE ATOMIC STEPS.** "
    "Your reasoning must consist of **MAXIMUM unique, non-redundant steps** that "
    "**incrementally transform facts from the context into the final answer**. "
    "Each step must be a single, short logical jump. Do not combine micro-inferences. "
    "Focus on UNPACKING every implicit connection and detail, making the reasoning "
    "process as long and granular as logically possible. "
    "Crucially, your thinking process must not contain repeated, cyclical, or "
    "excessively similar steps. "
    "Always conclude with 'Final Answer: ' followed by your final, clear answer."
)


def build_user_content(question: str, context: str | None) -> str:
    base = (
        "TASK: Solve the following complex riddle or question.\n\n"
        "INPUT QUESTION:\n"
        f"{question}\n\n"
    )
    if context:
        base = (
            "TASK: Read the context carefully and answer the question.\n\n"
            f"CONTEXT:\n{context}\n\n"
            "INPUT QUESTION:\n"
            f"{question}\n\n"
        )
    instructions = (
        "INSTRUCTIONS:\n"
        "1. Produce a numbered list of reasoning steps (1., 2., 3., ...).\n"
        "2. Start by detailing your step-by-step reasoning (Chain-of-Thought). "
        "**Your goal is to generate as many unique, non-repetitive steps as possible "
        "to fully decompose the problem**\n"
        "3. Each step should be ONE short sentence performing ONE single micro-inference "
        "(atomic). **For example, a typical human thought might take 3-5 of your atomic "
        "steps to fully explain.**\n"
        "4. **Deeply decompose the problem:** Assume the reader knows nothing, and "
        "explicitly state every piece of information used and every trivial logical "
        "deduction made.\n"
        "5. Cite or paraphrase concrete evidence from the context as you go; avoid "
        "vague phrasing.\n"
        "6. Do not skip intermediate hops; unfold implicit links explicitly.\n"
        "7. Ensure every step is unique and directly advances the solution. "
        "**If you find yourself stuck in a loop or repeating steps, STOP the reasoning "
        "immediately and proceed directly to the final answer.**\n"
        "8. Conclude your entire response with the final answer, and it must be prefixed "
        "by the exact string: 'Final Answer: '.\n"
        "9. STOP GENERATING any further text or reasoning immediately after the "
        "'Final Answer:' line."
    )
    return base + instructions


def extract_final_answer(text: str) -> str:
    prefix = "Final Answer: "
    try:
        idx = text.rindex(prefix)
        return text[idx + len(prefix):].strip()
    except ValueError:
        return ""


def parse_steps(text: str) -> list[dict]:
    steps = []
    step_id = 0
    for line in text.split("\n"):
        line = line.strip()
        if line and not line.startswith("Final Answer:") and len(line) > 5:
            steps.append({"step_id": step_id, "text": line})
            step_id += 1
    return steps


def load_model(model_path: str, device: str):
    logger.info(f"Loading model from: {model_path}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()
    logger.info("Model loaded.")
    return model, tokenizer


def generate_response(
    item: dict,
    model,
    tokenizer,
    device: str,
    max_new_tokens: int,
) -> str | None:
    question = item["question"].strip()
    context = item.get("context", "").strip() or None

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_content(question, context)},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    try:
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=1.2,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
        generated = output[0, input_ids.shape[-1]:]
        return tokenizer.decode(generated, skip_special_tokens=True).strip()
    except Exception as e:
        logger.error(f"Generation failed for id={item.get('id')}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Generate CoT responses for CHAINED dataset")
    parser.add_argument("--input",  required=True, help="Source benchmark JSON (questions + ground truth)")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--model",  required=True, help="HuggingFace model path or repo ID")
    parser.add_argument("--model-name", default=None,
                        help="Model name stored in the dataset (defaults to basename of --model)")
    parser.add_argument("--subset", required=True, choices=["MuSiQue", "bbh"],
                        help="Dataset subset name")
    parser.add_argument("--runs",   type=int, default=3,
                        help="Number of generation runs per question (default: 3)")
    parser.add_argument("--device", default="cuda:0",
                        help="Torch device string, e.g. cuda:0 or cpu (default: cuda:0)")
    parser.add_argument("--max-new-tokens", type=int, default=4096,
                        help="Max new tokens per generation (default: 4096)")
    parser.add_argument("--batch-size", type=int, default=5,
                        help="Save checkpoint every N entries (default: 5)")
    parser.add_argument("--log-interval", type=int, default=50,
                        help="Log average step count every N entries (default: 50)")
    args = parser.parse_args()

    model_name = args.model_name or Path(args.model).name
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Set up file logging alongside console logging
    log_file = output_path.with_suffix(".log")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(file_handler)

    # Load source data
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
    with open(input_path, encoding="utf-8") as f:
        source_data: list[dict] = json.load(f)
    logger.info(f"Loaded {len(source_data)} questions from {input_path}")

    # Resume from checkpoint if output already exists
    results: list[dict] = []
    num_processed = 0
    start_index = 0
    if output_path.exists():
        try:
            with open(output_path, encoding="utf-8") as f:
                results = json.load(f)
            num_processed = len(results)
            start_index = num_processed // args.runs
            logger.info(
                f"Resuming: {num_processed} entries already done, "
                f"continuing from question index {start_index}"
            )
        except (json.JSONDecodeError, IOError):
            logger.warning("Output file is empty or corrupted, starting fresh.")

    model, tokenizer = load_model(args.model, args.device)

    total_steps = sum(len(r["steps"]) for r in results)
    logger.info(f"Generating for {len(source_data) - start_index} remaining questions ...")

    for q_idx, item in enumerate(tqdm(source_data[start_index:], desc="Questions")):
        if "id" not in item or "question" not in item:
            logger.warning(f"Skipping invalid item at index {start_index + q_idx}")
            continue

        original_id = item["id"]
        question = item["question"].strip()
        ground_truth = item.get("answer", "")
        context = item.get("context", "").strip()

        for run in range(args.runs):
            response_text = generate_response(
                item, model, tokenizer, args.device, args.max_new_tokens
            )
            if not response_text:
                logger.warning(
                    f"Run {run + 1}/{args.runs} failed for original_id={original_id}, skipping."
                )
                continue

            steps = parse_steps(response_text)
            answer = extract_final_answer(response_text)
            num_processed += 1

            entry = {
                "id": num_processed,
                "original_id": original_id,
                "subset": args.subset,
                "model": model_name,
                "context": context,
                "query": question,
                "response": response_text,
                "steps": steps,
                "answer": answer,
                "ground_truth": ground_truth,
            }
            results.append(entry)
            total_steps += len(steps)

            if num_processed % args.log_interval == 0:
                avg = total_steps / num_processed
                logger.info(f"[{num_processed} entries] avg steps/entry: {avg:.2f}")

        if num_processed % args.batch_size == 0:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=4)

    # Final save
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    avg_final = total_steps / num_processed if num_processed else 0
    logger.info(f"Done. {num_processed} entries saved to {output_path}")
    logger.info(f"Final avg steps/entry: {avg_final:.2f}")
    logger.info(f"Log saved to {log_file}")


if __name__ == "__main__":
    main()
