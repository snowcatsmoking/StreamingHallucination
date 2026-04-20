"""
Annotate hallucination labels for each reasoning step in CHAINED dataset.

For every entry in the input file the script:
  1. Calls a judge LLM to determine whether the model's final answer is
     semantically correct (falls back to exact-string match on failure).
  2. Calls a tagger LLM (can be the same model) to assign per-step
     step_hallucination and cumulative_hallucination boolean labels,
     given the known answer correctness as a hint.
  3. Writes tagged results incrementally to the output file so that a
     crashed run can be restarted with --start / --end.

The tagger and judge are accessed via an OpenAI-compatible chat completions
endpoint, so any provider (OpenAI, vLLM, Together, etc.) works.

Usage:
    python tag.py \\
        --input   CHAINED_raw.json \\
        --output  CHAINED_raw_tags.json \\
        --api-key sk-... \\
        --api-url https://api.openai.com/v1/chat/completions \\
        --model   gpt-4o \\
        --start   0 \\
        --end     3000

    # Use a different model for the semantic judge:
    python tag.py ... --judge-model gpt-3.5-turbo
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Logging — set up after we know the output path; see main()
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts and schema
# ---------------------------------------------------------------------------

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "integer"},
        "steps_with_tags": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "step_id": {"type": "integer"},
                    "step_hallucination": {"type": "boolean"},
                    "cumulative_hallucination": {"type": "boolean"},
                },
                "required": ["step_id", "step_hallucination", "cumulative_hallucination"],
            },
        },
    },
    "required": ["id", "steps_with_tags"],
}

SYSTEM_PROMPT = """
ROLE: Expert Hallucination Annotator for Step-by-Step Reasoning.

TASK: Evaluate each step in the reasoning chain with full knowledge of whether the final answer is correct.

You will receive:
- A query/question
- Context (for MuSiQue subset only)
- A sequence of reasoning steps
- Model's final answer
- Ground truth answer
- Knowledge of whether the final answer is CORRECT or INCORRECT

--- MANDATORY OUTPUT CONSTRAINTS ---
1. **Output Format:** Return ONLY a single JSON object conforming exactly to the defined 'responseSchema'.
2. **Exclusions:** DO NOT include any explanatory text, markdown, or conversation. DO NOT output the Schema definition itself.
3. **Tagging:** Every step MUST receive both 'step_hallucination' and 'cumulative_hallucination' fields.
4. **Values:** Tags MUST be strict Boolean values (true or false), not strings.

--- HALLUCINATION DEFINITIONS ---

1. **step_hallucination (Local Factual Error):**
   Set to 'true' if the current step IN ISOLATION contains factual errors, misinformation, unsupported claims, fabricated information, or logically invalid inferences.
   Set to 'false' if the step is factually correct and logically sound on its own.

2. **cumulative_hallucination (Deviation from Ground Truth):**
   Set to 'true' if the accumulated reasoning so far deviates from the ground truth direction and would lead to an incorrect answer.
   Set to 'false' if the reasoning is aligned with the ground truth and on the correct path.

--- EVALUATION STRATEGY ---

You know whether the final answer is CORRECT or INCORRECT. Use this to guide your evaluation:

**If answer is INCORRECT:**
- Identify where reasoning started to deviate from ground truth
- Determine which step errors contributed to the wrong answer
- Mark cumulative_hallucination=true where deviation occurs
- The last step MUST have cumulative_hallucination=true (since it leads to wrong answer)

**If answer is CORRECT:**
- The reasoning chain successfully reached the correct answer
- This means the accumulated reasoning is aligned with the ground truth
- The last step SHOULD have cumulative_hallucination=false (aligned with ground truth)
- If there were errors along the way, they either:
  * Were corrected/recovered (cum can go from true back to false)
  * Were minor and didn't prevent reaching the correct conclusion
- Evaluate realistically: if the chain reached the correct answer, it was fundamentally on the right path

**Key Principles:**
- Final answer correctness is strong evidence of reasoning alignment
- cumulative_hallucination measures "on track vs off track" - correct answer means on track
- Evaluate step_hallucination and cumulative_hallucination independently
- Allow for recovery patterns (cum: true→false) when reasoning self-corrects
- Trust the final outcome: correct answer = reasoning ultimately aligned with ground truth
"""


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _openai_headers(api_key: str) -> dict:
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def clean_and_parse_json(text: str) -> dict:
    """Extract the first valid JSON object that contains 'steps_with_tags'."""
    text = text.strip()
    candidates = []

    # Try sequential raw_decode to handle "schema + result" concatenations
    try:
        decoder = json.JSONDecoder()
        pos = 0
        while pos < len(text):
            while pos < len(text) and text[pos].isspace():
                pos += 1
            if pos >= len(text):
                break
            obj, end = decoder.raw_decode(text[pos:])
            candidates.append(obj)
            pos += end
    except Exception:
        pass

    # Fallback: markdown code blocks
    if not candidates:
        for block in re.findall(r"`{3}(?:json)?\s*(.*?)\s*`{3}", text, re.DOTALL):
            try:
                candidates.append(json.loads(block))
            except Exception:
                pass

    # Fallback: outermost braces
    if not candidates:
        try:
            s, e = text.find("{"), text.rfind("}")
            if s != -1 and e != -1:
                candidates.append(json.loads(text[s : e + 1]))
        except Exception:
            pass

    # Pick the best candidate: prefer one with 'steps_with_tags', skip schema defs
    for obj in candidates:
        if not isinstance(obj, dict):
            continue
        if obj.get("type") == "object" and "properties" in obj:
            continue
        if "steps_with_tags" in obj:
            return obj

    # Last resort: any non-schema dict
    for obj in reversed(candidates):
        if isinstance(obj, dict) and not (obj.get("type") == "object" and "properties" in obj):
            return obj

    raise ValueError(f"No valid JSON found in response. Preview: {text[:200]}")


def _chat(
    messages: list[dict],
    model: str,
    api_url: str,
    api_key: str,
    max_tokens: int,
    temperature: float,
    max_retries: int,
    label: str,
) -> str | None:
    """POST to an OpenAI-compatible chat completions endpoint with retries."""
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    headers = _openai_headers(api_key)

    for attempt in range(max_retries):
        try:
            resp = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=60)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            if content:
                return content
            logger.error(f"{label} attempt {attempt + 1}: empty content")
        except Exception as e:
            logger.error(f"{label} attempt {attempt + 1} failed: {e}")

        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)

    return None


# ---------------------------------------------------------------------------
# Semantic judge
# ---------------------------------------------------------------------------

def judge_correctness(
    answer: str,
    ground_truth: str,
    query: str,
    api_url: str,
    api_key: str,
    judge_model: str,
    max_retries: int,
) -> tuple[bool, bool]:
    """
    Returns (is_correct, judge_failed).
    Falls back to exact-string match when the API call fails.
    """
    prompt = (
        "You are an answer correctness judge. Compare the model's answer with the "
        "ground truth and determine if they are semantically equivalent.\n\n"
        f"QUERY: {query}\n"
        f"MODEL ANSWER: {answer}\n"
        f"GROUND TRUTH: {ground_truth}\n\n"
        "Rules:\n"
        "1. Focus on semantic equivalence, not exact string matching\n"
        "2. Ignore formatting differences (spaces, punctuation, capitalisation)\n"
        "3. For multiple choice: (A), A, 'The answer is A' are all equivalent\n"
        "4. For numeric: 5, 5.0, 'five' are all equivalent\n"
        "5. Synonyms and paraphrases that convey the same meaning are equivalent\n\n"
        'Respond with ONLY: {"is_correct": true}  or  {"is_correct": false}'
    )

    content = _chat(
        messages=[
            {"role": "system", "content": "You are an answer correctness judge."},
            {"role": "user", "content": prompt},
        ],
        model=judge_model,
        api_url=api_url,
        api_key=api_key,
        max_tokens=100,
        temperature=0.0,
        max_retries=max_retries,
        label="JUDGE",
    )

    if content:
        try:
            s, e = content.index("{"), content.rindex("}") + 1
            return bool(json.loads(content[s:e]).get("is_correct", False)), False
        except Exception:
            pass

    logger.warning("Semantic judge failed, falling back to exact-string match")
    return (answer.strip() == ground_truth.strip()), True


# ---------------------------------------------------------------------------
# Tagger
# ---------------------------------------------------------------------------

def _build_tagging_prompt(entry: dict, is_correct: bool) -> str:
    parts = [
        f"DATA ENTRY ID: {entry.get('id', 'N/A')}",
        f"SUBSET: {entry.get('subset', '')}",
        f"\nQUERY: {entry.get('query', 'N/A')}",
    ]

    if entry.get("subset") == "MuSiQue" and entry.get("context"):
        parts.append(f"\nCONTEXT:\n{entry['context']}")

    steps = entry.get("steps", [])
    if steps:
        parts.append("\nREASONING STEPS:")
        for step in steps:
            parts.append(f"Step {step.get('step_id', '?')}: {step.get('text', '')}")

    parts += [
        f"\nMODEL'S FINAL ANSWER: {entry.get('answer', 'N/A')}",
        f"GROUND TRUTH ANSWER: {entry.get('ground_truth', 'N/A')}",
        f"\n**ANSWER CORRECTNESS: {'CORRECT' if is_correct else 'INCORRECT'}**",
        "\n**EVALUATION TASK:**",
    ]

    if is_correct:
        parts += [
            "The final answer is CORRECT - the reasoning successfully reached ground truth.",
            "- The LAST step should have cumulative_hallucination=false",
            "- Errors along the way were either recovered or minor",
            "- Correct final answer = reasoning was fundamentally on the right path",
        ]
    else:
        parts += [
            "The final answer is INCORRECT - the reasoning deviated from ground truth.",
            "- The LAST step MUST have cumulative_hallucination=true",
            "- Identify where the reasoning started to deviate",
        ]

    parts += [
        f"\n\nREQUIRED JSON SCHEMA:\n{json.dumps(RESPONSE_SCHEMA, indent=2)}",
        "\n***ANNOTATION INSTRUCTIONS:***",
        "1. Evaluate each step with full knowledge of the final outcome",
        "2. For step_hallucination: Is THIS step factually correct in isolation?",
        "3. For cumulative_hallucination: Did the reasoning SO FAR deviate from ground truth?",
        "4. Return ONLY the JSON object with 'id' and 'steps_with_tags'",
        "5. Use boolean values (true/false), not strings",
        "6. DO NOT repeat the schema definition.",
        "\nOUTPUT ONLY THE JSON OBJECT NOW:",
    ]

    return "\n".join(parts)


def tag_entry(
    entry: dict,
    api_url: str,
    api_key: str,
    tagger_model: str,
    judge_model: str,
    max_retries: int,
    tracker: "BatchTracker",
) -> dict:
    entry_id = entry.get("id", "N/A")
    t0 = time.time()

    is_correct, judge_failed = judge_correctness(
        answer=entry.get("answer", ""),
        ground_truth=entry.get("ground_truth", ""),
        query=entry.get("query", ""),
        api_url=api_url,
        api_key=api_key,
        judge_model=judge_model,
        max_retries=max_retries,
    )
    if judge_failed:
        tracker.judge_fallbacks.append(entry_id)
    logger.info(f"ID {entry_id}: answer correct={is_correct}")

    content = _chat(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_tagging_prompt(entry, is_correct)},
        ],
        model=tagger_model,
        api_url=api_url,
        api_key=api_key,
        max_tokens=4096,
        temperature=0.0,
        max_retries=max_retries,
        label=f"TAG id={entry_id}",
    )

    if not content:
        tracker.add_failure(entry_id, "API failed after max retries")
        return {"id": entry_id, "error": "Tagging failed"}

    try:
        result = clean_and_parse_json(content)
    except ValueError as e:
        tracker.add_failure(entry_id, str(e))
        return {"id": entry_id, "error": "Tagging failed"}

    if "steps_with_tags" not in result:
        tracker.add_failure(entry_id, "steps_with_tags missing from response")
        return {"id": entry_id, "error": "Tagging failed"}

    result["is_correct"] = is_correct

    # Quality flag: correct answer but last step has cumulative_hallucination=True
    last_cum = result["steps_with_tags"][-1].get("cumulative_hallucination") if result["steps_with_tags"] else None
    logic_violation = is_correct and last_cum
    if logic_violation:
        logger.warning(f"ID {entry_id}: logic violation - correct answer but last cum=True")

    tracker.add_success(entry_id, is_correct, time.time() - t0, logic_violation)
    logger.info(f"ID {entry_id}: tagged successfully")
    return result


# ---------------------------------------------------------------------------
# BatchTracker
# ---------------------------------------------------------------------------

class BatchTracker:
    def __init__(self):
        self.total = 0
        self.successes = 0
        self.failures = 0
        self.correct = 0
        self.incorrect = 0
        self.judge_fallbacks: list = []
        self.failed_ids: list = []
        self.failed_reasons: dict = {}
        self.logic_violations: list = []
        self.times: list[float] = []
        self._start = time.time()

    def add_success(self, entry_id, is_correct: bool, elapsed: float, logic_violation: bool):
        self.total += 1
        self.successes += 1
        self.times.append(elapsed)
        if is_correct:
            self.correct += 1
        else:
            self.incorrect += 1
        if logic_violation:
            self.logic_violations.append(entry_id)

    def add_failure(self, entry_id, reason: str):
        self.total += 1
        self.failures += 1
        self.failed_ids.append(entry_id)
        self.failed_reasons[entry_id] = reason
        logger.error(f"ID {entry_id}: tagging failed — {reason}")

    def print_summary(self):
        elapsed = time.time() - self._start
        avg = sum(self.times) / len(self.times) if self.times else 0
        rate = f"{self.successes / self.total * 100:.1f}%" if self.total else "N/A"
        logger.info("=" * 60)
        logger.info(f"Processed: {self.total}  Success: {self.successes}  Failed: {self.failures}  ({rate})")
        logger.info(f"Correct answers: {self.correct}  Incorrect: {self.incorrect}")
        logger.info(f"Judge fallbacks (string match): {len(self.judge_fallbacks)}")
        logger.info(f"Logic violations: {len(self.logic_violations)}")
        logger.info(f"Elapsed: {elapsed:.0f}s  Avg/sample: {avg:.1f}s")
        if self.failed_ids:
            logger.info(f"Failed IDs: {self.failed_ids}")
        if self.logic_violations:
            logger.info(f"Logic violation IDs: {self.logic_violations}")
        logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Tag hallucination labels for CHAINED dataset steps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input",       required=True, help="Input JSON (output of merge.py merge)")
    parser.add_argument("--output",      required=True, help="Output tag JSON (feed to merge.py inject)")
    parser.add_argument("--api-key",     default=os.environ.get("OPENAI_API_KEY", ""),
                        help="API key (default: $OPENAI_API_KEY)")
    parser.add_argument("--api-url",     required=True,
                        help="OpenAI-compatible chat completions endpoint URL")
    parser.add_argument("--model",       required=True, help="Tagger model name")
    parser.add_argument("--judge-model", default=None,
                        help="Judge model name (defaults to --model)")
    parser.add_argument("--start",       type=int, default=0,
                        help="Start index in the input array (default: 0)")
    parser.add_argument("--end",         type=int, default=None,
                        help="End index (exclusive). Defaults to end of file.")
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--save-every",  type=int, default=5,
                        help="Checkpoint every N samples (default: 5)")
    args = parser.parse_args()

    if not args.api_key:
        print("Error: --api-key is required (or set OPENAI_API_KEY)", file=sys.stderr)
        sys.exit(1)

    judge_model = args.judge_model or args.model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # File logging alongside console
    log_path = output_path.with_suffix(".log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )

    # Load input
    with open(args.input, encoding="utf-8") as f:
        all_data: list[dict] = json.load(f)
    end = args.end if args.end is not None else len(all_data)
    batch = all_data[args.start:end]
    logger.info(f"Loaded {len(all_data)} entries, processing [{args.start}, {end}) = {len(batch)} entries")
    logger.info(f"Tagger: {args.model}  Judge: {judge_model}")

    tracker = BatchTracker()
    results: list[dict] = []

    for i, entry in enumerate(batch):
        entry_id = entry.get("id", "N/A")
        logger.info(f"[{i + 1}/{len(batch)}] Processing id={entry_id}")

        try:
            result = tag_entry(
                entry=entry,
                api_url=args.api_url,
                api_key=args.api_key,
                tagger_model=args.model,
                judge_model=judge_model,
                max_retries=args.max_retries,
                tracker=tracker,
            )
        except Exception as e:
            logger.error(f"Unexpected error for id={entry_id}: {e}")
            tracker.add_failure(entry_id, str(e))
            result = {"id": entry_id, "error": str(e)}

        results.append(result)

        if (i + 1) % args.save_every == 0:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            logger.info(f"Checkpoint saved ({i + 1}/{len(batch)})")
            tracker.print_summary()

        time.sleep(0.5)

    # Final save
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    logger.info(f"Done. Results -> {output_path}")
    logger.info(f"Log -> {log_path}")
    tracker.print_summary()


if __name__ == "__main__":
    main()
