"""
Query Llama-3.1-8B-Instruct on the Task A / Task B test sets and save
all responses to results/.

Provider note: early runs used Groq (llama-3.1-8b-instant), which
deprecated the model on 2026-08-16. Remaining runs used DeepInfra, which
serves the same Meta-Llama-3.1-8B-Instruct weights. This script supports
DeepInfra only. Each result record stores model, provider and timestamp.

Usage:
    export DEEPINFRA_API_KEY="your_key_here"   # or set it in .env
    python evaluate.py                         # run all conditions
    python evaluate.py --task B --lang en --variant bare --format words
    python evaluate.py --dry-run --task B --lang en   # print prompts only

Existing result files are skipped, so interrupted runs can be resumed.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # reads DEEPINFRA_API_KEY from .env if present

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
PROVIDER = "deepinfra"
BASE_URL = "https://api.deepinfra.com/v1/openai"
MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
TEMPERATURE = 0.0
MAX_TOKENS = 10

RATE_LIMIT_DELAY = 0.2  # seconds between requests (a polite pace, not a hard limit)
MAX_RETRIES = 3

DATASETS_DIR = Path("datasets")
RESULTS_DIR = Path("results")
PROMPTS_DIR = Path(__file__).parent / "prompts"

LANGUAGES = {"en": "English", "fr": "French", "da": "Danish", "ja": "Japanese"}
VARIANTS = ["bare", "sentence"]

# Task A has no separate "mixed" format: an expression is a single number,
# so "mixed" would be identical to "digits".
FORMATS = {"A": ["words", "digits"], "B": ["words", "digits", "mixed"]}


# ─────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────

def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8").rstrip("\n")


SYSTEM_PROMPTS = {"A": load_prompt("task_a_system.txt"), "B": load_prompt("task_b_system.txt")}
USER_TEMPLATES = {"A": load_prompt("task_a_user.txt"), "B": load_prompt("task_b_user.txt")}


def build_prompt(task: str, record: dict, fmt: str) -> str:
    """Fill the user prompt template for one record and format."""
    if task == "A":
        content = record["content_words"] if fmt == "words" else record["content_digits"]
        return USER_TEMPLATES["A"].format(lang_name=LANGUAGES[record["lang"]], content=content)
    # Task B: pick the A/B fields for this format (words, digits or mixed)
    return USER_TEMPLATES["B"].format(a=record[f"a_{fmt}"], b=record[f"b_{fmt}"])


# ─────────────────────────────────────────────
# Model call and parsing
# ─────────────────────────────────────────────

def call_llm(client, system_prompt: str, user_prompt: str) -> str:
    """Return the model's reply in lower case, or "ERROR" after repeated failures."""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )
            return response.choices[0].message.content.strip().lower()
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait = 2 ** attempt
                print(f"    [retry {attempt + 1}] {e}. Waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"    [failed] {e}")
                return "ERROR"


def parse_response(task: str, response: str) -> str:
    """Map a raw reply to a label, or "invalid" if it doesn't match."""
    r = response.strip().lower()
    if task == "A":
        if "ungrammatical" in r:
            return "ungrammatical"
        if "grammatical" in r:
            return "grammatical"
    else:
        if r.startswith("a"):
            return "A"
        if r.startswith("b"):
            return "B"
    return "invalid"


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────

def load_jsonl(path: Path) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def evaluate_condition(client, task, lang, variant, fmt, split, dry_run):
    """Run one task x language x variant x format condition."""
    t = task.lower()
    data_path = DATASETS_DIR / f"task_{t}_{lang}_{variant}_{split}.jsonl"
    result_path = RESULTS_DIR / f"task_{t}_{lang}_{variant}_{fmt}_{split}.jsonl"

    if not data_path.exists():
        print(f"  [skip] {data_path} not found (run generate_dataset.py)")
        return
    if result_path.exists() and not dry_run:
        print(f"  [skip] already evaluated: {result_path}")
        return

    print(f"  Task {task} | {lang} | {variant} | {fmt}")
    records = load_jsonl(data_path)
    system_prompt = SYSTEM_PROMPTS[task]

    if dry_run:
        print(f"--- SYSTEM ---\n{system_prompt}\n--- USER ---\n{build_prompt(task, records[0], fmt)}\n")
        return

    results, correct = [], 0
    for i, rec in enumerate(records, start=1):
        raw = call_llm(client, system_prompt, build_prompt(task, rec, fmt))
        pred = parse_response(task, raw)
        is_correct = pred == rec["label_text"]
        correct += is_correct

        results.append({
            **rec,
            "format": fmt,
            "prediction": pred,
            "raw_response": raw,
            "correct": is_correct,
            "model": MODEL,
            "provider": PROVIDER,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })

        if i % 50 == 0:
            print(f"    [{i}/{len(records)}] running accuracy: {correct / i:.3f}")
        time.sleep(RATE_LIMIT_DELAY)

    # Written only once the whole condition is done, so a crash never leaves
    # a half-finished file that would later be skipped as "already evaluated".
    with open(result_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  -> accuracy {correct / len(records):.3f} ({correct}/{len(records)})")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate the numerical probing tasks.")
    parser.add_argument("--task", choices=["A", "B", "both"], default="both")
    parser.add_argument("--lang", choices=list(LANGUAGES) + ["all"], default="all")
    parser.add_argument("--variant", choices=VARIANTS + ["all"], default="all")
    parser.add_argument("--format", choices=["words", "digits", "mixed", "all"], default="all")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--dry-run", action="store_true",
                        help="print one example prompt per condition, no API calls")
    args = parser.parse_args()

    client = None
    if not args.dry_run:
        api_key = os.environ.get("DEEPINFRA_API_KEY")
        if not api_key:
            sys.exit("DEEPINFRA_API_KEY is not set. Add it to .env or run:\n"
                     "  export DEEPINFRA_API_KEY='your_key_here'")
        client = OpenAI(api_key=api_key, base_url=BASE_URL)
        RESULTS_DIR.mkdir(exist_ok=True)
        print(f"Model: {MODEL} via {PROVIDER}")

    tasks = ["A", "B"] if args.task == "both" else [args.task]
    langs = list(LANGUAGES) if args.lang == "all" else [args.lang]
    variants = VARIANTS if args.variant == "all" else [args.variant]

    for task in tasks:
        formats = FORMATS[task] if args.format == "all" else [args.format]
        formats = [f for f in formats if f in FORMATS[task]]  # e.g. no "mixed" for Task A
        print(f"\n===== Task {task} =====")
        for lang in langs:
            for variant in variants:
                for fmt in formats:
                    evaluate_condition(client, task, lang, variant, fmt, args.split, args.dry_run)

    print("\nDone. Next: python analyze.py")


if __name__ == "__main__":
    main()