"""
evaluate.py
===========
Runs Task A and Task B prompts against the Groq API (LLaMA 3)
and saves results to results/ folder.

Requirements:
    pip install groq

Usage:
    1. Set your Groq API key:
       export GROQ_API_KEY="your_key_here"
       OR edit GROQ_API_KEY variable below.

    2. Run:
       python evaluate.py

    Options:
       --task A          # only run Task A
       --task B          # only run Task B
       --lang en         # only run English
       --variant bare    # only run bare variant
       --format words    # only run words format
       --split test      # which split to evaluate (default: test)
       --dry-run         # print prompts without calling API
"""

import os
import sys
import json
import time
import argparse
import random
from pathlib import Path

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# ─────────────────────────────────────────────
# CONFIG — edit your key here if not using env var
# ─────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "YOUR_GROQ_API_KEY_HERE")
MODEL = "llama-3.1-8b-instant"       # free, fast, multilingual
# Alternatives: "llama-3.1-70b-versatile", "mixtral-8x7b-32768", "gemma2-9b-it"

DATASETS_DIR = Path("datasets")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

RATE_LIMIT_DELAY = 0.5   # seconds between requests (be nice to free tier)
MAX_RETRIES = 3

LANGUAGES = ["en", "fr", "da", "ja"]
VARIANTS = ["bare", "sentence"]
FORMATS = ["words", "digits", "mixed"]


# ─────────────────────────────────────────────
# PROMPT BUILDERS
# ─────────────────────────────────────────────

TASK_A_SYSTEM = """You are a linguistic expert. Your job is to judge whether a number expression is grammatically valid in the given language.
Respond with ONLY one word: "grammatical" or "ungrammatical". No explanation."""

TASK_B_SYSTEM = """You are a mathematics expert. Your job is to compare two quantities and determine which is larger.
Respond with ONLY the letter "A" or "B" to indicate which quantity is larger. No explanation."""


def build_task_a_prompt(record: dict, fmt: str, lang: str) -> str:
    """Build prompt for Task A (grammaticality judgment)."""
    lang_name = {
        "en": "English", "fr": "French", "da": "Danish", "ja": "Japanese"
    }[lang]

    content = record["content_words"] if fmt == "words" else record["content_digits"]

    return f"""Language: {lang_name}
Number expression: "{content}"

Is this number expression grammatically valid in {lang_name}?
Answer with ONLY one word: grammatical or ungrammatical."""


def build_task_b_prompt(record: dict, fmt: str) -> str:
    """Build prompt for Task B (value comparison)."""
    if fmt == "words":
        a_val = record["a_words"]
        b_val = record["b_words"]
    elif fmt == "digits":
        a_val = record["a_digits"]
        b_val = record["b_digits"]
    else:  # mixed: A=digits, B=words
        a_val = record["a_mixed"]
        b_val = record["b_mixed"]

    return f"""Which quantity is larger?

A: {a_val}
B: {b_val}

Answer with ONLY the letter A or B."""


# ─────────────────────────────────────────────
# API CALL
# ─────────────────────────────────────────────

def call_groq(client, system_prompt: str, user_prompt: str, dry_run: bool = False) -> str:
    """Call Groq API and return the model's response text."""
    if dry_run:
        print("\n--- SYSTEM ---")
        print(system_prompt)
        print("--- USER ---")
        print(user_prompt)
        print("--- END ---\n")
        return "grammatical"  # dummy

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=10,
                temperature=0.0,   # deterministic
            )
            return response.choices[0].message.content.strip().lower()
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait = 2 ** attempt
                print(f"  [Retry {attempt+1}] Error: {e}. Waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  [FAILED] {e}")
                return "ERROR"


# ─────────────────────────────────────────────
# RESPONSE PARSING
# ─────────────────────────────────────────────

def parse_task_a_response(response: str) -> str:
    """Parse model response to 'grammatical' or 'ungrammatical'."""
    r = response.lower().strip()
    if "ungrammatical" in r:
        return "ungrammatical"
    if "grammatical" in r:
        return "grammatical"
    return "invalid"


def parse_task_b_response(response: str) -> str:
    """Parse model response to 'A' or 'B'."""
    r = response.upper().strip()
    if r.startswith("A"):
        return "A"
    if r.startswith("B"):
        return "B"
    return "invalid"


# ─────────────────────────────────────────────
# EVALUATION LOOP
# ─────────────────────────────────────────────

def evaluate_task_a(client, lang: str, variant: str, fmt: str, split: str, dry_run: bool):
    """Evaluate Task A for one condition."""
    data_path = DATASETS_DIR / f"task_a_{lang}_{variant}_{split}.jsonl"
    if not data_path.exists():
        print(f"  [SKIP] {data_path} not found.")
        return

    result_path = RESULTS_DIR / f"task_a_{lang}_{variant}_{fmt}_{split}.jsonl"
    if result_path.exists():
        print(f"  [SKIP] Already evaluated: {result_path}")
        return

    print(f"  Evaluating Task A | lang={lang} | variant={variant} | format={fmt} | split={split}")

    records = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    results = []
    correct = 0

    for i, rec in enumerate(records):
        user_prompt = build_task_a_prompt(rec, fmt, lang)
        raw = call_groq(client, TASK_A_SYSTEM, user_prompt, dry_run)
        pred = parse_task_a_response(raw)

        gold_text = rec["label_text"]   # "grammatical" or "ungrammatical"
        is_correct = (pred == gold_text)
        if is_correct:
            correct += 1

        result_rec = {
            **rec,
            "format": fmt,
            "prediction": pred,
            "raw_response": raw,
            "correct": is_correct,
        }
        results.append(result_rec)

        if (i + 1) % 50 == 0:
            acc = correct / (i + 1)
            print(f"    [{i+1}/{len(records)}] Running accuracy: {acc:.3f}")

        time.sleep(RATE_LIMIT_DELAY)

    # Save results
    with open(result_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    final_acc = correct / len(records) if records else 0
    print(f"  -> Accuracy: {final_acc:.3f} ({correct}/{len(records)})")
    return final_acc


def evaluate_task_b(client, lang: str, variant: str, fmt: str, split: str, dry_run: bool):
    """Evaluate Task B for one condition."""
    data_path = DATASETS_DIR / f"task_b_{lang}_{variant}_{split}.jsonl"
    if not data_path.exists():
        print(f"  [SKIP] {data_path} not found.")
        return

    result_path = RESULTS_DIR / f"task_b_{lang}_{variant}_{fmt}_{split}.jsonl"
    if result_path.exists():
        print(f"  [SKIP] Already evaluated: {result_path}")
        return

    print(f"  Evaluating Task B | lang={lang} | variant={variant} | format={fmt} | split={split}")

    records = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    results = []
    correct = 0

    for i, rec in enumerate(records):
        user_prompt = build_task_b_prompt(rec, fmt)
        raw = call_groq(client, TASK_B_SYSTEM, user_prompt, dry_run)
        pred = parse_task_b_response(raw)

        gold = rec["label_text"]   # "A" or "B"
        is_correct = (pred == gold)
        if is_correct:
            correct += 1

        result_rec = {
            **rec,
            "format": fmt,
            "prediction": pred,
            "raw_response": raw,
            "correct": is_correct,
        }
        results.append(result_rec)

        if (i + 1) % 50 == 0:
            acc = correct / (i + 1)
            print(f"    [{i+1}/{len(records)}] Running accuracy: {acc:.3f}")

        time.sleep(RATE_LIMIT_DELAY)

    # Save results
    with open(result_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    final_acc = correct / len(records) if records else 0
    print(f"  -> Accuracy: {final_acc:.3f} ({correct}/{len(records)})")
    return final_acc


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate numerical probing tasks via Groq API")
    parser.add_argument("--task", choices=["A", "B", "both"], default="both")
    parser.add_argument("--lang", choices=LANGUAGES + ["all"], default="all")
    parser.add_argument("--variant", choices=VARIANTS + ["all"], default="all")
    parser.add_argument("--format", choices=FORMATS + ["all"], default="all")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--dry-run", action="store_true", help="Print prompts without calling API")
    args = parser.parse_args()

    # Validate API key
    if not args.dry_run:
        if GROQ_API_KEY == "YOUR_GROQ_API_KEY_HERE":
            print("[ERROR] Please set your GROQ_API_KEY.")
            print("        export GROQ_API_KEY='your_key_here'")
            print("        OR edit the GROQ_API_KEY variable in evaluate.py")
            sys.exit(1)
        if not GROQ_AVAILABLE:
            print("[ERROR] groq package not installed. Run: pip install groq")
            sys.exit(1)

    client = None
    if not args.dry_run and GROQ_AVAILABLE:
        client = Groq(api_key=GROQ_API_KEY)
        print(f"Using model: {MODEL}")

    langs = LANGUAGES if args.lang == "all" else [args.lang]
    variants = VARIANTS if args.variant == "all" else [args.variant]
    formats = FORMATS if args.format == "all" else [args.format]
    tasks = ["A", "B"] if args.task == "both" else [args.task]

    print("=" * 60)
    print("Starting Evaluation")
    print(f"  Tasks:    {tasks}")
    print(f"  Langs:    {langs}")
    print(f"  Variants: {variants}")
    print(f"  Formats:  {formats}")
    print(f"  Split:    {args.split}")
    print("=" * 60)

    for task in tasks:
        print(f"\n{'='*20} TASK {task} {'='*20}")
        for lang in langs:
            for variant in variants:
                for fmt in formats:
                    if task == "A":
                        evaluate_task_a(client, lang, variant, fmt, args.split, args.dry_run)
                    else:
                        evaluate_task_b(client, lang, variant, fmt, args.split, args.dry_run)

    print("\n" + "=" * 60)
    print("Evaluation complete! Results saved to ./results/")
    print("Run python analyze.py to generate accuracy tables and plots.")
    print("=" * 60)


if __name__ == "__main__":
    main()
