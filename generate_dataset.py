"""
Generate the Task A (grammaticality) and Task B (value comparison) datasets.

Languages: English, French, Danish, Japanese.
Numbers: integers 1-999.

Writes train/val/test JSONL files to datasets/ for each task, language
and variant (bare number or in a sentence). Number format (words, digits
or mixed) is selected later in evaluate.py.

Output is reproducible for a fixed SEED and num2words version.

Usage:
    python generate_dataset.py
"""

import json
import random
import re
from pathlib import Path
from num2words import num2words

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SEED = 42
NUMBER_RANGE = (1, 999)
TRAIN_SIZE = 500
VAL_SIZE = 100
TEST_SIZE = 200  # only the test split is evaluated

LANGUAGES = {"en": "English", "fr": "French", "da": "Danish", "ja": "Japanese"}
VARIANTS = ["bare", "sentence"]

OUT_DIR = Path("datasets")
TEMPLATES_PATH = Path(__file__).parent / "templates" / "sentence_templates.json"


def load_templates() -> dict:
    """Sentence templates per language, each with a {number} placeholder."""
    with open(TEMPLATES_PATH, encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────
# Number words
# ─────────────────────────────────────────────


def number_to_words(n: int, lang: str) -> str:
    """Spell out an integer in the given language."""
    words = num2words(n, lang=lang)
    if lang == "da":
        words = fix_danish_sixty(words)
    return words


def fix_danish_sixty(words: str) -> str:
    """
    num2words (0.5.14) spells Danish 60 as 'treds' (and 65 as 'femogtreds')
    instead of the correct 'tres' / 'femogtres'. 'halvtreds' (50) and its
    compounds are already correct, so they are left untouched.
    """
    return re.sub(r"(?<!halv)treds", "tres", words)


def make_ungrammatical(n1: int, n2: int, lang: str, rng: random.Random) -> str:
    """Concatenate two valid number words in random order, e.g. 'fifty-five two hundred'."""
    w1, w2 = number_to_words(n1, lang), number_to_words(n2, lang)
    return rng.choice([f"{w1} {w2}", f"{w2} {w1}"])


# ─────────────────────────────────────────────
# Dataset builders
# ─────────────────────────────────────────────


def build_task_a(lang, variant, size, templates, rng, concat_rng) -> list:
    """Task A: is the number expression grammatical?"""
    lo, hi = NUMBER_RANGE
    records, seen, attempts = [], set(), 0

    while len(records) < size and attempts < size * 10:
        attempts += 1

        if rng.random() < 0.5:  # grammatical
            n = rng.randint(lo, hi)
            if ("gram", n) in seen:
                continue
            seen.add(("gram", n))
            words, digits = number_to_words(n, lang), str(n)
            extra = {"label": 1, "label_text": "grammatical", "number": n}

        else:  # ungrammatical
            n1, n2 = rng.randint(lo, hi), rng.randint(lo, hi)
            key = ("ungram", min(n1, n2), max(n1, n2))
            if n1 == n2 or key in seen:
                continue
            seen.add(key)
            words = make_ungrammatical(n1, n2, lang, concat_rng)
            digits = f"{n1} {n2}"
            extra = {
                "label": 0,
                "label_text": "ungrammatical",
                "number": None,
                "number_parts": [n1, n2],
            }

        if variant == "sentence":
            tmpl = rng.choice(templates)
            words = tmpl.replace("{number}", words)
            digits = tmpl.replace("{number}", digits)

        records.append(
            {
                "task": "A",
                "lang": lang,
                "variant": variant,
                **extra,
                "content_words": words,
                "content_digits": digits,
            }
        )

    rng.shuffle(records)
    return records


def build_task_b(lang, variant, size, templates, rng) -> list:
    """Task B: which of two quantities (A or B) is larger?"""
    lo, hi = NUMBER_RANGE
    records, seen, attempts = [], set(), 0

    while len(records) < size and attempts < size * 10:
        attempts += 1
        n1, n2 = rng.randint(lo, hi), rng.randint(lo, hi)
        key = (min(n1, n2), max(n1, n2))
        if n1 == n2 or key in seen:
            continue
        seen.add(key)

        a_words, b_words = number_to_words(n1, lang), number_to_words(n2, lang)
        a_digits, b_digits = str(n1), str(n2)

        if variant == "sentence":
            t1, t2 = rng.choice(templates), rng.choice(templates)
            a_words, b_words = t1.replace("{number}", a_words), t2.replace(
                "{number}", b_words
            )
            a_digits, b_digits = t1.replace("{number}", a_digits), t2.replace(
                "{number}", b_digits
            )

        records.append(
            {
                "task": "B",
                "lang": lang,
                "variant": variant,
                "n1": n1,
                "n2": n2,
                "label": 0 if n1 > n2 else 1,  # 0: A larger, 1: B larger
                "label_text": "A" if n1 > n2 else "B",
                "a_words": a_words,
                "b_words": b_words,
                "a_digits": a_digits,
                "b_digits": b_digits,
                "a_mixed": a_digits,
                "b_mixed": b_words,  # mixed: A digits, B words
            }
        )

    rng.shuffle(records)
    return records


# ─────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────
 
def split_and_save(records: list, base_name: str):
    """Write train/val/test JSONL files."""
    val_end = TRAIN_SIZE + VAL_SIZE
    splits = {
        "train": records[:TRAIN_SIZE],
        "val": records[TRAIN_SIZE:val_end],
        "test": records[val_end:val_end + TEST_SIZE],
    }
    for name, data in splits.items():
        path = OUT_DIR / f"{base_name}_{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for rec in data:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  {len(data):>4} records -> {path}")
 
 
def main():
    # Two independent seeded streams. concat_rng only picks the word order of
    # ungrammatical items; keeping it separate preserves the exact datasets
    # used for the published results.
    rng = random.Random(SEED)
    concat_rng = random.Random(SEED)
 
    templates = load_templates()
    total = TRAIN_SIZE + VAL_SIZE + TEST_SIZE
    OUT_DIR.mkdir(exist_ok=True)
 
    for lang, lang_name in LANGUAGES.items():
        print(f"\n[{lang_name}]")
        for variant in VARIANTS:
            recs = build_task_a(lang, variant, total, templates[lang], rng, concat_rng)
            split_and_save(recs, f"task_a_{lang}_{variant}")
            recs = build_task_b(lang, variant, total, templates[lang], rng)
            split_and_save(recs, f"task_b_{lang}_{variant}")
 
    print(f"\nDone. Datasets written to {OUT_DIR}/")
 
 
if __name__ == "__main__":
    main()
