"""
generate_dataset.py
====================
Generates multilingual numerical probing datasets for Task A (Grammaticality)
and Task B (Value Comparison) across formats: digits, words, mixed.
Languages: English, French, Danish, Japanese.

Requirements:
    pip install num2words

Usage:
    python generate_dataset.py
    -> Creates datasets/ folder with .jsonl files
"""

import json
import random
import os
from itertools import product

try:
    from num2words import num2words
    NUM2WORDS_AVAILABLE = True
except ImportError:
    NUM2WORDS_AVAILABLE = False
    print("[WARNING] num2words not installed. Run: pip install num2words")
    print("          Falling back to English-only built-in converter.\n")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SEED = 42
NUMBER_RANGE = (1, 999)
TRAIN_SIZE = 500      # per condition (reduce if rate-limited)
VAL_SIZE = 100
TEST_SIZE = 200

LANGUAGES = {
    "en": "English",
    "fr": "French",
    "da": "Danish",
    "ja": "Japanese",
}

# num2words language codes
NUM2WORDS_LANG = {
    "en": "en",
    "fr": "fr",
    "da": "da",
    "ja": "ja",
}

# Sentence templates per language (with {number} placeholder)
SENTENCE_TEMPLATES = {
    "en": [
        "There are {number} books on the shelf.",
        "She counted {number} apples in the basket.",
        "The store has {number} items in stock.",
        "He walked {number} steps today.",
        "The class has {number} students.",
    ],
    "fr": [
        "Il y a {number} livres sur l'étagère.",
        "Elle a compté {number} pommes dans le panier.",
        "Le magasin a {number} articles en stock.",
        "Il a marché {number} pas aujourd'hui.",
        "La classe a {number} élèves.",
    ],
    "da": [
        "Der er {number} bøger på hylden.",
        "Hun talte {number} æbler i kurven.",
        "Butikken har {number} varer på lager.",
        "Han gik {number} skridt i dag.",
        "Klassen har {number} elever.",
    ],
    "ja": [
        "棚には{number}冊の本があります。",
        "彼女はかごの中に{number}個のリンゴを数えました。",
        "店には{number}点の商品があります。",
        "彼は今日{number}歩歩きました。",
        "クラスには{number}人の生徒がいます。",
    ],
}


# ─────────────────────────────────────────────
# NUMBER CONVERSION
# ─────────────────────────────────────────────

def number_to_words(n: int, lang: str) -> str:
    """Convert integer to spelled-out words in given language."""
    if NUM2WORDS_AVAILABLE:
        try:
            return num2words(n, lang=NUM2WORDS_LANG[lang])
        except Exception:
            pass
    # Fallback: English only, basic
    return _basic_english_words(n)


def _basic_english_words(n: int) -> str:
    """Basic English number-to-words (fallback when num2words unavailable)."""
    ones = ["", "one", "two", "three", "four", "five", "six", "seven",
            "eight", "nine", "ten", "eleven", "twelve", "thirteen",
            "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
    tens = ["", "", "twenty", "thirty", "forty", "fifty",
            "sixty", "seventy", "eighty", "ninety"]

    if n == 0:
        return "zero"
    if n < 20:
        return ones[n]
    if n < 100:
        return tens[n // 10] + ("-" + ones[n % 10] if n % 10 else "")
    if n < 1000:
        rest = n % 100
        base = ones[n // 100] + " hundred"
        if rest == 0:
            return base
        return base + " and " + _basic_english_words(rest)
    return str(n)


# ─────────────────────────────────────────────
# UNGRAMMATICAL NUMBER GENERATION (Task A)
# ─────────────────────────────────────────────

def make_ungrammatical(n1: int, n2: int, lang: str) -> str:
    """
    Create an ungrammatical number by concatenating two valid number words.
    e.g. "fifty-five" + "two hundred" -> "fifty-five two hundred"
    """
    w1 = number_to_words(n1, lang)
    w2 = number_to_words(n2, lang)
    # Randomly choose concatenation style
    styles = [
        f"{w1} {w2}",
        f"{w2} {w1}",
    ]
    return random.choice(styles)


# ─────────────────────────────────────────────
# DATASET BUILDERS
# ─────────────────────────────────────────────

def build_task_a(lang: str, variant: str, size: int, rng: random.Random) -> list:
    """
    Task A: Grammaticality Judgment
    variant: 'bare' | 'sentence'
    format: handled at prompt time (digits/words/mixed)
    Returns list of dicts.
    """
    lo, hi = NUMBER_RANGE
    records = []
    seen = set()

    templates = SENTENCE_TEMPLATES[lang]
    attempts = 0

    while len(records) < size and attempts < size * 10:
        attempts += 1
        is_grammatical = rng.random() < 0.5

        if is_grammatical:
            n = rng.randint(lo, hi)
            key = ("gram", n)
            if key in seen:
                continue
            seen.add(key)

            word_form = number_to_words(n, lang)
            digit_form = str(n)

            if variant == "bare":
                content_words = word_form
                content_digits = digit_form
            else:
                tmpl = rng.choice(templates)
                content_words = tmpl.replace("{number}", word_form)
                content_digits = tmpl.replace("{number}", digit_form)

            records.append({
                "task": "A",
                "lang": lang,
                "variant": variant,
                "label": 1,           # 1 = grammatical
                "label_text": "grammatical",
                "number": n,
                "content_words": content_words,
                "content_digits": content_digits,
            })

        else:
            n1 = rng.randint(lo, hi)
            n2 = rng.randint(lo, hi)
            if n1 == n2:
                continue
            key = ("ungram", min(n1, n2), max(n1, n2))
            if key in seen:
                continue
            seen.add(key)

            ungram_words = make_ungrammatical(n1, n2, lang)

            if variant == "bare":
                content_words = ungram_words
                content_digits = f"{n1} {n2}"   # digits version of ungrammatical
            else:
                tmpl = rng.choice(templates)
                content_words = tmpl.replace("{number}", ungram_words)
                content_digits = tmpl.replace("{number}", f"{n1} {n2}")

            records.append({
                "task": "A",
                "lang": lang,
                "variant": variant,
                "label": 0,           # 0 = ungrammatical
                "label_text": "ungrammatical",
                "number": None,
                "number_parts": [n1, n2],
                "content_words": content_words,
                "content_digits": content_digits,
            })

    rng.shuffle(records)
    return records


def build_task_b(lang: str, variant: str, size: int, rng: random.Random) -> list:
    """
    Task B: Value Comparison
    variant: 'bare' | 'sentence'
    Returns list of dicts with pairs.
    """
    lo, hi = NUMBER_RANGE
    records = []
    seen = set()

    templates = SENTENCE_TEMPLATES[lang]
    attempts = 0

    while len(records) < size and attempts < size * 10:
        attempts += 1
        n1 = rng.randint(lo, hi)
        n2 = rng.randint(lo, hi)
        if n1 == n2:
            continue
        key = (min(n1, n2), max(n1, n2))
        if key in seen:
            continue
        seen.add(key)

        w1 = number_to_words(n1, lang)
        w2 = number_to_words(n2, lang)
        d1 = str(n1)
        d2 = str(n2)

        label = 0 if n1 > n2 else 1   # 0: A>B, 1: B>A  (same as paper)
        label_text = "A" if n1 > n2 else "B"

        if variant == "bare":
            a_words, b_words = w1, w2
            a_digits, b_digits = d1, d2
        else:
            tmpl1 = rng.choice(templates)
            tmpl2 = rng.choice(templates)
            a_words = tmpl1.replace("{number}", w1)
            b_words = tmpl2.replace("{number}", w2)
            a_digits = tmpl1.replace("{number}", d1)
            b_digits = tmpl2.replace("{number}", d2)

        records.append({
            "task": "B",
            "lang": lang,
            "variant": variant,
            "n1": n1,
            "n2": n2,
            "label": label,
            "label_text": label_text,   # which side is LARGER
            "a_words": a_words,
            "b_words": b_words,
            "a_digits": a_digits,
            "b_digits": b_digits,
            # mixed: A is digits, B is words
            "a_mixed": a_digits,
            "b_mixed": b_words,
        })

    rng.shuffle(records)
    return records


# ─────────────────────────────────────────────
# SPLIT & SAVE
# ─────────────────────────────────────────────

def split_and_save(records: list, out_dir: str, base_name: str):
    """Split into train/val/test and save as JSONL."""
    os.makedirs(out_dir, exist_ok=True)

    n = len(records)
    train_end = TRAIN_SIZE
    val_end = train_end + VAL_SIZE

    splits = {
        "train": records[:train_end],
        "val": records[train_end:val_end],
        "test": records[val_end:val_end + TEST_SIZE],
    }

    for split_name, split_data in splits.items():
        path = os.path.join(out_dir, f"{base_name}_{split_name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for rec in split_data:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  Saved {len(split_data):>4} records -> {path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    random.seed(SEED)
    rng = random.Random(SEED)

    out_dir = "datasets"
    total_needed = TRAIN_SIZE + VAL_SIZE + TEST_SIZE

    print("=" * 60)
    print("Generating Multilingual Numerical Probing Datasets")
    print("=" * 60)

    for lang in LANGUAGES:
        lang_name = LANGUAGES[lang]
        print(f"\n[{lang_name}]")

        for variant in ["bare", "sentence"]:
            print(f"  Task A ({variant})...")
            records_a = build_task_a(lang, variant, total_needed, rng)
            split_and_save(records_a, out_dir, f"task_a_{lang}_{variant}")

            print(f"  Task B ({variant})...")
            records_b = build_task_b(lang, variant, total_needed, rng)
            split_and_save(records_b, out_dir, f"task_b_{lang}_{variant}")

    print("\n" + "=" * 60)
    print("Dataset generation complete!")
    print(f"All files saved to: ./{out_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
