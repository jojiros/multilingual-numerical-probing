"""
analyze.py
==========
Reads results/ JSONL files and produces:
  - accuracy_summary.csv       (full accuracy table)
  - figures/                   (bar charts and heatmaps)
  - error_analysis.json        (common error patterns)

Requirements:
    pip install pandas matplotlib seaborn

Usage:
    python analyze.py
"""

import json
import os
import csv
from pathlib import Path
from collections import defaultdict

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("[WARNING] pandas/matplotlib/seaborn not installed.")
    print("          Run: pip install pandas matplotlib seaborn")
    print("          Continuing with text-only output.\n")

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

LANGUAGES = ["en", "fr", "da", "ja"]
LANG_NAMES = {"en": "English", "fr": "French", "da": "Danish", "ja": "Japanese"}
VARIANTS = ["bare", "sentence"]
FORMATS = ["words", "digits", "mixed"]
TASKS = ["A", "B"]


# ─────────────────────────────────────────────
# LOAD RESULTS
# ─────────────────────────────────────────────

def load_all_results() -> list:
    """Load all result JSONL files into a flat list of records."""
    all_records = []
    for path in sorted(RESULTS_DIR.glob("*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                all_records.append(rec)
    print(f"Loaded {len(all_records)} records from {RESULTS_DIR}/")
    return all_records


# ─────────────────────────────────────────────
# ACCURACY COMPUTATION
# ─────────────────────────────────────────────

def compute_accuracy_table(records: list) -> list:
    """Compute accuracy per (task, lang, variant, format) condition."""
    buckets = defaultdict(lambda: {"correct": 0, "total": 0})

    for rec in records:
        task = rec.get("task", "?")
        lang = rec.get("lang", "?")
        variant = rec.get("variant", "?")
        fmt = rec.get("format", "?")
        key = (task, lang, variant, fmt)
        buckets[key]["total"] += 1
        if rec.get("correct"):
            buckets[key]["correct"] += 1

    rows = []
    for (task, lang, variant, fmt), counts in sorted(buckets.items()):
        acc = counts["correct"] / counts["total"] if counts["total"] > 0 else 0
        rows.append({
            "task": task,
            "lang": lang,
            "lang_name": LANG_NAMES.get(lang, lang),
            "variant": variant,
            "format": fmt,
            "accuracy": round(acc, 4),
            "correct": counts["correct"],
            "total": counts["total"],
        })
    return rows


def save_csv(rows: list, path: str = "accuracy_summary.csv"):
    """Save accuracy table to CSV."""
    if not rows:
        print("[WARN] No results to save.")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved accuracy table -> {path}")


def print_accuracy_table(rows: list):
    """Pretty-print accuracy table to console."""
    print("\n" + "=" * 70)
    print(f"{'Task':<6} {'Lang':<10} {'Variant':<10} {'Format':<8} {'Accuracy':>10} {'N':>6}")
    print("-" * 70)
    for r in rows:
        print(f"{r['task']:<6} {r['lang_name']:<10} {r['variant']:<10} {r['format']:<8} "
              f"{r['accuracy']:>10.3f} {r['total']:>6}")
    print("=" * 70)


# ─────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────

def plot_task_accuracy_by_format(df, task: str):
    """Bar chart: accuracy by format, grouped by language."""
    subset = df[(df["task"] == task) & (df["variant"] == "bare")]
    if subset.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    lang_names = [LANG_NAMES[l] for l in LANGUAGES if l in subset["lang"].values]
    x = range(len(lang_names))
    width = 0.25
    colors = {"words": "#4C72B0", "digits": "#DD8452", "mixed": "#55A868"}

    for i, fmt in enumerate(FORMATS):
        fmt_data = subset[subset["format"] == fmt]
        accs = []
        for lang in LANGUAGES:
            row = fmt_data[fmt_data["lang"] == lang]
            accs.append(row["accuracy"].values[0] if not row.empty else 0)
        offset = (i - 1) * width
        bars = ax.bar([xi + offset for xi in x], accs, width,
                      label=fmt.capitalize(), color=colors[fmt], alpha=0.85)

    ax.set_xlabel("Language", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(f"Task {task} — Accuracy by Format (Bare Numbers)", fontsize=14)
    ax.set_xticks(list(x))
    ax.set_xticklabels([LANG_NAMES[l] for l in LANGUAGES])
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.4, label="Chance (0.5)")
    ax.legend(title="Format")
    ax.grid(axis="y", alpha=0.3)

    path = FIGURES_DIR / f"task_{task}_by_format_bare.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved figure -> {path}")


def plot_bare_vs_sentence(df, task: str):
    """Bar chart: bare vs sentence accuracy per language and format."""
    subset = df[df["task"] == task]
    if subset.empty:
        return

    for fmt in FORMATS:
        fmt_data = subset[subset["format"] == fmt]
        if fmt_data.empty:
            continue

        fig, ax = plt.subplots(figsize=(9, 5))
        x = range(len(LANGUAGES))
        width = 0.35
        colors = {"bare": "#2196F3", "sentence": "#FF9800"}

        for i, variant in enumerate(VARIANTS):
            var_data = fmt_data[fmt_data["variant"] == variant]
            accs = []
            for lang in LANGUAGES:
                row = var_data[var_data["lang"] == lang]
                accs.append(row["accuracy"].values[0] if not row.empty else 0)
            offset = (i - 0.5) * width
            ax.bar([xi + offset for xi in x], accs, width,
                   label=variant.capitalize(), color=colors[variant], alpha=0.85)

        ax.set_xlabel("Language", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title(f"Task {task} ({fmt.capitalize()}) — Bare vs Sentence", fontsize=13)
        ax.set_xticks(list(x))
        ax.set_xticklabels([LANG_NAMES[l] for l in LANGUAGES])
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5, color="red", linestyle="--", alpha=0.4, label="Chance (0.5)")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        path = FIGURES_DIR / f"task_{task}_{fmt}_bare_vs_sentence.png"
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Saved figure -> {path}")


def plot_heatmap(df, task: str, variant: str):
    """Heatmap: language x format accuracy."""
    subset = df[(df["task"] == task) & (df["variant"] == variant)]
    if subset.empty:
        return

    pivot_data = {}
    for fmt in FORMATS:
        fmt_data = subset[subset["format"] == fmt]
        row_vals = {}
        for lang in LANGUAGES:
            cell = fmt_data[fmt_data["lang"] == lang]
            row_vals[LANG_NAMES[lang]] = cell["accuracy"].values[0] if not cell.empty else 0
        pivot_data[fmt.capitalize()] = row_vals

    import pandas as pd_local
    heatmap_df = pd_local.DataFrame(pivot_data).T

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(heatmap_df, annot=True, fmt=".2f", cmap="YlOrRd",
                vmin=0.4, vmax=1.0, ax=ax, linewidths=0.5,
                annot_kws={"size": 12})
    ax.set_title(f"Task {task} ({variant.capitalize()}) — Accuracy Heatmap", fontsize=13)
    ax.set_xlabel("Language")
    ax.set_ylabel("Format")

    path = FIGURES_DIR / f"task_{task}_{variant}_heatmap.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved figure -> {path}")


def plot_format_comparison_overall(df):
    """Summary plot: average accuracy per format across all conditions."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {"words": "#4C72B0", "digits": "#DD8452", "mixed": "#55A868"}

    for ax, task in zip(axes, ["A", "B"]):
        subset = df[df["task"] == task]
        fmt_means = {}
        fmt_stds = {}
        for fmt in FORMATS:
            vals = subset[subset["format"] == fmt]["accuracy"].values
            fmt_means[fmt] = vals.mean() if len(vals) > 0 else 0
            fmt_stds[fmt] = vals.std() if len(vals) > 0 else 0

        fmts = list(FORMATS)
        means = [fmt_means[f] for f in fmts]
        stds = [fmt_stds[f] for f in fmts]
        bar_colors = [colors[f] for f in fmts]

        ax.bar(fmts, means, yerr=stds, color=bar_colors, alpha=0.85,
               capsize=5, error_kw={"linewidth": 1.5})
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5, color="red", linestyle="--", alpha=0.4, label="Chance")
        ax.set_title(f"Task {task} — Mean Accuracy by Format", fontsize=13)
        ax.set_xlabel("Format")
        ax.set_ylabel("Mean Accuracy ± SD")
        ax.set_xticklabels([f.capitalize() for f in fmts])
        ax.grid(axis="y", alpha=0.3)
        ax.legend()

    path = FIGURES_DIR / "overall_format_comparison.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved figure -> {path}")


# ─────────────────────────────────────────────
# ERROR ANALYSIS
# ─────────────────────────────────────────────

def error_analysis(records: list) -> dict:
    """Identify common error patterns."""
    errors = defaultdict(list)
    invalid_responses = []

    for rec in records:
        if rec.get("correct"):
            continue
        task = rec.get("task")
        pred = rec.get("prediction", "")
        raw = rec.get("raw_response", "")

        if pred == "invalid":
            invalid_responses.append({
                "task": task,
                "lang": rec.get("lang"),
                "format": rec.get("format"),
                "raw_response": raw,
            })

        key = f"task_{task}_{rec.get('lang')}_{rec.get('format')}"
        errors[key].append({
            "prediction": pred,
            "gold": rec.get("label_text"),
            "raw": raw,
        })

    summary = {
        "total_invalid_responses": len(invalid_responses),
        "invalid_examples": invalid_responses[:10],
        "error_counts_by_condition": {k: len(v) for k, v in errors.items()},
        "most_confused": sorted(errors.items(), key=lambda x: len(x[1]), reverse=True)[:5],
    }

    # Simplify for JSON serialization
    summary["most_confused"] = [
        {"condition": k, "error_count": len(v), "examples": v[:3]}
        for k, v in summary["most_confused"]
    ]

    with open("error_analysis.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("Saved error analysis -> error_analysis.json")
    return summary


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Numerical Probing — Analysis")
    print("=" * 60)

    records = load_all_results()
    if not records:
        print("[ERROR] No results found in ./results/")
        print("        Run evaluate.py first.")
        return

    rows = compute_accuracy_table(records)
    print_accuracy_table(rows)
    save_csv(rows)

    # Error analysis
    error_summary = error_analysis(records)
    print(f"\nInvalid responses (model didn't follow format): {error_summary['total_invalid_responses']}")

    # Plotting
    if PLOTTING_AVAILABLE:
        import pandas as pd
        df = pd.DataFrame(rows)
        print("\nGenerating figures...")

        for task in TASKS:
            plot_task_accuracy_by_format(df, task)
            plot_bare_vs_sentence(df, task)
            for variant in VARIANTS:
                plot_heatmap(df, task, variant)

        plot_format_comparison_overall(df)
        print(f"\nAll figures saved to ./{FIGURES_DIR}/")
    else:
        print("\n[SKIP] Install pandas+matplotlib+seaborn for plots.")
        print("       pip install pandas matplotlib seaborn")

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("  accuracy_summary.csv  — full accuracy table")
    print("  error_analysis.json   — error patterns")
    print("  figures/              — all plots")
    print("=" * 60)


if __name__ == "__main__":
    main()
