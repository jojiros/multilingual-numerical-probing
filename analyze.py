"""
Summarise the model responses in results/ into tables, figures and an error report.

Outputs:
    accuracy_summary.csv   accuracy per task x language x variant x format
    error_analysis.json    error counts, answer bias, unparseable replies
    figures/*.png          bar charts and heatmaps

Needs no API key: it only reads the saved responses.

Usage:
    python analyze.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save figures to files without opening a window
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")
CSV_PATH = Path("results/accuracy_summary.csv")
ERRORS_PATH = Path("results/error_analysis.json")

LANGUAGES = {"en": "English", "fr": "French", "da": "Danish", "ja": "Japanese"}
VARIANTS = ["bare", "sentence"]
FORMATS = ["words", "digits", "mixed"]
TASKS = ["A", "B"]
FORMAT_COLORS = {"words": "#4C72B0", "digits": "#DD8452", "mixed": "#55A868"}
VARIANT_COLORS = {"bare": "#2196F3", "sentence": "#FF9800"}


# ─────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────

def load_results() -> pd.DataFrame:
    """One row per model response, from every results/*.jsonl file."""
    rows = []
    for path in sorted(RESULTS_DIR.glob("*.jsonl")):
        with open(path, encoding="utf-8") as f:
            rows.extend(json.loads(line) for line in f)
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Task A "mixed" is identical to "digits" (an expression holds one number),
    # so results from older runs that still include it are dropped.
    dup = (df["task"] == "A") & (df["format"] == "mixed")
    if dup.any():
        print(f"Ignoring {dup.sum()} Task A 'mixed' records (duplicate of 'digits').")
        df = df[~dup]

    print(f"Loaded {len(df)} responses from {RESULTS_DIR}/")
    return df


# ─────────────────────────────────────────────
# Accuracy table
# ─────────────────────────────────────────────

def accuracy_table(df: pd.DataFrame) -> pd.DataFrame:
    """Accuracy, correct and total per task / language / variant / format."""
    table = (
        df.groupby(["task", "lang", "variant", "format"])["correct"]
        .agg(correct="sum", total="count")
        .reset_index()
    )
    table["correct"] = table["correct"].astype(int)
    table["accuracy"] = (table["correct"] / table["total"]).round(4)
    table.insert(2, "lang_name", table["lang"].map(LANGUAGES))
    return table[["task", "lang", "lang_name", "variant", "format", "accuracy", "correct", "total"]]


def print_table(table: pd.DataFrame):
    print()
    print(table[["task", "lang_name", "variant", "format", "accuracy", "total"]]
          .to_string(index=False, float_format="%.3f"))


# ─────────────────────────────────────────────
# Error analysis
# ─────────────────────────────────────────────

def error_analysis(df: pd.DataFrame) -> dict:
    """
    Error counts per condition, plus how often each answer was given.
    The answer distribution shows bias: gold labels are roughly 50/50, so a
    model that answers "A" 90% of the time is guessing by position, not by value.
    """
    df = df.assign(condition="task_" + df["task"] + "_" + df["lang"] + "_"
                   + df["variant"] + "_" + df["format"])
    wrong = df[~df["correct"]]

    api_errors = df[df["raw_response"] == "ERROR"]
    unparseable = df[(df["prediction"] == "invalid") & (df["raw_response"] != "ERROR")]

    shares = df.groupby("condition")["prediction"].value_counts(normalize=True).round(3)
    answer_share = {cond: shares[cond].to_dict() for cond in sorted(df["condition"].unique())}

    error_counts = wrong["condition"].value_counts()
    most_confused = [
        {
            "condition": cond,
            "error_count": int(n),
            "examples": wrong[wrong["condition"] == cond]
            .head(3)[["prediction", "label_text", "raw_response"]]
            .rename(columns={"label_text": "gold", "raw_response": "raw"})
            .to_dict(orient="records"),
        }
        for cond, n in error_counts.head(5).items()
    ]

    summary = {
        "api_errors": int(len(api_errors)),
        "unparseable_responses": int(len(unparseable)),
        "unparseable_examples": unparseable.head(10)[
            ["task", "lang", "variant", "format", "raw_response"]].to_dict(orient="records"),
        "error_counts_by_condition": {k: int(v) for k, v in sorted(error_counts.items())},
        "answer_share_by_condition": answer_share,
        "most_confused": most_confused,
    }
    with open(ERRORS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved {ERRORS_PATH}")
    return summary


# ─────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────

def grouped_bars(ax, data: pd.DataFrame, group_col: str, colors: dict):
    """Bars per language, one colour per value of group_col."""
    langs = [l for l in LANGUAGES if l in data["lang"].values]
    groups = [g for g in colors if g in data[group_col].values]
    width = 0.8 / len(groups)
    for i, g in enumerate(groups):
        accs = (data[data[group_col] == g].set_index("lang")["accuracy"]
                .reindex(langs).fillna(0))
        offset = (i - len(groups) / 2 + 0.5) * width
        ax.bar([x + offset for x in range(len(langs))], accs, width,
               label=g.capitalize(), color=colors[g], alpha=0.85)
    ax.set_xticks(range(len(langs)))
    ax.set_xticklabels([LANGUAGES[l] for l in langs])
    style_axes(ax)


def style_axes(ax):
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.4, label="Chance (0.5)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()


def save(fig, name: str):
    path = FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_by_format(table, task):
    data = table[(table["task"] == task) & (table["variant"] == "bare")]
    if data.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    grouped_bars(ax, data, "format", FORMAT_COLORS)
    ax.set_title(f"Task {task}: accuracy by format (bare numbers)")
    save(fig, f"task_{task}_by_format_bare.png")


def plot_bare_vs_sentence(table, task):
    for fmt in FORMATS:
        data = table[(table["task"] == task) & (table["format"] == fmt)]
        if data.empty:
            continue
        fig, ax = plt.subplots(figsize=(10, 5))
        grouped_bars(ax, data, "variant", VARIANT_COLORS)
        ax.set_title(f"Task {task} ({fmt}): bare vs sentence")
        save(fig, f"task_{task}_{fmt}_bare_vs_sentence.png")


def plot_heatmap(table, task, variant):
    data = table[(table["task"] == task) & (table["variant"] == variant)]
    if data.empty:
        return
    grid = (data.pivot(index="format", columns="lang", values="accuracy")
            .reindex(index=[f for f in FORMATS if f in data["format"].values],
                     columns=[l for l in LANGUAGES if l in data["lang"].values]))
    grid.index = [f.capitalize() for f in grid.index]
    grid.columns = [LANGUAGES[l] for l in grid.columns]

    fig, ax = plt.subplots(figsize=(8, len(grid) + 1.5))
    sns.heatmap(grid, annot=True, fmt=".2f", cmap="YlOrRd", vmin=0.4, vmax=1.0,
                linewidths=0.5, annot_kws={"size": 12}, ax=ax)
    ax.set_title(f"Task {task} ({variant}): accuracy")
    ax.set_xlabel("Language")
    ax.set_ylabel("Format")
    save(fig, f"task_{task}_{variant}_heatmap.png")


def plot_format_overview(table):
    """Mean accuracy per format, averaged over languages and variants (± SD)."""
    tasks = [t for t in TASKS if t in table["task"].values]
    fig, axes = plt.subplots(1, len(tasks), figsize=(6 * len(tasks), 5), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        stats = (table[table["task"] == task].groupby("format")["accuracy"]
                 .agg(["mean", "std"]).reindex([f for f in FORMATS])).dropna(how="all")
        ax.bar(range(len(stats)), stats["mean"], yerr=stats["std"].fillna(0), capsize=5,
               color=[FORMAT_COLORS[f] for f in stats.index], alpha=0.85)
        ax.set_xticks(range(len(stats)))
        ax.set_xticklabels([f.capitalize() for f in stats.index])
        style_axes(ax)
        ax.set_ylabel("Mean accuracy ± SD")
        ax.set_title(f"Task {task}: mean accuracy by format")
    save(fig, "overall_format_comparison.png")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    df = load_results()
    if df.empty:
        print(f"No results found in {RESULTS_DIR}/. Run evaluate.py first.")
        return

    table = accuracy_table(df)
    print_table(table)
    table.to_csv(CSV_PATH, index=False)
    print(f"\nSaved {CSV_PATH}")

    summary = error_analysis(df)
    print(f"API errors: {summary['api_errors']}, "
          f"unparseable replies: {summary['unparseable_responses']}")

    FIGURES_DIR.mkdir(exist_ok=True)
    for task in TASKS:
        plot_by_format(table, task)
        plot_bare_vs_sentence(table, task)
        for variant in VARIANTS:
            plot_heatmap(table, task, variant)
    plot_format_overview(table)


if __name__ == "__main__":
    main()