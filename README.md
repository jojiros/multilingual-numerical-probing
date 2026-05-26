# Multilingual Numerical Understanding — Probing via Prompting

A term project investigating how well multilingual LLMs understand numbers
across formats (digits, words, mixed) and languages (EN, FR, DA, JA).

Inspired by: *Johnson et al. (2020), "Probing for Multilingual Numerical
Understanding in Transformer-Based Language Models"* — but modernized
from probing classifiers to **prompt-based evaluation** using LLaMA 3 via Groq.

---

## Project Structure

```
numerical_probing/
├── generate_dataset.py   # Step 1: generate all JSONL datasets
├── evaluate.py           # Step 2: run prompts against Groq API
├── analyze.py            # Step 3: compute accuracy + generate plots
├── datasets/             # auto-created by generate_dataset.py
├── results/              # auto-created by evaluate.py
└── figures/              # auto-created by analyze.py
```

---

## Setup

### 1. Install dependencies

```bash
pip install num2words groq pandas matplotlib seaborn
```

### 2. Get a free Groq API key

1. Go to https://console.groq.com
2. Sign up (free, no credit card)
3. Create an API key under "API Keys"

### 3. Set your API key

```bash
# Option A: environment variable (recommended)
export GROQ_API_KEY="gsk_your_key_here"

# Option B: edit evaluate.py line ~30
GROQ_API_KEY = "gsk_your_key_here"
```

---

## Running the Pipeline

### Step 1 — Generate datasets

```bash
python generate_dataset.py
```

Creates `datasets/` with JSONL files for each condition:
- `task_a_{lang}_{variant}_{split}.jsonl`  (Task A: grammaticality)
- `task_b_{lang}_{variant}_{split}.jsonl`  (Task B: value comparison)

Conditions:
- **Languages**: en, fr, da, ja
- **Variants**: bare (numbers alone), sentence (numbers in context)
- **Splits**: train, val, test

### Step 2 — Evaluate

```bash
# Evaluate everything (takes ~30-60 min on free Groq tier)
python evaluate.py

# Quick test: one condition only
python evaluate.py --task B --lang en --variant bare --format words

# Dry run (prints prompts, no API calls)
python evaluate.py --dry-run --task B --lang en --variant bare
```

**Formats evaluated** (added over original paper):
| Format | Task A example | Task B example |
|--------|---------------|----------------|
| words  | "five hundred two" | "sixty" vs "seventy" |
| digits | "502" | "60" vs "70" |
| mixed  | — | "60" vs "seventy" |

### Step 3 — Analyze

```bash
python analyze.py
```

Produces:
- `accuracy_summary.csv` — full accuracy table (paste into paper)
- `error_analysis.json` — error patterns
- `figures/` — bar charts and heatmaps (include in paper)

---

## Groq Rate Limits (Free Tier)

| Model | Requests/min | Tokens/min | Requests/day |
|-------|-------------|------------|--------------|
| llama-3.1-8b-instant | 30 | 131,072 | 14,400 |

At 200 test samples × 4 langs × 2 variants × 3 formats × 2 tasks = **9,600 requests**.
This fits in the daily limit. With 0.5s delay between requests, full eval takes ~80 min.

**Tips to stay within limits:**
- Run one task at a time: `--task A` then `--task B`
- Results are cached — re-running skips already-evaluated conditions
- Reduce TEST_SIZE in generate_dataset.py if needed (try 100)

---

## Tasks

### Task A: Grammaticality Judgment

The model sees a number expression and decides if it's valid.

**Prompt example:**
```
Language: English
Number expression: "five hundred two"

Is this number expression grammatically valid in English?
Answer with ONLY one word: grammatical or ungrammatical.
```

Labels: `grammatical` (1) / `ungrammatical` (0)
Chance baseline: **50%**

### Task B: Value Comparison

The model compares two quantities.

**Prompt example (words format):**
```
Which quantity is larger?

A: sixty
B: seventy

Answer with ONLY the letter A or B.
```

**Prompt example (mixed format):**
```
Which quantity is larger?

A: 60
B: seventy

Answer with ONLY the letter A or B.
```

Labels: `A` (first is larger) / `B` (second is larger)
Chance baseline: **50%**

---

## Research Questions

1. Do models perform better with digits or spelled-out numbers?
2. Can models generalize between formats (mixed condition)?
3. Are there differences across languages with different number-system structures?
4. Does embedding numbers in sentences help or hurt performance?

---

## Expected Paper Sections

1. **Introduction** — motivation, connection to Johnson et al.
2. **Related Work** — Johnson et al., Wallace et al., prompting vs probing
3. **Methods** — tasks, formats, languages, prompt design, dataset generation
4. **Results** — accuracy tables (from CSV), heatmaps, bar charts
5. **Discussion** — format effects, language effects, bare vs sentence
6. **Conclusion** — summary, limitations, future work

---

## Model Used

- **Model**: `llama-3.1-8b-instant` (LLaMA 3.1 8B)
- **Provider**: Groq (free tier)
- **Temperature**: 0.0 (deterministic)
- **Max tokens**: 10

To try a larger model, change `MODEL` in `evaluate.py`:
```python
MODEL = "llama-3.1-70b-versatile"  # better but slower
```

---

## Citation

Johnson, D., Mak, D., Barker, D., & Loessberg-Zahl, L. (2020).
Probing for Multilingual Numerical Understanding in Transformer-Based
Language Models. *Proceedings of the Third BlackboxNLP Workshop*, 184–192.
