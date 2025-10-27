# Citations Pipeline Work Test - Research Engineer

## Overview

Can you predict which of two research papers will be more influential? Questions like this are a powerful way to evaluate an AI model's research taste. Your task is to create a data pipeline for generating such questions.

**The challenge**: Given two arXiv papers, can Claude predict
which will have more citations, only by reasoning on the quality of the papers?

**What you'll build**: A pipeline from scraping arXiv papers →
collecting citations → matching papers to compare → generating questions → evaluating
multiple Claude models. Does Sonnet 4 significantly outperform
Haiku 3.5 on criteria like accuracy, brier score and AUROC? If not, your evaluation has problems.

**Why this matters**: Building great AI evaluations involves reasoning about how to collect, clean and manipulate data so that you can use it to measure a specific skill. An easy way to build a bad evaluation, and to fail this test, is to mess up that process. Your task is to make good design choices and go through the cleaning and filtering work to produce high quality questions. See the section on 'Data and Question Quality.'

**AI Use**: Using AI coding assistants is allowed, but you're being scored on code *concision* and data quality. Coding assistants can produce bloated code, and introduce bugs that mess up your data quality. Be warned!

**Billing**: We'll reimburse you up to $250 Anthropic API credits by default, just send us a screenshot. If you want more, please reach out.

**Time**: 8-12 hours 

**Full requirements**: See sections below for detailed instructions

---

## Quick Start

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="your-key"

# Start building your pipeline
jupyter notebook generate_forecasting_dataset.ipynb
```

## What's Provided

| File                                 | Purpose                                                          |
| ------------------------------------ | ---------------------------------------------------------------- |
| `src/eval.py`                        | Evaluation framework - use `evaluate_and_plot()` to run models   |
| `src/model_client.py`                | LLM API client for Claude, GPT, Gemini                           |
| `src/data_classes.py`                | `ForecastingQuestion` and `ArxivPaper` dataclasses - your schema |
| `generate_forecasting_dataset.ipynb` | Template for your submission                                     |

**Note on source code**: The `src/` directory contains utility code to make your life easier - you don't need to read or modify it. Just import what you need (dataclasses, evaluation functions) and focus on building your data pipeline in the notebook.

## What You Build

**Everything** in `generate_forecasting_dataset.ipynb`:

```python
# 1. Your paper dataclass
@dataclass
class ArxivPaper:
    arxiv_id: str
    title: str
    full_text: str
    published_timestamp: str
    categories: List[str]
    citations: int
    citation_timestamp: str

# 2. Scrape arXiv papers
papers = scrape_arxiv_papers(categories=[...], ...)

# 3. Get citations from API (Semantic Scholar, Google Scholar, etc.)
papers = add_citation_counts(papers)

# 4. Create smart pairs (avoid spurious cues!)
pairs = pair_papers(papers, ...)

# 5. Generate questions with full paper text
questions = [make_question(a, b) for a, b in pairs]

# 6. Export using provided schema
from src.data_classes import ForecastingQuestion
df = pd.DataFrame([q.__dict__ for q in questions])
df.to_parquet("citations_dataset.parquet")
```

## Data and Question Quality

Your evaluation is only as good as your data and question quality. At each step of your
pipeline, you should be thinking about these potential issues:

### 1. Spurious Cues

**Definition**: The goal of this evaluation is to measure Claude's ability to assess the quality of a paper. We do that by having Claude guess which of two papers got more citations. Therefore, a spurious cue is any feature of the question makes it 'too easy' by letting Claude guess which paper has more citations without reasoning about the quality of a paper.

There are several potential spurious cues in this data pipeline! A major part of your task is to reason about what they might be, and then filter them out. 

### 2. Ambiguity

**Definition**: Unclear or inconsistent ground truth, resolution criteria, or
question wording.

If you can't definitively say what the "correct" answer is, neither can a model.
Ambiguity comes from incomplete data, unreliable sources, edge cases in
resolution criteria, or questions that could reasonably be interpreted multiple
ways.

### 3. Signal-to-Noise Ratio

**Definition**: How much does performance require genuine reasoning vs. random
guessing?

If the task is too easy (obvious differences) or too hard (pure noise), it's not a good evaluation. You need questions where reasoning helps but
isn't trivial - where a smart human could do better than chance, and a smarter
human could do better than that.

### 4. Selection Effects and Biases

**Definition**: Your sampling process creates patterns that don't reflect the
real task.

If you only select papers that meet certain criteria, or pair them in systematic
ways, you may introduce hidden correlations. Models may learn these sampling
artifacts rather than the underlying task.

### 5. Data Contamination

**Definition**: Training data overlap that allows memorization instead of
reasoning.

If your evaluation data was in the model's training set, you're measuring
memory, not capability. This means being careful with
paper publication dates and model training cutoff dates.

---

**Key insight**: Every pipeline decision affects these quality gates. You must
explicitly analyze how your approach addresses each gate, and document the
tradeoffs you made.

**Our Advice**: Sometimes the only way to ensure you're catching problems is to 

## Evaluation

Use the provided framework:

```python
from src.eval import evaluate_and_plot

predictions, metrics = await evaluate_and_plot(
    questions,
    model_ids=[
        "claude-3-5-haiku-latest",
        "claude-sonnet-4-20250929"
    ],
    output_dir=Path("evaluation_results"),
    experiment_name="my_pipeline"
)
```

Results auto-saved to `evaluation_results/`.

## Required Schema and Format

See `src/data_classes.py` for the schema. Your parquet must have these columns:

- **`question`** - Full text with paper content (70k-100k chars). Use XML
  structure:
  ```
  Will paper A receive more citations than paper B by {timestamp}? Yes or No?
  Here are the titles, abstracts, text and publication dates for both papers.

  <paper_a>
  <title>{paper_a_title}</title>
  <full_text>{paper_a_full_text}</full_text>
  <publication_date>{paper_a_published_timestamp}</publication_date>
  </paper_a>

  <paper_b>
  <title>{paper_b_title}</title>
  <full_text>{paper_b_full_text}</full_text>
  <publication_date>{paper_b_published_timestamp}</publication_date>
  </paper_b>

  Resolution Criteria:
  This question resolves to "Yes" if Paper A has more citations than Paper B on {timestamp}.
  This question resolves to "No" if Paper B has more citations than Paper A on {timestamp}.

  Question: Will paper A have more citations on {timestamp}, than paper B, Yes or No?
  ```

- **`resolution`** - Float: `1.0` if Paper A has more citations (Yes), `0.0` if
  Paper B has more (No)
- **`creation_date`** - ISO 8601 timestamp: `"2025-01-15T00:00:00Z"`
- **`resolution_date`** - ISO 8601 timestamp when citations were measured
- **`metadata`** - JSON string with paper details:
  ```json
  {
    "paper_a_id": "2401.12345",
    "paper_b_id": "2401.67890",
    "paper_a_citations": 42,
    "paper_b_citations": 15
  }
  ```
- **`uuid`** - Unique identifier (use `uuid.uuid4()`)
- **`resolution_evidence`** - Human-readable explanation:
  `"Paper A (2401.12345) has 42 citations. Paper B (2401.67890) has 15 citations."`
- **`pipeline`** - Your pipeline identifier: `"arxiv_citations_v1"`
- **`numerical_resolution`** - Boolean: `True` (always true for this task)

## Deliverables

```
├── generate_forecasting_dataset.ipynb  # Your pipeline + documentation
├── citations_dataset.parquet           # 500 high quality questions
└── evaluation_results/                 # Results for Sonnet 4 and Haiku 3.5 on 100 questions
```


## Performance Rubric

1. Is your code clean, concise and well documented?
2. Do you reason well about potential quality issues in your pipeline? Did you make good design choices to produce high quality questions?
3. Are your final questions high quality?

## Submission

Upload your three deliverables to a google drive folder named "Citations Pipeline Work Test - [Your Name]" and share it with: everett@watertightai.com, tom@watertightai.com

**Questions?** Call or text anytime as much as you need 5207802861

---

_Build your pipeline. Demonstrate your thinking. Show your work._
