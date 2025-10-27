# Citations Pipeline Worktest - Research Engineer

## Overview

Can you predict which research paper will be more influential? Forecasting
questions are a powerful way to assess how good the world model of llms are. In
this worktest we want to understand if you can effectively write a data pipeline
that helps generate such questions at scale.

**The challenge**: Given two arXiv papers with full text, can Claude predict
which will have more citations? But here's the catch - you need to design this
evaluation carefully. Include author names and the model just pattern-matches on
"famous researcher = more citations." Your job is to create questions that
require actual reasoning about research quality.

**What you'll build**: A complete pipeline from scraping arXiv papers →
collecting citations → creating smart pairs → generating questions → evaluating
multiple Claude models. The key test: does Sonnet 4 significantly outperform
Haiku 3.5? If not, your evaluation has problems.

**Why this matters**: Building good evaluations is hard. Most fail because of
spurious correlations, ambiguous questions, or low signal-to-noise. This
worktest tests whether you can design a data pipeline that avoids these
pitfalls.

**Time**: 8-12 hours | **Submit within**: 1 week

**Full requirements**: See sections below for detailed instructions

---

## What We're Testing

- Data pipeline design
- Critical thinking about spurious correlations
- Understanding model scaling properties
- Clear technical documentation
- Ability to write clean, robust & understandable code

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

## What You Build

Build **everything** from scratch in `generate_forecasting_dataset.ipynb`:

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
papers = scrape_arxiv_papers(categories=['cs.LG', 'cs.AI'], ...)

# 3. Get citations from API (Semantic Scholar, Google Scholar, etc.)
papers = add_citation_counts(papers)

# 4. Create smart pairs (avoid spurious cues!)
pairs = pair_papers(papers, min_citation_diff=2, max_days_apart=21)

# 5. Generate questions with full paper text
questions = [make_question(a, b) for a, b in pairs]

# 6. Export using provided schema
from src.data_classes import ForecastingQuestion
df = pd.DataFrame([q.__dict__ for q in questions])
df.to_parquet("citations_dataset.parquet")
```

## Critical: Quality Gates at Every Step

Your evaluation is only as good as your data quality. At each step of your
pipeline, evaluate against these quality gates:

### 1. Spurious Cues

**Definition**: Features that leak the answer without requiring the reasoning
you want to test.

Models can exploit shortcuts: metadata that correlates with the answer. If
present, models pattern-match instead of reasoning about the actual question.

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

If the task is too easy (obvious differences) or too hard (pure noise), you
can't measure model capabilities. You need questions where reasoning helps but
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
memory, not capability. For recent models, this means being careful with
publication dates and public datasets.

---

**Key insight**: Every pipeline decision affects these quality gates. You must
explicitly analyze how your approach addresses each gate, and document the
tradeoffs you made.

## Evaluation

Use the provided framework:

```python
from src.eval import evaluate_and_plot

predictions, metrics = await evaluate_and_plot(
    questions,
    model_ids=[
        "claude-3-5-haiku-latest",
        "claude-3-7-sonnet-20250219",
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
├── citations_dataset.parquet           # Your generated dataset (must match schema above)
└── evaluation_results/                 # Model performance
```

**In your notebook**, document:

1. Your approach (categories, pairing strategy, citation source)
2. How you avoided spurious cues (author names, venues, etc.)
3. Analysis of common evaluation mistakes (spurious cues, ambiguity, low
   signal-to-noise, etc.)
4. Evaluation results and scaling analysis (Haiku → Sonnet 3.7 → Sonnet 4)
5. 100 example predictions with model reasoning

## Success Criteria

1. **Model scaling works**: Sonnet 4 >> Sonnet 3.7 >> Haiku 3.5
   - If flat or inverse scaling → problem with your evaluation
2. **No spurious cues**: You tell us what you think the spurious cues here might
   be
3. **Quality over quantity**: 200 excellent questions > 1,000 mediocre ones
4. **Clear documentation**: Reproducible from your notebook

## Submission

Email to: everett@watertightai.com, tom@watertightai.com

**Subject**: "Citations Pipeline Worktest - [Your Name]"

**Questions?** Call or text anytime as much as you need 5207802861

---

_Build your pipeline. Demonstrate your thinking. Show your work._
