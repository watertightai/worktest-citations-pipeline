"""
General evaluation utilities for forecasting questions.

This module provides dataset-agnostic utilities for evaluating language models
on forecasting questions. Works with any ForecastingQuestion objects.

Example usage (all-in-one):
    from src.eval import evaluate_and_plot
    from src.forecasting_question import ForecastingQuestion
    import asyncio

    # Load questions (from parquet, JSONL, etc.)
    questions = [...]

    # Evaluate, calculate metrics, plot, and save everything
    predictions, metrics = await evaluate_and_plot(
        questions,
        model_ids=["claude-sonnet-4-5-20250929", "claude-opus-4-1-20250805"],
        output_dir=Path("experiments/my-eval"),
        experiment_name="my_experiment"
    )

Example usage (step-by-step):
    from src.eval import evaluate_model_on_questions, save_evaluation_results, ModelPrediction
    from src.plotting import plot_model_comparison
    import asyncio

    # Run evaluation
    results = await evaluate_model_on_questions(
        questions,
        model_ids=["claude-opus-4-1", "claude-sonnet-4-0"]
    )  # Returns Dict[str, List[ModelPrediction]]

    # Save results
    save_evaluation_results(results, Path("experiments/eval-fred"))
"""

import asyncio
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy import stats
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from tqdm import tqdm

from src.forecasting_question import ForecastingQuestion
from src.model_client import ChatMessage, MessageRole, ModelClient


@dataclass
class ModelPrediction:
    """Result from a single model prediction on a forecasting question."""

    # Core prediction data
    model_id: str
    question_uuid: str
    predicted_probability: float  # 0-1
    actual_outcome: Optional[float]  # 0.0, 1.0, or None if unresolved

    # Model response
    prediction_text: str  # Full model response (reasoning + answer)

    # Derived fields
    correct: Optional[bool]  # None if actual_outcome is None

    # Question metadata (unpacked from question.metadata JSON)
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dict for JSON serialization."""
        return {
            "model_id": self.model_id,
            "question_uuid": self.question_uuid,
            "predicted_probability": self.predicted_probability,
            "actual_outcome": self.actual_outcome,
            "prediction_text": self.prediction_text,
            "correct": self.correct,
            **self.metadata,  # Flatten metadata into result
        }


# Anthropic API Rate Limits (Tier 4)
# Based on https://docs.anthropic.com/en/api/rate-limits
RATE_LIMITS = {
    # Claude 4.x models (all Claude 4.x versions)
    "claude-4": {
        "rpm": 4000,
        "tpm_in": 2_000_000,
        "tpm_out": 1_000_000,
    },  # Sonnet 4.x default
    "claude-sonnet-4": {"rpm": 4000, "tpm_in": 2_000_000, "tpm_out": 1_000_000},
    "claude-opus-4": {"rpm": 4000, "tpm_in": 2_000_000, "tpm_out": 400_000},
    # Claude 3.7
    "claude-3-7-sonnet": {"rpm": 4000, "tpm_in": 200_000, "tpm_out": 80_000},
    # Claude 3.5
    "claude-3-5-sonnet": {"rpm": 4000, "tpm_in": 400_000, "tpm_out": 80_000},
    "claude-3-5-haiku": {"rpm": 4000, "tpm_in": 400_000, "tpm_out": 80_000},
    # Claude 3 (legacy)
    "claude-3-opus": {"rpm": 4000, "tpm_in": 400_000, "tpm_out": 80_000},
    "claude-3-sonnet": {"rpm": 4000, "tpm_in": 400_000, "tpm_out": 80_000},
    "claude-3-haiku": {"rpm": 4000, "tpm_in": 400_000, "tpm_out": 80_000},
    # Conservative default for unknown models
    "default": {"rpm": 1000, "tpm_in": 100_000, "tpm_out": 50_000},
}

# Default prompts
DEFAULT_SYSTEM_PROMPT = """You are a forecasting expert. Provide your forecast as a single probability number between 0 and 1."""

DEFAULT_USER_PROMPT_TEMPLATE = """{question}

Be very concise in your reasoning before answering. After your reasoning, on a new line write your final answer in the format:

P=[probability between 0 and 1, where 1 means the answer is definitely yes and 0 means the answer is definitely no]"""


def get_rate_limits(model_id: str) -> Dict[str, int]:
    """
    Get rate limits for a specific model.

    Matches model_id to rate limit configuration using substring matching.

    Args:
        model_id: Model identifier (e.g., "claude-sonnet-4-0")

    Returns:
        Dict with keys: rpm, tpm_in, tpm_out

    Example:
        >>> get_rate_limits("claude-sonnet-4-0")
        {'rpm': 4000, 'tpm_in': 2000000, 'tpm_out': 1000000}
    """
    # Try to match model_id to a known pattern
    for pattern, limits in RATE_LIMITS.items():
        if pattern != "default" and pattern in model_id:
            return limits

    # Fallback to default
    return RATE_LIMITS["default"]


def estimate_question_tokens(question: ForecastingQuestion) -> int:
    """
    Estimate token count for a question.

    Uses rough heuristic: ~4 characters per token for English text.
    This is conservative (actual is ~3.5-4.5) to avoid hitting limits.

    Args:
        question: ForecastingQuestion object

    Returns:
        Estimated token count
    """
    # Count characters in question text
    char_count = len(question.question)

    # Conservative estimate: 4 chars per token
    token_estimate = char_count // 4

    # Add overhead for system prompt and formatting (~200 tokens)
    return token_estimate + 200


def calculate_optimal_batch_size(
    model_id: str,
    avg_question_tokens: int = 500,
    avg_response_tokens: int = 150,
) -> int:
    """
    Calculate optimal batch size to maximize throughput without hitting rate limits.

    The bottleneck is usually output tokens per minute (tpm_out), not requests per minute.
    We calculate how many requests can fit in 10 seconds (1/6 of the minute window).

    Args:
        model_id: Model identifier
        avg_question_tokens: Estimated input tokens per question (default: 500)
        avg_response_tokens: Estimated output tokens per response (default: 150)

    Returns:
        Optimal batch size for concurrent requests

    Example:
        Claude Sonnet 4:
            - TPM out: 1,000,000
            - Avg response: 150 tokens
            - Max responses/min: 1,000,000 / 150 = 6,666
            - Batch size (10s window): 6,666 / 6 = 111

        Claude 3.7 Sonnet:
            - TPM out: 80,000 (bottleneck!)
            - Avg response: 150 tokens
            - Max responses/min: 80,000 / 150 = 533
            - Batch size (10s window): 533 / 6 = 89
    """
    limits = get_rate_limits(model_id)

    # Calculate max requests based on output token limit (usually bottleneck)
    max_by_output = limits["tpm_out"] // avg_response_tokens

    # Calculate max requests based on input token limit
    max_by_input = limits["tpm_in"] // avg_question_tokens

    # Calculate max requests based on RPM limit
    max_by_rpm = limits["rpm"]

    # Take minimum (most restrictive constraint)
    max_requests_per_min = min(max_by_output, max_by_input, max_by_rpm)

    # Batch size = requests we can do in ~10 seconds (1/6 of minute)
    # Conservative to avoid bursting limits
    batch_size = max(1, int(max_requests_per_min / 6))

    return batch_size


def extract_probability(text: str) -> Optional[float]:
    """
    Extract probability from model response.

    Tries multiple patterns to parse probability from text:
    - P= format: "P=0.65", "P=0.725" (preferred format, expects 0-1)
    - Plain numbers: "65", "72.5"
    - Percentages: "65%", "72.5 percent"
    - With labels: "probability: 65"
    - Decimals: "0.65"

    Args:
        text: Model response text

    Returns:
        Probability as float between 0 and 1, or None if parsing fails

    Example:
        >>> extract_probability("P=0.65")
        0.65
        >>> extract_probability("72.5%")
        0.725
        >>> extract_probability("0.65")
        0.65
    """
    # First try P= format (expects 0-1 range)
    p_match = re.search(
        r"P\s*=\s*(\d+(?:\.\d+)?)", text.strip(), re.MULTILINE | re.IGNORECASE
    )
    if p_match:
        value = float(p_match.group(1))
        # P= format expects value already in 0-1 range
        return max(0.0, min(1.0, value))

    # Other patterns (may be percentages or decimals)
    patterns = [
        r"^\s*(\d+(?:\.\d+)?)\s*$",  # Just a number
        r"^\s*(\d+(?:\.\d+)?)\s*\n",  # Number followed by newline
        r"^\s*(\d+(?:\.\d+)?)\s",  # Number followed by space
        r"(\d+(?:\.\d+)?)\s*%",  # Number with percent sign
        r"(\d+(?:\.\d+)?)\s*percent",  # Number with "percent"
        r"probability.*?(\d+(?:\.\d+)?)",  # "probability: 65"
        r"(\d\.\d+)",  # Decimal format
    ]

    for pattern in patterns:
        match = re.search(pattern, text.strip(), re.MULTILINE | re.IGNORECASE)
        if match:
            value = float(match.group(1))
            # Convert percentage to decimal if needed
            if value > 1:
                value = value / 100
            # Clamp to [0, 1]
            return max(0.0, min(1.0, value))

    return None


async def predict_question_async(
    question: ForecastingQuestion,
    model_id: str,
    client: ModelClient,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    user_prompt_template: str = DEFAULT_USER_PROMPT_TEMPLATE,
    max_retries: int = 3,
    temperature: float = 1.0,
) -> Optional[ModelPrediction]:
    """
    Make a probability prediction for a single question with retries.

    Args:
        question: ForecastingQuestion object
        model_id: Model identifier (e.g., "claude-opus-4-1")
        client: ModelClient instance
        system_prompt: System message for the model
        user_prompt_template: Template for user message (must contain {question})
        max_retries: Maximum retry attempts for rate limits/errors
        temperature: Sampling temperature (0.0-1.0+, default 1.0)

    Returns:
        ModelPrediction object, or None if all retries failed
    """
    # Prepare messages
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
        ChatMessage(
            role=MessageRole.USER,
            content=user_prompt_template.format(question=question.question),
        ),
    ]

    # Retry logic with exponential backoff
    for attempt in range(max_retries):
        try:
            response = await client(
                model_id=model_id,
                messages=messages,
                max_tokens=800,
                temperature=temperature,
            )

            # Extract probability from response
            predicted_prob = extract_probability(response.completion)

            if predicted_prob is None:
                print(
                    f"⚠️  [{model_id}] Failed to parse probability from: {response.completion[:100]}"
                )
                predicted_prob = 0.5  # Default to 50% if parsing fails

            actual_outcome = question.resolution

            # Parse metadata
            try:
                metadata = (
                    json.loads(question.metadata)
                    if isinstance(question.metadata, str)
                    else question.metadata
                )
            except (json.JSONDecodeError, TypeError):
                metadata = {}

            # Build result
            return ModelPrediction(
                model_id=model_id,
                question_uuid=str(question.uuid),
                prediction_text=response.completion,
                predicted_probability=predicted_prob,
                actual_outcome=actual_outcome,
                correct=(predicted_prob >= 0.5 and actual_outcome == 1.0)
                or (predicted_prob < 0.5 and actual_outcome == 0.0)
                if actual_outcome is not None
                else None,
                metadata=metadata,
            )

        except Exception as e:
            error_str = str(e).lower()

            # Check for rate limit errors
            if "rate limit" in error_str or "429" in error_str:
                wait_time = (2**attempt) * 5  # Exponential backoff: 5s, 10s, 20s
                print(
                    f"⏳ Rate limit hit, waiting {wait_time}s... (attempt {attempt + 1}/{max_retries})"
                )
                await asyncio.sleep(wait_time)

            # Check for server errors
            elif "500" in error_str or "internal server error" in error_str:
                wait_time = (2**attempt) * 2  # Exponential backoff: 2s, 4s, 8s
                print(
                    f"⚠️  Server error, waiting {wait_time}s... (attempt {attempt + 1}/{max_retries})"
                )
                await asyncio.sleep(wait_time)

            # Non-retryable errors
            else:
                print(f"❌ Non-retryable error for question {question.uuid}: {e}")
                return None

    # Max retries exceeded
    print(f"❌ Max retries exceeded for question {question.uuid}")
    return None


async def evaluate_model_on_questions(
    questions: List[ForecastingQuestion],
    model_ids: List[str],
    batch_size: Optional[int] = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    user_prompt_template: str = DEFAULT_USER_PROMPT_TEMPLATE,
    temperature: float = 1.0,
) -> Dict[str, List[ModelPrediction]]:
    """
    Evaluate multiple models on a list of forecasting questions.

    This is the main evaluation function. It:
    1. Auto-tunes batch size per model (if batch_size=None) to max throughput
    2. Processes questions in batches (for rate limiting)
    3. Handles retries automatically
    4. Shows progress bars
    5. Returns predictions for all models

    Args:
        questions: List of ForecastingQuestion objects
        model_ids: List of model identifiers to evaluate
        batch_size: Number of questions to process concurrently.
                   If None, automatically calculates optimal size per model.
        system_prompt: System message for models
        user_prompt_template: User message template
        temperature: Sampling temperature (0.0-1.0+, default 1.0)

    Returns:
        Dict mapping model_id -> list of ModelPrediction objects

    Example:
        # Auto-tune batch size (recommended)
        results = await evaluate_model_on_questions(
            questions,
            ["claude-opus-4-1", "claude-sonnet-4-0"]
        )

        # Manual batch size
        results = await evaluate_model_on_questions(
            questions,
            ["claude-opus-4-1"],
            batch_size=50
        )
    """
    client = ModelClient()
    all_results = {model_id: [] for model_id in model_ids}

    for model_id in model_ids:
        print(f"\n{'='*80}")
        print(f"Evaluating {model_id}")
        print(f"{'='*80}")

        # Calculate optimal batch size for this model if not specified
        if batch_size is None:
            # Estimate average token count from a sample of questions
            sample_size = min(10, len(questions))
            token_estimates = [
                estimate_question_tokens(q) for q in questions[:sample_size]
            ]
            avg_tokens = int(np.mean(token_estimates))

            print(f"Estimated avg input tokens per question: {avg_tokens:,}")

            optimal_batch = calculate_optimal_batch_size(
                model_id, avg_question_tokens=avg_tokens
            )
            limits = get_rate_limits(model_id)
            print(f"Auto-tuned batch size: {optimal_batch}")
            print(
                f"  (RPM: {limits['rpm']}, TPM in: {limits['tpm_in']:,}, TPM out: {limits['tpm_out']:,})"
            )
        else:
            optimal_batch = batch_size
            print(f"Using fixed batch size: {optimal_batch}")

        # Process questions in batches
        for i in range(0, len(questions), optimal_batch):
            batch = questions[i : i + optimal_batch]
            batch_num = i // optimal_batch + 1
            total_batches = (len(questions) - 1) // optimal_batch + 1

            print(f"  Batch {batch_num}/{total_batches}...")

            # Create tasks for batch
            tasks = [
                predict_question_async(
                    q,
                    model_id,
                    client,
                    system_prompt,
                    user_prompt_template,
                    temperature=temperature,
                )
                for q in batch
            ]

            # Run batch concurrently
            batch_results = await asyncio.gather(*tasks)
            all_results[model_id].extend([r for r in batch_results if r is not None])

            # Small delay between batches
            if i + optimal_batch < len(questions):
                await asyncio.sleep(1)

        # Calculate and print accuracy immediately after model completes
        predictions = all_results[model_id]
        valid_predictions = [p for p in predictions if p.actual_outcome is not None]

        if valid_predictions:
            correct = sum(1 for p in valid_predictions if p.correct)
            accuracy = correct / len(valid_predictions)
            print(f"✅ Completed {len(predictions)} predictions for {model_id}")
            print(
                f"   Accuracy: {accuracy:.2%} ({correct}/{len(valid_predictions)} correct)"
            )
        else:
            print(f"✅ Completed {len(predictions)} predictions for {model_id}")

    return all_results


def calculate_metrics(predictions: List[ModelPrediction]) -> Dict:
    """
    Calculate evaluation metrics from predictions.

    Computes:
    - Accuracy: % of correct predictions
    - Brier Score: mean squared error of probabilities (lower is better)
    - AUROC: area under ROC curve (only if both classes present)
    - Log Loss: negative log likelihood (lower is better)
    - Base Rate: proportion of positive outcomes in the dataset

    Args:
        predictions: List of ModelPrediction objects

    Returns:
        Dict with metric values and counts

    Example:
        metrics = calculate_metrics(predictions)
        # {
        #     "accuracy": 0.72,
        #     "brier_score": 0.18,
        #     "auroc": 0.78,
        #     "log_loss": 0.52,
        #     "base_rate": 0.30,  # 30% positive outcomes
        #     "baseline_accuracy": 0.70,  # Best naive strategy: always predict No
        #     "correct_predictions": 720,
        #     "total_predictions": 1000,
        #     "skipped_none_outcomes": 0
        # }
    """
    # Filter out predictions with None actual_outcome
    valid_predictions = [p for p in predictions if p.actual_outcome is not None]

    if not valid_predictions:
        return {
            "accuracy": 0.0,
            "brier_score": 1.0,
            "auroc": None,
            "log_loss": float("inf"),
            "base_rate": None,
            "baseline_accuracy": None,
            "correct_predictions": 0,
            "total_predictions": 0,
            "skipped_none_outcomes": len(predictions),
        }

    predicted_probs = np.array([p.predicted_probability for p in valid_predictions])
    actual_outcomes = np.array(
        [int(p.actual_outcome) for p in valid_predictions], dtype=np.int32
    )

    # Calculate base rate (proportion of positive outcomes)
    base_rate = np.mean(actual_outcomes)

    # Calculate baseline accuracy (best you can do by always predicting one class)
    baseline_accuracy = max(base_rate, 1 - base_rate)

    # Calculate metrics
    correct_predictions = sum(1 for p in valid_predictions if p.correct)
    accuracy = correct_predictions / len(valid_predictions)
    brier_score = brier_score_loss(actual_outcomes, predicted_probs)

    # AUROC requires both classes
    if len(np.unique(actual_outcomes)) > 1:
        auroc = roc_auc_score(actual_outcomes, predicted_probs)
    else:
        auroc = None

    # Log loss requires both classes
    if len(np.unique(actual_outcomes)) > 1:
        logloss = log_loss(actual_outcomes, predicted_probs)
    else:
        logloss = None

    return {
        "accuracy": float(accuracy),
        "brier_score": float(brier_score),
        "auroc": float(auroc) if auroc is not None else None,
        "log_loss": float(logloss) if logloss is not None else None,
        "base_rate": float(base_rate),
        "baseline_accuracy": float(baseline_accuracy),
        "correct_predictions": int(correct_predictions),
        "total_predictions": len(valid_predictions),
        "skipped_none_outcomes": len(predictions) - len(valid_predictions),
    }


def bootstrap_confidence_interval(
    data, metric_fn, n_bootstrap=10000, confidence_level=0.95
):
    """
    Compute bootstrap confidence interval for a metric.

    Args:
        data: Array of predictions and ground truth
        metric_fn: Function that computes metric from data
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)

    Returns:
        (lower_bound, upper_bound, point_estimate)
    """
    bootstrap_metrics = []
    n_samples = len(data)

    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        sample_data = data[indices]
        bootstrap_metrics.append(metric_fn(sample_data))

    # Calculate confidence interval
    alpha = 1 - confidence_level
    lower = np.percentile(bootstrap_metrics, 100 * alpha / 2)
    upper = np.percentile(bootstrap_metrics, 100 * (1 - alpha / 2))
    point_estimate = metric_fn(data)

    return lower, upper, point_estimate


def compute_confidence_intervals(
    predictions: List[ModelPrediction], confidence_level=0.95
):
    """
    Compute bootstrap confidence intervals for all metrics.

    Args:
        predictions: List of ModelPrediction objects
        confidence_level: Confidence level (e.g., 0.95 for 95% CI)

    Returns:
        Dict with confidence intervals for each metric
    """
    # Create array with predictions and ground truth
    pred_data = np.array(
        [[p.predicted_probability, p.actual_outcome] for p in predictions]
    )

    # Brier score
    def brier_fn(data):
        preds, truth = data[:, 0], data[:, 1]
        return np.mean((preds - truth) ** 2)

    # Accuracy (using 0.5 threshold)
    def accuracy_fn(data):
        preds, truth = data[:, 0], data[:, 1]
        return np.mean((preds > 0.5) == (truth > 0.5))

    # Log score (careful with 0/1 predictions)
    def log_score_fn(data):
        preds, truth = data[:, 0], data[:, 1]
        # Clip predictions to avoid log(0)
        preds = np.clip(preds, 1e-10, 1 - 1e-10)
        return -np.mean(truth * np.log(preds) + (1 - truth) * np.log(1 - preds))

    # AUROC
    def auroc_fn(data):
        preds, truth = data[:, 0], data[:, 1]
        try:
            return roc_auc_score(truth, preds)
        except ValueError:
            return np.nan

    intervals = {}

    # Compute confidence intervals for each metric
    lower, upper, point = bootstrap_confidence_interval(
        pred_data, brier_fn, confidence_level=confidence_level
    )
    intervals["brier_score"] = (lower, upper)

    lower, upper, point = bootstrap_confidence_interval(
        pred_data, accuracy_fn, confidence_level=confidence_level
    )
    intervals["accuracy"] = (lower, upper)

    lower, upper, point = bootstrap_confidence_interval(
        pred_data, log_score_fn, confidence_level=confidence_level
    )
    intervals["log_score"] = (lower, upper)

    # Only compute AUROC if we have both classes
    if len(np.unique(pred_data[:, 1])) > 1:
        lower, upper, point = bootstrap_confidence_interval(
            pred_data, auroc_fn, confidence_level=confidence_level
        )
        intervals["auroc"] = (lower, upper)

    return intervals


def save_evaluation_results(
    all_results: Dict[str, List[ModelPrediction]],
    output_dir: Path,
    experiment_name: str = "eval",
) -> None:
    """
    Save evaluation results to disk.

    Creates:
    - predictions_{model_id}.jsonl - one file per model with all predictions
    - model_comparison.json - metrics for all models
    - summary.txt - human-readable summary

    Args:
        all_results: Dict from evaluate_model_on_questions
        output_dir: Directory to save results
        experiment_name: Name for this evaluation run

    Example:
        save_evaluation_results(
            results,
            Path("experiments/eval-fred"),
            experiment_name="fred_diff20"
        )
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Saving results to {output_dir}")
    print(f"{'='*80}")

    # Calculate metrics for each model
    model_metrics = {}
    for model_id, predictions in all_results.items():
        metrics = calculate_metrics(predictions)
        model_metrics[model_id] = metrics

        # Save predictions JSONL
        model_name = model_id.replace("/", "_").replace(".", "_").replace("-", "_")
        pred_file = output_dir / f"predictions_{model_name}.jsonl"
        with open(pred_file, "w") as f:
            for pred in predictions:
                f.write(json.dumps(pred.to_dict()) + "\n")
        print(f"✅ Saved {len(predictions)} predictions to {pred_file.name}")

    # Save comparison metrics JSON
    comparison_file = output_dir / f"{experiment_name}_model_comparison.json"
    with open(comparison_file, "w") as f:
        json.dump(model_metrics, f, indent=2)
    print(f"✅ Saved metrics comparison to {comparison_file.name}")

    # Save human-readable summary
    summary_file = output_dir / f"{experiment_name}_summary.txt"
    with open(summary_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write(f"EVALUATION SUMMARY: {experiment_name}\n")
        f.write("=" * 80 + "\n\n")

        # Display base rate (should be same across all models)
        base_rate = next(iter(model_metrics.values()))["base_rate"]
        baseline_accuracy = next(iter(model_metrics.values()))["baseline_accuracy"]
        if base_rate is not None:
            f.write(f"Dataset Base Rate: {base_rate:.2%}\n")
            f.write(f"  (Proportion of questions resolving to 'Yes')\n")
            f.write(f"Baseline Accuracy: {baseline_accuracy:.2%}\n")
            f.write(f"  (Naive strategy: always predict majority class)\n\n")

        for model_id in all_results.keys():
            metrics = model_metrics[model_id]
            f.write(f"{model_id}:\n")
            f.write(f"  Accuracy:           {metrics['accuracy']:.2%}\n")
            f.write(f"  Brier Score:        {metrics['brier_score']:.4f}\n")
            if metrics["auroc"] is not None:
                f.write(f"  AUROC:              {metrics['auroc']:.4f}\n")
            f.write(f"  Log Loss:           {metrics['log_loss']:.4f}\n")
            f.write(
                f"  Correct:            {metrics['correct_predictions']}/{metrics['total_predictions']}\n"
            )
            f.write("\n")

        # Best model by each metric
        f.write("=" * 80 + "\n")
        f.write("BEST MODEL BY METRIC\n")
        f.write("=" * 80 + "\n\n")

        best_accuracy = max(model_metrics.items(), key=lambda x: x[1]["accuracy"])
        f.write(
            f"Best Accuracy:      {best_accuracy[0]} ({best_accuracy[1]['accuracy']:.2%})\n"
        )

        best_brier = min(model_metrics.items(), key=lambda x: x[1]["brier_score"])
        f.write(
            f"Best Brier Score:   {best_brier[0]} ({best_brier[1]['brier_score']:.4f})\n"
        )

        if any(m["auroc"] is not None for m in model_metrics.values()):
            best_auroc = max(
                model_metrics.items(),
                key=lambda x: x[1]["auroc"] if x[1]["auroc"] is not None else 0,
            )
            f.write(
                f"Best AUROC:         {best_auroc[0]} ({best_auroc[1]['auroc']:.4f})\n"
            )

        best_logloss = min(model_metrics.items(), key=lambda x: x[1]["log_loss"])
        f.write(
            f"Best Log Loss:      {best_logloss[0]} ({best_logloss[1]['log_loss']:.4f})\n"
        )

    print(f"✅ Saved summary to {summary_file.name}")
    print(f"\n{'='*80}")
    print("Evaluation complete!")
    print(f"{'='*80}")

    return model_metrics


async def evaluate_and_plot(
    questions: List[ForecastingQuestion],
    model_ids: List[str],
    output_dir: Optional[Path] = None,
    experiment_name: str = "eval",
    batch_size: Optional[int] = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    user_prompt_template: str = DEFAULT_USER_PROMPT_TEMPLATE,
    save_plot: bool = True,
    metrics_to_plot: List[str] = ["accuracy", "brier_score", "auroc"],
    temperature: float = 1.0,
):
    """
    All-in-one: evaluate models, calculate metrics, plot comparison, save results.

    This is a high-level convenience function that combines the full workflow:
    1. Run evaluate_model_on_questions() on all models
    2. Calculate metrics for each model
    3. Generate comparison plot
    4. Save everything to output_dir

    Args:
        questions: List of ForecastingQuestion objects to evaluate
        model_ids: List of model identifiers to compare
        output_dir: Directory to save results. If None, uses a temp directory.
        experiment_name: Name for this evaluation run
        batch_size: Batch size for API calls (None = auto-tune per model)
        system_prompt: System message for models
        user_prompt_template: User message template
        save_plot: Whether to save the comparison plot
        metrics_to_plot: Which metrics to include in plot
        temperature: Sampling temperature (0.0-1.0+, default 1.0)

    Returns:
        Tuple of (predictions_dict, metrics_dict) where:
            - predictions_dict: Dict[str, List[ModelPrediction]]
            - metrics_dict: Dict[str, Dict] with calculated metrics per model

    Example:
        >>> questions = load_fred_observations(...)
        >>> predictions, metrics = await evaluate_and_plot(
        ...     questions,
        ...     ["claude-sonnet-4-5-20250929", "claude-opus-4-1-20250805"],
        ...     output_dir=Path("experiments/fred-eval"),
        ...     experiment_name="fred_diff20"
        ... )
        >>> # Results saved to:
        >>> # - experiments/fred-eval/predictions_*.jsonl
        >>> # - experiments/fred-eval/fred_diff20_model_comparison.json
        >>> # - experiments/fred-eval/fred_diff20_comparison.png
        >>> # - experiments/fred-eval/fred_diff20_summary.txt
    """
    from src.plotting import plot_model_comparison

    # Default output directory
    if output_dir is None:
        from src.config import RESULTS_DIR

        output_dir = RESULTS_DIR / experiment_name

    output_dir = Path(output_dir)

    print(f"\n{'='*80}")
    print(f"EVALUATION AND PLOTTING PIPELINE")
    print(f"{'='*80}")
    print(f"Experiment: {experiment_name}")
    print(f"Models: {', '.join(model_ids)}")
    print(f"Questions: {len(questions)}")
    print(f"Output: {output_dir}")
    print(f"{'='*80}\n")

    # Step 1: Run evaluation
    print("Step 1/4: Running model evaluations...")
    all_results = await evaluate_model_on_questions(
        questions=questions,
        model_ids=model_ids,
        batch_size=batch_size,
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
        temperature=temperature,
    )

    # Step 2: Calculate metrics and confidence intervals
    print("\nStep 2/4: Calculating metrics and confidence intervals...")
    model_metrics = {}
    confidence_intervals = {}
    for model_id, predictions in all_results.items():
        metrics = calculate_metrics(predictions)
        intervals = compute_confidence_intervals(predictions)
        model_metrics[model_id] = metrics
        confidence_intervals[model_id] = intervals
        print(f"  {model_id}:")
        print(
            f"    Accuracy: {metrics['accuracy']:.2%} [{intervals['accuracy'][0]:.2%}, {intervals['accuracy'][1]:.2%}]"
        )
        print(
            f"    Brier Score: {metrics['brier_score']:.4f} [{intervals['brier_score'][0]:.4f}, {intervals['brier_score'][1]:.4f}]"
        )
        if metrics["auroc"] is not None:
            print(
                f"    AUROC: {metrics['auroc']:.4f} [{intervals['auroc'][0]:.4f}, {intervals['auroc'][1]:.4f}]"
            )

    # Step 3: Generate plot
    if save_plot:
        print("\nStep 3/4: Generating comparison plot...")
        plot_path = output_dir / f"{experiment_name}_comparison.png"
        # Extract baseline accuracy (should be same for all models)
        baseline_accuracy = next(iter(model_metrics.values())).get("baseline_accuracy")
        plot_model_comparison(
            model_metrics,
            output_path=plot_path,
            metrics_to_plot=metrics_to_plot,
            title=f"Model Comparison: {experiment_name}",
            baseline_accuracy=baseline_accuracy,
            confidence_intervals=confidence_intervals,
        )
    else:
        print("\nStep 3/4: Skipping plot (save_plot=False)")

    # Step 4: Save all results
    print("\nStep 4/4: Saving results...")
    save_evaluation_results(all_results, output_dir, experiment_name)

    print(f"\n{'='*80}")
    print("PIPELINE COMPLETE!")
    print(f"{'='*80}")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*80}\n")

    return all_results, model_metrics
