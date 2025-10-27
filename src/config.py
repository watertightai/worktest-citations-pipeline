"""
Configuration for the citations pipeline.

Defines paths for data directories and results.
"""

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
ARXIV_DATA_DIR = DATA_DIR / "arxiv_papers"
FRED_DATA_DIR = DATA_DIR / "fred_indicators"

# Results directory
RESULTS_DIR = PROJECT_ROOT / "evaluation_results"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
ARXIV_DATA_DIR.mkdir(exist_ok=True)
FRED_DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)