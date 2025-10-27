import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class ForecastingQuestion:
    """
    A forecasting question matching Anthropic's required Parquet schema.

    Attributes:
        question: Full question text including resolution criteria
        creation_date: ISO format date when question was created (nullable)
        resolution_date: ISO format date when question can be resolved (nullable)
        resolution: None if unresolved; 1.0 for YES; 0.0 for NO; or a real number
        uuid: UUID string for tracking
        metadata: JSON string containing raw source data and metadata
        resolution_evidence: Human/Claude-readable explanation of creation/resolution
        pipeline: Unique identifier for the pipeline that created this question
        numerical_resolution: If True, resolution is a real number (not binary)
    """

    question: str
    creation_date: Optional[str]
    resolution_date: Optional[str]
    resolution: Optional[float]
    uuid: str
    metadata: str  # JSON string
    resolution_evidence: Optional[str]
    pipeline: str
    numerical_resolution: bool = False


@dataclass
class ArxivPaper:
    """ArXiv paper with metadata and content."""

    arxiv_id: str
    original_arxiv_id: str
    title: str
    full_text: str
    authors: List[str]
    published_timestamp: str
    categories: List[str]
    pdf_url: str
    citations: int
    citation_timestamp: str
    paper_type: Optional[str] = None  # method, survey, benchmark, resource