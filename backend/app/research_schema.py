"""Evidence contracts for documented ML applications and tutorial cases."""
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl


class Evidence(BaseModel):
    field: str = Field(min_length=1)
    locator: str = Field(min_length=1, description="Section or code symbol supporting the claim")
    summary: str = Field(min_length=1, description="Paraphrase, not an invented quotation")


class ResearchCase(BaseModel):
    title: str = Field(min_length=3, max_length=300)
    source_url: HttpUrl
    source_kind: Literal["tutorial", "paper", "field_report"]
    situation: str = Field(min_length=10)
    domain: str
    task_type: Literal["classification", "regression", "clustering", "forecasting", "other"]
    model_family: Literal["ml", "dl"] = "ml"
    models: list[str] = Field(min_length=1)
    model_reason: str | None = None
    dataset_name: str | None = None
    dataset_url: HttpUrl | None = None
    dataset_access: Literal["public", "restricted", "synthetic", "unknown"] = "unknown"
    dataset_license: str | None = None
    features: list[str] = Field(default_factory=list)
    preparation: list[str] = Field(default_factory=list)
    evaluation: list[str] = Field(default_factory=list)
    reported_results: str | None = None
    authors: list[str] = Field(default_factory=list)
    published_date: str | None = None
    evidence: list[Evidence] = Field(min_length=1)
    unknowns: list[str] = Field(default_factory=list)
    confidence: Literal["curated", "extracted_unreviewed"] = "extracted_unreviewed"


class RefreshRequest(BaseModel):
    force: bool = False
