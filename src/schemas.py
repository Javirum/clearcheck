"""Output schemas and LangGraph state definitions for ClearCheck."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# Verdict enum
# ---------------------------------------------------------------------------

class VerdictLevel(str, Enum):
    TRUE = "true"
    MISLEADING = "misleading"
    FALSE = "false"
    UNCERTAIN = "uncertain"


VERDICT_COLORS = {
    VerdictLevel.TRUE: "green",
    VerdictLevel.MISLEADING: "yellow",
    VerdictLevel.FALSE: "red",
    VerdictLevel.UNCERTAIN: "yellow",
}


# ---------------------------------------------------------------------------
# Source citation
# ---------------------------------------------------------------------------

class SourceCitation(BaseModel):
    name: str = Field(description="Name of the source (e.g. 'Snopes', 'Reuters')")
    url: str = Field(description="URL to the source article")
    snippet: str = Field(description="Brief relevant excerpt from the source")


# ---------------------------------------------------------------------------
# Structured verdict (returned by the LangGraph agent)
# ---------------------------------------------------------------------------

class Verdict(BaseModel):
    claim: str = Field(description="The claim that was checked, restated clearly")
    verdict: VerdictLevel = Field(description="Trust verdict: true, misleading, false, or uncertain")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0-1")
    explanation: str = Field(description="Plain-language explanation at grade 6-8 reading level")
    sources: list[SourceCitation] = Field(description="Sources consulted with citations")
    educational_tip: str = Field(description="Educational tip to help the user think critically")
    reasoning_chain: str = Field(description="Step-by-step reasoning for the audit log")


# ---------------------------------------------------------------------------
# LLM validation result
# ---------------------------------------------------------------------------

class ValidationResult(BaseModel):
    is_valid: bool = Field(description="Whether the verdict passes all quality checks")
    issues: list[str] = Field(default_factory=list, description="List of issues found")
    corrected_verdict: Optional[Verdict] = Field(
        default=None,
        description="Corrected verdict if the original had issues",
    )


# ---------------------------------------------------------------------------
# Evidence containers
# ---------------------------------------------------------------------------

class PineconeResult(BaseModel):
    text: str
    source: str
    score: float
    metadata: dict = Field(default_factory=dict)


class TavilyResult(BaseModel):
    title: str
    url: str
    content: str
    score: float


class FactCheckResult(BaseModel):
    claim_text: str
    publisher: str
    url: str
    rating: str


class GatheredEvidence(BaseModel):
    pinecone_results: list[PineconeResult] = Field(default_factory=list)
    tavily_results: list[TavilyResult] = Field(default_factory=list)
    factcheck_results: list[FactCheckResult] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    claim: str
    evidence: GatheredEvidence
    verdict: Verdict
    validation: ValidationResult
    final_verdict: Verdict
