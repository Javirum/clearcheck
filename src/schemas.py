"""Output schemas and LangGraph state definitions for NOPE."""

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
    scam_assessment: Optional[ScamAssessment] = Field(default=None, description="Scam assessment if fraud indicators detected")


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


class ImageAnalysisResult(BaseModel):
    description: str = Field(description="What the image shows")
    ai_generation_signals: list[str] = Field(default_factory=list, description="Signs of AI generation")
    manipulation_signals: list[str] = Field(default_factory=list, description="Signs of manipulation")
    authenticity_assessment: str = Field(default="", description="Overall authenticity assessment")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence in the assessment")


# ---------------------------------------------------------------------------
# Scam detection schemas
# ---------------------------------------------------------------------------

class ScamAnalysisResult(BaseModel):
    scam_likelihood: float = Field(ge=0.0, le=1.0, description="How likely this is a scam (0-1)")
    scam_type: Optional[str] = Field(default=None, description="Detected scam category")
    red_flags_detected: list[str] = Field(default_factory=list, description="Specific red flags found")
    urgency_score: float = Field(default=0.0, ge=0.0, le=1.0, description="How much urgency/pressure language is present")


class URLSafetyVerdict(BaseModel):
    url: str = Field(description="The URL that was checked")
    is_safe: bool = Field(description="Whether the URL appears safe")
    threats: list[str] = Field(default_factory=list, description="Threat types detected")
    expanded_url: Optional[str] = Field(default=None, description="Expanded URL if shortened")
    details: str = Field(default="", description="Additional details about the URL")


class URLSafetyResult(BaseModel):
    urls_found: list[str] = Field(default_factory=list, description="URLs extracted from the text")
    results: list[URLSafetyVerdict] = Field(default_factory=list, description="Per-URL safety verdicts")
    any_unsafe: bool = Field(default=False, description="True if any URL was flagged as unsafe")


class ScamAssessment(BaseModel):
    is_likely_scam: bool = Field(description="Whether this appears to be a scam")
    scam_type: Optional[str] = Field(default=None, description="Type of scam detected")
    scam_confidence: float = Field(ge=0.0, le=1.0, description="Confidence in scam detection")
    red_flags: list[str] = Field(default_factory=list, description="Red flags identified")


class GatheredEvidence(BaseModel):
    pinecone_results: list[PineconeResult] = Field(default_factory=list)
    tavily_results: list[TavilyResult] = Field(default_factory=list)
    factcheck_results: list[FactCheckResult] = Field(default_factory=list)
    image_analysis: Optional[ImageAnalysisResult] = Field(default=None, description="Image analysis if an image was provided")
    scam_analysis: Optional[ScamAnalysisResult] = Field(default=None, description="Scam pattern analysis results")
    url_safety: Optional[URLSafetyResult] = Field(default=None, description="URL safety check results")
    errors: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    claim: str
    image_b64: str
    media_type: str
    evidence: GatheredEvidence
    verdict: Verdict
    validation: ValidationResult
    final_verdict: Verdict


# ---------------------------------------------------------------------------
# Image checker schemas
# ---------------------------------------------------------------------------

class ImageVerdictLevel(str, Enum):
    AUTHENTIC = "authentic"
    AI_GENERATED = "ai_generated"
    MANIPULATED = "manipulated"
    OUT_OF_CONTEXT = "out_of_context"
    UNCERTAIN = "uncertain"


IMAGE_VERDICT_EMOJI = {
    ImageVerdictLevel.AUTHENTIC: "✅",
    ImageVerdictLevel.AI_GENERATED: "🤖",
    ImageVerdictLevel.MANIPULATED: "🖌️",
    ImageVerdictLevel.OUT_OF_CONTEXT: "🔄",
    ImageVerdictLevel.UNCERTAIN: "❓",
}

IMAGE_VERDICT_LABEL = {
    ImageVerdictLevel.AUTHENTIC: "Looks Like the Real Deal",
    ImageVerdictLevel.AI_GENERATED: "Probably AI-Made",
    ImageVerdictLevel.MANIPULATED: "This Might Be Edited",
    ImageVerdictLevel.OUT_OF_CONTEXT: "Real Image, Wrong Story",
    ImageVerdictLevel.UNCERTAIN: "We're Not Sure Yet",
}


class ReverseImageResult(BaseModel):
    title: str = Field(description="Title of the matching page")
    url: str = Field(description="URL where the image was found")
    content: str = Field(description="Snippet or description from the page")
    score: float = Field(default=0.0, description="Relevance score")


class ImageMetadata(BaseModel):
    format: Optional[str] = Field(default=None, description="Image format (JPEG, PNG, etc.)")
    width: Optional[int] = Field(default=None, description="Image width in pixels")
    height: Optional[int] = Field(default=None, description="Image height in pixels")
    exif: dict = Field(default_factory=dict, description="EXIF metadata if available")
    file_size_bytes: Optional[int] = Field(default=None, description="File size in bytes")


class ImageEvidence(BaseModel):
    reverse_search_results: list[ReverseImageResult] = Field(default_factory=list)
    metadata: Optional[ImageMetadata] = Field(default=None)
    errors: list[str] = Field(default_factory=list)


class ImageVerdict(BaseModel):
    description: str = Field(description="Brief description of the image content")
    verdict: ImageVerdictLevel = Field(description="Image authenticity verdict")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0-1")
    explanation: str = Field(description="Plain-language explanation at grade 6-8 reading level")
    ai_generation_signals: list[str] = Field(default_factory=list, description="Signs of AI generation detected")
    manipulation_signals: list[str] = Field(default_factory=list, description="Signs of digital manipulation detected")
    context_analysis: str = Field(default="", description="Analysis of whether the image is used in proper context")
    sources: list[SourceCitation] = Field(default_factory=list, description="Sources consulted")
    educational_tip: str = Field(description="Tip to help users spot fake images")
    reasoning_chain: str = Field(description="Step-by-step reasoning for the audit log")


class ImageValidationResult(BaseModel):
    is_valid: bool = Field(description="Whether the verdict passes all quality checks")
    issues: list[str] = Field(default_factory=list, description="List of issues found")
    corrected_verdict: Optional[ImageVerdict] = Field(
        default=None,
        description="Corrected verdict if the original had issues",
    )


class ImageAgentState(TypedDict, total=False):
    image_b64: str
    media_type: str
    user_context: str
    evidence: ImageEvidence
    verdict: ImageVerdict
    validation: ImageValidationResult
    final_verdict: ImageVerdict
