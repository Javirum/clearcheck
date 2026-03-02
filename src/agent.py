"""LangGraph agent for ClearCheck: Claude analysis + LLM validation."""

from __future__ import annotations

import json

from langchain_anthropic import ChatAnthropic
from langgraph.graph import END, StateGraph

from src.config import ANTHROPIC_API_KEY, LLM_MODEL, LLM_MODEL_FAST
from src.evidence import gather_evidence
from src.schemas import (
    AgentState,
    GatheredEvidence,
    ValidationResult,
    Verdict,
    VerdictLevel,
)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

ANALYSIS_PROMPT = """\
You are ClearCheck, a fact-checking assistant designed for older adults (55+).
Your job is to analyze a claim using evidence from multiple sources and return a structured verdict.

## CLAIM TO CHECK
{claim}

## EVIDENCE FROM KNOWLEDGE BASE (Pinecone)
{pinecone_evidence}

## EVIDENCE FROM WEB SEARCH (Tavily)
{tavily_evidence}

## EVIDENCE FROM FACT-CHECK ORGANIZATIONS (Google Fact Check API)
{factcheck_evidence}

## ERRORS DURING EVIDENCE GATHERING
{errors}

## INSTRUCTIONS

1. Analyze ALL evidence carefully. Cross-reference sources.
2. Determine a verdict:
   - "true" — the claim is accurate and supported by reliable evidence
   - "misleading" — the claim contains some truth but is missing important context or is exaggerated
   - "false" — the claim is factually incorrect based on available evidence
   - "uncertain" — there is insufficient evidence to make a confident determination
3. If evidence is insufficient or contradictory, you MUST return "uncertain". Do NOT guess.
4. Write your explanation at a grade 6-8 reading level. Use short sentences. Avoid jargon.
5. Include a confidence score (0.0 to 1.0). Be honest about uncertainty.
6. Cite specific sources with URLs from the evidence provided. Only cite sources that actually appear in the evidence — NEVER invent URLs.
7. Include an educational tip that helps the reader think critically about this type of content.
8. Provide a step-by-step reasoning chain for the audit log.

## OUTPUT FORMAT

Return a valid JSON object with exactly these fields:
{{
  "claim": "<the claim restated clearly>",
  "verdict": "<true|misleading|false|uncertain>",
  "confidence": <0.0 to 1.0>,
  "explanation": "<plain-language explanation, 2-4 sentences, grade 6-8 reading level>",
  "sources": [
    {{"name": "<source name>", "url": "<actual URL from evidence>", "snippet": "<relevant excerpt>"}}
  ],
  "educational_tip": "<one practical tip for spotting this type of misinformation>",
  "reasoning_chain": "<step-by-step reasoning>"
}}

Return ONLY the JSON object, no other text."""

VALIDATION_PROMPT = """\
You are a quality validator for a fact-checking system. Review this verdict for quality and accuracy.

## ORIGINAL CLAIM
{claim}

## VERDICT TO VALIDATE
{verdict_json}

## AVAILABLE EVIDENCE
{evidence_summary}

## VALIDATION CHECKS

Check each of the following and report any issues:

1. **Verdict consistency**: Does the verdict level (true/misleading/false/uncertain) match the explanation?
2. **Source grounding**: Are all cited sources actually present in the available evidence? Are URLs real (not fabricated)?
3. **Confidence calibration**: Is the confidence score appropriate given the evidence? Low evidence should mean low confidence.
4. **Explanation clarity**: Is the explanation written at a grade 6-8 reading level? Is it clear and helpful?
5. **Completeness**: Does the response include all required fields?
6. **Uncertainty handling**: If evidence is insufficient, is the verdict "uncertain" rather than a forced conclusion?

## OUTPUT FORMAT

Return a valid JSON object:
{{
  "is_valid": <true or false>,
  "issues": ["<issue 1>", "<issue 2>"],
  "corrected_verdict": null or <corrected verdict JSON if changes are needed>
}}

If the verdict passes all checks, return is_valid=true with an empty issues list and corrected_verdict=null.
If there are issues, set is_valid=false, list them, and provide a corrected_verdict with the fixes applied.

Return ONLY the JSON object, no other text."""


# ---------------------------------------------------------------------------
# Helper: format evidence for prompt
# ---------------------------------------------------------------------------

def _format_evidence(evidence: GatheredEvidence) -> dict[str, str]:
    pinecone_text = ""
    if evidence.pinecone_results:
        for r in evidence.pinecone_results:
            pinecone_text += f"- [Score: {r.score:.2f}] {r.text}\n  Sources: {r.source}\n  Details: {r.metadata.get('explanation', '')}\n\n"
    else:
        pinecone_text = "No matches found in knowledge base.\n"

    tavily_text = ""
    if evidence.tavily_results:
        for r in evidence.tavily_results:
            tavily_text += f"- [{r.title}]({r.url})\n  {r.content[:300]}\n\n"
    else:
        tavily_text = "No web search results found.\n"

    factcheck_text = ""
    if evidence.factcheck_results:
        for r in evidence.factcheck_results:
            factcheck_text += f"- Publisher: {r.publisher} | Rating: {r.rating}\n  Claim: {r.claim_text}\n  URL: {r.url}\n\n"
    else:
        factcheck_text = "No published fact-checks found.\n"

    errors_text = "\n".join(evidence.errors) if evidence.errors else "None"

    return {
        "pinecone_evidence": pinecone_text,
        "tavily_evidence": tavily_text,
        "factcheck_evidence": factcheck_text,
        "errors": errors_text,
    }


def _parse_json_response(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown code blocks."""
    text = text.strip()
    if text.startswith("```"):
        # Remove markdown code fences
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    return json.loads(text)


# ---------------------------------------------------------------------------
# LangGraph nodes
# ---------------------------------------------------------------------------

def gather_evidence_node(state: AgentState) -> AgentState:
    """Gather evidence from all three sources in parallel."""
    claim = state["claim"]
    evidence = gather_evidence(claim)
    return {"evidence": evidence}


def analyze_node(state: AgentState) -> AgentState:
    """Claude analyzes all evidence and produces a structured verdict."""
    llm = ChatAnthropic(
        model=LLM_MODEL,
        api_key=ANTHROPIC_API_KEY,
        max_tokens=2000,
        temperature=0,
    )

    evidence = state["evidence"]
    formatted = _format_evidence(evidence)

    prompt = ANALYSIS_PROMPT.format(claim=state["claim"], **formatted)
    response = llm.invoke(prompt)

    try:
        verdict_data = _parse_json_response(response.content)
        verdict = Verdict(**verdict_data)
    except Exception as e:
        # Fallback: return uncertain verdict if parsing fails
        verdict = Verdict(
            claim=state["claim"],
            verdict=VerdictLevel.UNCERTAIN,
            confidence=0.0,
            explanation="I wasn't able to complete the analysis. Please try again or rephrase your question.",
            sources=[],
            educational_tip="When a fact-check tool can't verify something, try searching for the claim on trusted news sites yourself.",
            reasoning_chain=f"Analysis failed to produce valid output: {e}",
        )

    return {"verdict": verdict}


def validate_node(state: AgentState) -> AgentState:
    """Secondary LLM call validates the verdict for quality."""
    llm = ChatAnthropic(
        model=LLM_MODEL_FAST,
        api_key=ANTHROPIC_API_KEY,
        max_tokens=2000,
        temperature=0,
    )

    evidence = state["evidence"]
    verdict = state["verdict"]
    formatted = _format_evidence(evidence)
    evidence_summary = "\n".join(
        f"[{k}]\n{v}" for k, v in formatted.items() if k != "errors"
    )

    prompt = VALIDATION_PROMPT.format(
        claim=state["claim"],
        verdict_json=verdict.model_dump_json(indent=2),
        evidence_summary=evidence_summary,
    )
    response = llm.invoke(prompt)

    try:
        validation_data = _parse_json_response(response.content)
        # Parse corrected verdict if present
        corrected = None
        if validation_data.get("corrected_verdict"):
            corrected = Verdict(**validation_data["corrected_verdict"])
        validation = ValidationResult(
            is_valid=validation_data["is_valid"],
            issues=validation_data.get("issues", []),
            corrected_verdict=corrected,
        )
    except Exception:
        validation = ValidationResult(is_valid=True, issues=[])

    # Use corrected verdict if validation found issues
    final = validation.corrected_verdict if validation.corrected_verdict else verdict
    return {"validation": validation, "final_verdict": final}


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("gather_evidence", gather_evidence_node)
    graph.add_node("analyze", analyze_node)
    graph.add_node("validate", validate_node)

    graph.set_entry_point("gather_evidence")
    graph.add_edge("gather_evidence", "analyze")
    graph.add_edge("analyze", "validate")
    graph.add_edge("validate", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def check_claim(claim: str) -> tuple[Verdict, ValidationResult, GatheredEvidence]:
    """Run the full verification pipeline on a claim.

    Returns (final_verdict, validation_result, gathered_evidence).
    """
    app = build_graph()
    result = app.invoke({"claim": claim})
    return (
        result["final_verdict"],
        result.get("validation", ValidationResult(is_valid=True)),
        result["evidence"],
    )
