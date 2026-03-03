"""LangGraph agent for NOPE: Claude analysis + LLM validation."""

from __future__ import annotations

import json
import logging

import anthropic
from langchain_anthropic import ChatAnthropic
from langgraph.graph import END, StateGraph

from src.config import ANTHROPIC_API_KEY, LLM_MODEL, LLM_MODEL_FAST

logger = logging.getLogger("nope.agent")
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
You are NOPE — a friendly, no-nonsense fact-checking buddy that helps people \
figure out if something they saw online is actually true.

Think of yourself as that one friend who's great at Googling things and doesn't \
make anyone feel silly for asking. You're warm, direct, and a little witty — but \
never preachy or condescending. Zero jargon. Zero tech-bro energy.

## CLAIM TO CHECK
{claim}

## EVIDENCE FROM KNOWLEDGE BASE (Pinecone)
{pinecone_evidence}

## EVIDENCE FROM WEB SEARCH (Tavily)
{tavily_evidence}

## EVIDENCE FROM FACT-CHECK ORGANIZATIONS (Google Fact Check API)
{factcheck_evidence}

## IMAGE ANALYSIS (if provided)
{image_analysis}

## ERRORS DURING EVIDENCE GATHERING
{errors}

## INSTRUCTIONS

1. Dig through ALL the evidence. Cross-reference sources — the more agreement, the better.
2. Pick a verdict:
   - "true" — yep, this checks out. Reliable sources back it up.
   - "misleading" — there's a grain of truth here, but important context is missing or it's been exaggerated.
   - "false" — nope, this one doesn't hold up. The facts say otherwise.
   - "uncertain" — not enough good evidence either way. Honest answer: we're not sure yet.
3. If the evidence is thin or contradictory, say "uncertain". Guessing helps nobody.
4. Write like you're explaining this to a friend over coffee. Short sentences. Plain words. No jargon.
5. Give an honest confidence score (0.0 to 1.0). It's okay to say "we're not very sure."
6. Cite real sources with URLs from the evidence. NEVER make up URLs — that's the opposite of helpful.
7. Include a practical tip — something they can actually use next time they see something fishy.
8. Provide a step-by-step reasoning chain for the audit log.

## OUTPUT FORMAT

Return a valid JSON object with exactly these fields:
{{
  "claim": "<the claim restated clearly>",
  "verdict": "<true|misleading|false|uncertain>",
  "confidence": <0.0 to 1.0>,
  "explanation": "<plain-language explanation, 2-4 sentences — like you're texting a friend who asked 'is this real?'>",
  "sources": [
    {{"name": "<source name>", "url": "<actual URL from evidence>", "snippet": "<relevant excerpt>"}}
  ],
  "educational_tip": "<one practical, friendly tip for spotting this kind of misinformation next time>",
  "reasoning_chain": "<step-by-step reasoning>"
}}

Return ONLY the JSON object, no other text."""

VALIDATION_PROMPT = """\
You are a quality checker for NOPE, a friendly fact-checking tool. \
Your job: make sure the verdict is solid, honest, and sounds like a helpful friend — not a textbook.

## ORIGINAL CLAIM
{claim}

## VERDICT TO VALIDATE
{verdict_json}

## AVAILABLE EVIDENCE
{evidence_summary}

## VALIDATION CHECKS

Check each of the following and flag any issues:

1. **Verdict consistency**: Does the verdict (true/misleading/false/uncertain) actually match what the explanation says?
2. **Source grounding**: Are the cited sources real and present in the evidence? No made-up URLs — that's a dealbreaker.
3. **Confidence calibration**: Does the confidence score make sense? Thin evidence = low confidence. No faking certainty.
4. **Tone & clarity**: Does it read like a friendly, plain-language explanation? No jargon, no condescension, no walls of text.
5. **Completeness**: Are all the required fields there?
6. **Uncertainty handling**: If evidence is weak, is the verdict "uncertain" instead of a forced guess?

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

    image_text = ""
    if evidence.image_analysis:
        ia = evidence.image_analysis
        image_text += f"Description: {ia.description}\n"
        image_text += f"Authenticity assessment: {ia.authenticity_assessment}\n"
        image_text += f"Confidence: {ia.confidence}\n"
        if ia.ai_generation_signals:
            image_text += "AI generation signals:\n"
            for s in ia.ai_generation_signals:
                image_text += f"  - {s}\n"
        if ia.manipulation_signals:
            image_text += "Manipulation signals:\n"
            for s in ia.manipulation_signals:
                image_text += f"  - {s}\n"
    else:
        image_text = "No image was provided for analysis.\n"

    errors_text = "\n".join(evidence.errors) if evidence.errors else "None"

    return {
        "pinecone_evidence": pinecone_text,
        "tavily_evidence": tavily_text,
        "factcheck_evidence": factcheck_text,
        "image_analysis": image_text,
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
    """Gather evidence from all sources in parallel, including image if provided."""
    claim = state["claim"]
    evidence = gather_evidence(
        claim,
        image_b64=state.get("image_b64"),
        media_type=state.get("media_type"),
    )
    return {"evidence": evidence}


def analyze_node(state: AgentState) -> AgentState:
    """Claude analyzes all evidence and produces a structured verdict.

    When an image is present, uses Anthropic SDK directly for Vision support.
    """
    evidence = state["evidence"]
    formatted = _format_evidence(evidence)
    prompt_text = ANALYSIS_PROMPT.format(claim=state["claim"], **formatted)

    image_b64 = state.get("image_b64")
    media_type = state.get("media_type")

    try:
        if image_b64 and media_type:
            # Use Anthropic SDK directly for Vision support
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY, timeout=90.0)
            response = client.messages.create(
                model=LLM_MODEL,
                max_tokens=2000,
                temperature=0,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_b64,
                                },
                            },
                            {"type": "text", "text": prompt_text},
                        ],
                    }
                ],
            )
            response_content = response.content[0].text
        else:
            # Text-only: use LangChain as before
            llm = ChatAnthropic(
                model=LLM_MODEL,
                api_key=ANTHROPIC_API_KEY,
                max_tokens=2000,
                temperature=0,
            )
            response = llm.invoke(prompt_text)
            response_content = response.content

        verdict_data = _parse_json_response(response_content)
        verdict = Verdict(**verdict_data)
    except anthropic.AuthenticationError as e:
        logger.error("Anthropic auth failed in analysis: %s", e)
        verdict = Verdict(
            claim=state["claim"],
            verdict=VerdictLevel.UNCERTAIN,
            confidence=0.0,
            explanation="Hmm, something's off on our end — we couldn't connect to the analysis service. Not your fault! Try again in a bit.",
            sources=[],
            educational_tip="When a tool can't check something for you, try Googling the claim yourself plus the word 'fact check.' You'd be surprised how often that works.",
            reasoning_chain=f"Authentication error: {e}",
        )
    except anthropic.RateLimitError as e:
        logger.warning("Anthropic rate limit hit in analysis: %s", e)
        verdict = Verdict(
            claim=state["claim"],
            verdict=VerdictLevel.UNCERTAIN,
            confidence=0.0,
            explanation="We're getting a lot of requests right now — things are a little backed up. Give it a minute and try again!",
            sources=[],
            educational_tip="While you wait, here's a quick trick: copy the claim and paste it into Google with quotes around it. See what comes up!",
            reasoning_chain=f"Rate limited: {e}",
        )
    except (anthropic.APIConnectionError, anthropic.InternalServerError) as e:
        logger.error("Anthropic API error in analysis: %s", e)
        verdict = Verdict(
            claim=state["claim"],
            verdict=VerdictLevel.UNCERTAIN,
            confidence=0.0,
            explanation="We couldn't reach our analysis service — could be a hiccup on the internet's end. Try again in a sec!",
            sources=[],
            educational_tip="Pro tip: if a fact-checking tool isn't working, try searching for the claim on sites like Snopes, Reuters, or AP News.",
            reasoning_chain=f"API connection error: {e}",
        )
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.error("Failed to parse analysis response: %s", e)
        verdict = Verdict(
            claim=state["claim"],
            verdict=VerdictLevel.UNCERTAIN,
            confidence=0.0,
            explanation="Something went sideways with our analysis — sorry about that! Try rephrasing the claim or give it another go.",
            sources=[],
            educational_tip="Shorter, simpler claims are easier to check. If a claim has multiple parts, try checking each piece separately.",
            reasoning_chain=f"Analysis failed to produce valid output: {e}",
        )
    except Exception as e:
        logger.exception("Unexpected error in analysis node")
        verdict = Verdict(
            claim=state["claim"],
            verdict=VerdictLevel.UNCERTAIN,
            confidence=0.0,
            explanation="Well, that wasn't supposed to happen! Something unexpected went wrong. Give it another try — these things usually sort themselves out.",
            sources=[],
            educational_tip="If you keep running into trouble, try checking the claim on Snopes.com or FactCheck.org — they're great free resources.",
            reasoning_chain=f"Unexpected error: {e}",
        )

    return {"verdict": verdict}


def validate_node(state: AgentState) -> AgentState:
    """Secondary LLM call validates the verdict for quality."""
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

    try:
        llm = ChatAnthropic(
            model=LLM_MODEL_FAST,
            api_key=ANTHROPIC_API_KEY,
            max_tokens=2000,
            temperature=0,
        )
        response = llm.invoke(prompt)
        validation_data = _parse_json_response(response.content)
        corrected = None
        if validation_data.get("corrected_verdict"):
            corrected = Verdict(**validation_data["corrected_verdict"])
        validation = ValidationResult(
            is_valid=validation_data["is_valid"],
            issues=validation_data.get("issues", []),
            corrected_verdict=corrected,
        )
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.error("Failed to parse validation response: %s", e)
        validation = ValidationResult(is_valid=True, issues=[])
    except Exception as e:
        logger.exception("Validation node failed, accepting original verdict")
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

def check_claim(
    claim: str,
    image_b64: str | None = None,
    media_type: str | None = None,
) -> tuple[Verdict, ValidationResult, GatheredEvidence]:
    """Run the full verification pipeline on a claim.

    Optionally includes image analysis if image_b64 and media_type are provided.
    Returns (final_verdict, validation_result, gathered_evidence).
    """
    app = build_graph()
    state: dict = {"claim": claim}
    if image_b64 and media_type:
        state["image_b64"] = image_b64
        state["media_type"] = media_type
    result = app.invoke(state)
    return (
        result["final_verdict"],
        result.get("validation", ValidationResult(is_valid=True)),
        result["evidence"],
    )
