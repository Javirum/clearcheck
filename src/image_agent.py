"""LangGraph agent for NOPE image verification using Claude Vision."""

from __future__ import annotations

import json
import logging

import anthropic
from langgraph.graph import END, StateGraph

from src.config import ANTHROPIC_API_KEY, LLM_MODEL, LLM_MODEL_FAST

logger = logging.getLogger("nope.image_agent")
from src.image_evidence import gather_image_evidence
from src.schemas import (
    ImageAgentState,
    ImageEvidence,
    ImageValidationResult,
    ImageVerdict,
    ImageVerdictLevel,
)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

IMAGE_ANALYSIS_PROMPT = """\
You are NOPE — a friendly image detective that helps people figure out if a \
photo or image they saw online is legit, AI-generated, doctored, or taken out of context.

Your vibe: that one tech-savvy friend who explains things clearly without making \
anyone feel dumb. Warm, direct, a little witty. Zero condescension. Zero jargon.

## USER CONTEXT
{user_context}

## IMAGE METADATA
{metadata}

## REVERSE IMAGE SEARCH RESULTS
{reverse_search}

## EVIDENCE GATHERING ERRORS
{errors}

## INSTRUCTIONS

Take a good look at the image and all the evidence. Here's what to watch for:

1. **AI generation red flags**: weird hands or fingers, lighting that doesn't make sense, skin that looks too smooth to be real, text that's garbled or nonsensical, repeating patterns, things that just look... off.
2. **Editing/manipulation signs**: shadows going in different directions, copy-paste artifacts (elements that repeat), weird edges around objects that look pasted in, mismatched image quality in different areas.
3. **Context check**: Does this image actually show what people say it shows? Cross-reference with reverse search results. Has someone grabbed a real photo and used it to tell a completely different story? (Spoiler: this happens a LOT.)

Be straight about what you're not sure about. This is visual analysis, not a crime lab — say so when confidence is low.

## OUTPUT FORMAT

Return a valid JSON object with exactly these fields:
{{
  "description": "<brief description of what the image shows>",
  "verdict": "<authentic|ai_generated|manipulated|out_of_context|uncertain>",
  "confidence": <0.0 to 1.0>,
  "explanation": "<plain-language explanation, 2-4 sentences — like you're telling a friend what you spotted>",
  "ai_generation_signals": ["<signal 1>", "<signal 2>"],
  "manipulation_signals": ["<signal 1>", "<signal 2>"],
  "context_analysis": "<analysis of whether image is used in proper context>",
  "sources": [
    {{"name": "<source name>", "url": "<actual URL from evidence>", "snippet": "<relevant excerpt>"}}
  ],
  "educational_tip": "<one practical, friendly tip for spotting fake or manipulated images next time>",
  "reasoning_chain": "<step-by-step reasoning>"
}}

Return ONLY the JSON object, no other text."""

IMAGE_VALIDATION_PROMPT = """\
You are a quality checker for NOPE's image analysis. Make sure the verdict is solid, \
honest, and sounds like a helpful friend — not a forensics report.

## VERDICT TO VALIDATE
{verdict_json}

## AVAILABLE EVIDENCE
{evidence_summary}

## VALIDATION CHECKS

Check each of these and flag any issues:

1. **Verdict consistency**: Does the verdict actually match what the explanation says? No contradictions.
2. **Signal grounding**: Do the reported red flags make sense for what's in the image? No phantom signals.
3. **Confidence calibration**: Is the confidence score honest? Thin evidence = low confidence. AI image detection is tricky — confidence above 0.9 should be very rare.
4. **Tone & clarity**: Does it read like a friendly, plain-language explanation? No jargon, no condescension, no robot-speak.
5. **Uncertainty honesty**: Does it clearly say this is analysis, not proof? Nobody should think we ran it through a crime lab.
6. **Context analysis**: If reverse search results came back, were they actually used?

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
# Helpers
# ---------------------------------------------------------------------------

def _format_image_evidence(evidence: ImageEvidence) -> dict[str, str]:
    metadata_text = "No metadata available.\n"
    if evidence.metadata:
        m = evidence.metadata
        metadata_text = (
            f"Format: {m.format or 'Unknown'}\n"
            f"Dimensions: {m.width}x{m.height}\n"
            f"File size: {m.file_size_bytes or 'Unknown'} bytes\n"
        )
        if m.exif:
            metadata_text += "EXIF data:\n"
            for k, v in list(m.exif.items())[:15]:
                metadata_text += f"  {k}: {v}\n"

    search_text = ""
    if evidence.reverse_search_results:
        for r in evidence.reverse_search_results:
            search_text += f"- [{r.title}]({r.url})\n  {r.content[:300]}\n\n"
    else:
        search_text = "No reverse image search results found.\n"

    errors_text = "\n".join(evidence.errors) if evidence.errors else "None"

    return {
        "metadata": metadata_text,
        "reverse_search": search_text,
        "errors": errors_text,
    }


def _parse_json_response(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown code blocks."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    return json.loads(text)


# ---------------------------------------------------------------------------
# LangGraph nodes
# ---------------------------------------------------------------------------

def gather_image_evidence_node(state: ImageAgentState) -> ImageAgentState:
    """Gather image metadata and reverse search results."""
    import base64

    image_bytes = base64.b64decode(state["image_b64"])
    evidence = gather_image_evidence(
        image_bytes, state["image_b64"], state.get("user_context", "")
    )
    return {"evidence": evidence}


def _uncertain_image_verdict(reason: str) -> ImageVerdict:
    """Create a fallback UNCERTAIN verdict for image analysis failures."""
    return ImageVerdict(
        description="Unable to analyze image",
        verdict=ImageVerdictLevel.UNCERTAIN,
        confidence=0.0,
        explanation="We hit a snag and couldn't analyze this image — sorry about that! Try again, or if you have a different version of the image, give that a shot.",
        ai_generation_signals=[],
        manipulation_signals=[],
        context_analysis="Analysis could not be completed.",
        sources=[],
        educational_tip="Here's a handy trick: right-click any image and choose 'Search image with Google' (or drag it into Google Images). You can often find where it originally came from.",
        reasoning_chain=reason,
    )


def analyze_image_node(state: ImageAgentState) -> ImageAgentState:
    """Use Claude Vision to analyze the image for authenticity."""
    evidence = state["evidence"]
    formatted = _format_image_evidence(evidence)
    user_context = state.get("user_context", "No context provided")

    prompt_text = IMAGE_ANALYSIS_PROMPT.format(
        user_context=user_context, **formatted
    )

    try:
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
                                "media_type": state["media_type"],
                                "data": state["image_b64"],
                            },
                        },
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ],
        )
        response_text = response.content[0].text
        verdict_data = _parse_json_response(response_text)
        verdict = ImageVerdict(**verdict_data)
    except anthropic.AuthenticationError as e:
        logger.error("Anthropic auth failed in image analysis: %s", e)
        verdict = _uncertain_image_verdict(f"Authentication error: {e}")
    except anthropic.RateLimitError as e:
        logger.warning("Anthropic rate limit hit in image analysis: %s", e)
        verdict = _uncertain_image_verdict(f"Rate limited: {e}")
    except (anthropic.APIConnectionError, anthropic.InternalServerError) as e:
        logger.error("Anthropic API error in image analysis: %s", e)
        verdict = _uncertain_image_verdict(f"API error: {e}")
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.error("Failed to parse image analysis response: %s", e)
        verdict = _uncertain_image_verdict(f"Failed to parse response: {e}")
    except Exception as e:
        logger.exception("Unexpected error in image analysis node")
        verdict = _uncertain_image_verdict(f"Unexpected error: {e}")

    return {"verdict": verdict}


def validate_image_node(state: ImageAgentState) -> ImageAgentState:
    """Validate the image verdict for quality and consistency."""
    evidence = state["evidence"]
    verdict = state["verdict"]
    formatted = _format_image_evidence(evidence)
    evidence_summary = "\n".join(
        f"[{k}]\n{v}" for k, v in formatted.items() if k != "errors"
    )

    prompt = IMAGE_VALIDATION_PROMPT.format(
        verdict_json=verdict.model_dump_json(indent=2),
        evidence_summary=evidence_summary,
    )

    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY, timeout=60.0)
        response = client.messages.create(
            model=LLM_MODEL_FAST,
            max_tokens=2000,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        response_text = response.content[0].text
        validation_data = _parse_json_response(response_text)
        corrected = None
        if validation_data.get("corrected_verdict"):
            corrected = ImageVerdict(**validation_data["corrected_verdict"])
        validation = ImageValidationResult(
            is_valid=validation_data["is_valid"],
            issues=validation_data.get("issues", []),
            corrected_verdict=corrected,
        )
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        logger.error("Failed to parse image validation response: %s", e)
        validation = ImageValidationResult(is_valid=True, issues=[])
    except Exception as e:
        logger.exception("Image validation node failed, accepting original verdict")
        validation = ImageValidationResult(is_valid=True, issues=[])

    final = validation.corrected_verdict if validation.corrected_verdict else verdict
    return {"validation": validation, "final_verdict": final}


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def build_image_graph() -> StateGraph:
    graph = StateGraph(ImageAgentState)

    graph.add_node("gather_image_evidence", gather_image_evidence_node)
    graph.add_node("analyze_image", analyze_image_node)
    graph.add_node("validate_image", validate_image_node)

    graph.set_entry_point("gather_image_evidence")
    graph.add_edge("gather_image_evidence", "analyze_image")
    graph.add_edge("analyze_image", "validate_image")
    graph.add_edge("validate_image", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def check_image(
    image_b64: str, media_type: str, user_context: str = ""
) -> tuple[ImageVerdict, ImageValidationResult, ImageEvidence]:
    """Run the full image verification pipeline.

    Returns (final_verdict, validation_result, image_evidence).
    """
    app = build_image_graph()
    result = app.invoke({
        "image_b64": image_b64,
        "media_type": media_type,
        "user_context": user_context,
    })
    return (
        result["final_verdict"],
        result.get("validation", ImageValidationResult(is_valid=True)),
        result["evidence"],
    )
