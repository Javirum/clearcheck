"""NOPE — FastAPI server called by N8N chat workflow."""

import base64
import logging
import time
from typing import Optional

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from src.agent import check_claim
from src.audit_log import log_check, log_image_check
from src.image_agent import check_image
from src.schemas import ImageVerdictLevel, VerdictLevel, IMAGE_VERDICT_EMOJI, IMAGE_VERDICT_LABEL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("nope.api")

app = FastAPI(title="NOPE API")

VERDICT_EMOJI = {
    VerdictLevel.TRUE: "✅",
    VerdictLevel.MISLEADING: "⚠️",
    VerdictLevel.FALSE: "❌",
    VerdictLevel.UNCERTAIN: "❓",
}

VERDICT_LABEL = {
    VerdictLevel.TRUE: "Yep, This Checks Out",
    VerdictLevel.MISLEADING: "It's Complicated",
    VerdictLevel.FALSE: "Nope, Not True",
    VerdictLevel.UNCERTAIN: "We're Not Sure Yet",
}


class CheckRequest(BaseModel):
    claim: str


def _build_check_response(verdict, validation, evidence, elapsed):
    """Build the standard check response dict."""
    sources = [
        {"name": s.name, "url": s.url, "snippet": s.snippet}
        for s in verdict.sources
    ]
    response = {
        "verdict": verdict.verdict.value,
        "label": VERDICT_LABEL[verdict.verdict],
        "emoji": VERDICT_EMOJI[verdict.verdict],
        "confidence": verdict.confidence,
        "explanation": verdict.explanation,
        "sources": sources,
        "educational_tip": verdict.educational_tip,
        "reasoning_chain": verdict.reasoning_chain,
        "evidence_summary": {
            "knowledge_base_matches": len(evidence.pinecone_results),
            "web_search_results": len(evidence.tavily_results),
            "published_fact_checks": len(evidence.factcheck_results),
            "image_analyzed": evidence.image_analysis is not None,
            "errors": evidence.errors,
        },
        "validation_passed": validation.is_valid,
        "response_time_seconds": round(elapsed, 1),
    }
    if evidence.image_analysis:
        response["image_analysis"] = {
            "description": evidence.image_analysis.description,
            "ai_generation_signals": evidence.image_analysis.ai_generation_signals,
            "manipulation_signals": evidence.image_analysis.manipulation_signals,
            "authenticity_assessment": evidence.image_analysis.authenticity_assessment,
            "confidence": evidence.image_analysis.confidence,
        }
    return response


@app.post("/check")
def check(req: CheckRequest):
    """Run the full verification pipeline (text only, JSON body)."""
    start = time.time()
    try:
        verdict, validation, evidence = check_claim(req.claim.strip())
    except Exception as e:
        logger.exception("Pipeline failed for claim: %s", req.claim[:100])
        raise HTTPException(
            status_code=500,
            detail="Something went wrong on our end — sorry about that! Give it another try.",
        )
    elapsed = time.time() - start

    try:
        log_check(verdict, evidence, validation, response_time=elapsed)
    except Exception:
        logger.exception("Failed to write audit log")

    return _build_check_response(verdict, validation, evidence, elapsed)


async def _resolve_image(
    file: Optional[UploadFile], image_url: Optional[str]
) -> tuple[Optional[str], Optional[str]]:
    """Resolve image bytes from file or URL, return (b64, media_type) or (None, None)."""
    if file:
        content_type = file.content_type or ""
        if content_type not in SUPPORTED_MEDIA_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {content_type}. Supported: JPEG, PNG, GIF, WebP.",
            )
        image_bytes = await file.read()
    elif image_url:
        async with httpx.AsyncClient(timeout=30) as client:
            try:
                resp = await client.get(image_url)
                resp.raise_for_status()
            except httpx.HTTPError as e:
                raise HTTPException(
                    status_code=400, detail=f"Could not fetch image from URL: {e}"
                )
        content_type = resp.headers.get("content-type", "").split(";")[0].strip()
        if content_type not in SUPPORTED_MEDIA_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported image type from URL: {content_type}. Supported: JPEG, PNG, GIF, WebP.",
            )
        image_bytes = resp.content
    else:
        return None, None

    if len(image_bytes) > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Image too large ({len(image_bytes)} bytes). Max: {MAX_IMAGE_SIZE} bytes (20MB).",
        )
    return (
        base64.b64encode(image_bytes).decode("utf-8"),
        SUPPORTED_MEDIA_TYPES[content_type],
    )


@app.post("/check-with-image")
async def check_with_image(
    claim: str = Form(...),
    file: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None),
):
    """Run the full verification pipeline with an optional image (multipart form)."""
    image_b64, media_type = await _resolve_image(file, image_url)

    start = time.time()
    try:
        verdict, validation, evidence = check_claim(
            claim.strip(), image_b64=image_b64, media_type=media_type
        )
    except Exception as e:
        logger.exception("Pipeline failed for claim with image: %s", claim[:100])
        raise HTTPException(
            status_code=500,
            detail="Something went wrong on our end — sorry about that! Give it another try.",
        )
    elapsed = time.time() - start

    try:
        log_check(verdict, evidence, validation, response_time=elapsed)
    except Exception:
        logger.exception("Failed to write audit log")

    return _build_check_response(verdict, validation, evidence, elapsed)


MAX_IMAGE_SIZE = 20 * 1024 * 1024  # 20 MB
SUPPORTED_MEDIA_TYPES = {
    "image/jpeg": "image/jpeg",
    "image/png": "image/png",
    "image/gif": "image/gif",
    "image/webp": "image/webp",
}


@app.post("/check-image")
async def check_image_endpoint(
    file: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None),
    context: Optional[str] = Form(None),
):
    """Analyze an image for AI generation, manipulation, or misuse."""
    if not file and not image_url:
        raise HTTPException(
            status_code=400,
            detail="Provide either a file upload or an image_url.",
        )

    image_b64, media_type = await _resolve_image(file, image_url)
    user_context = context or ""

    start = time.time()
    try:
        verdict, validation, evidence = check_image(image_b64, media_type, user_context)
    except Exception as e:
        logger.exception("Image pipeline failed")
        raise HTTPException(
            status_code=500,
            detail="We hit a snag analyzing that image — try again or use a different image.",
        )
    elapsed = time.time() - start

    try:
        log_image_check(verdict, evidence, validation, user_context=user_context, response_time=elapsed)
    except Exception:
        logger.exception("Failed to write image audit log")

    sources = [
        {"name": s.name, "url": s.url, "snippet": s.snippet}
        for s in verdict.sources
    ]

    return {
        "verdict": verdict.verdict.value,
        "label": IMAGE_VERDICT_LABEL[verdict.verdict],
        "emoji": IMAGE_VERDICT_EMOJI[verdict.verdict],
        "confidence": verdict.confidence,
        "description": verdict.description,
        "explanation": verdict.explanation,
        "ai_generation_signals": verdict.ai_generation_signals,
        "manipulation_signals": verdict.manipulation_signals,
        "context_analysis": verdict.context_analysis,
        "sources": sources,
        "educational_tip": verdict.educational_tip,
        "reasoning_chain": verdict.reasoning_chain,
        "evidence_summary": {
            "reverse_search_results": len(evidence.reverse_search_results),
            "has_metadata": evidence.metadata is not None,
            "errors": evidence.errors,
        },
        "validation_passed": validation.is_valid,
        "response_time_seconds": round(elapsed, 1),
    }


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
