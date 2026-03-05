"""NOPE — FastAPI server called by N8N chat workflow."""

import base64
import json
import logging
import os
import time
from typing import Optional

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

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

# --- Rate limiting ---
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# --- CORS (env-var-driven allowlist) ---
_allowed_origins = os.environ.get(
    "ALLOWED_ORIGINS", "http://localhost:8000,http://localhost:3000"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Claim length validation ---
MAX_CLAIM_LENGTH = 5000
MAX_CHAT_BODY_SIZE = 25 * 1024 * 1024  # 25 MB (accommodates image uploads via chat)

# --- Static files ---
_static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=_static_dir), name="static")


@app.get("/")
def landing_page():
    """Serve the NOPE landing page."""
    return FileResponse(os.path.join(_static_dir, "index.html"))


@app.get("/config.js")
def config_js():
    """Inject runtime config (n8n webhook URL) as JavaScript."""
    webhook_url = os.environ.get("N8N_WEBHOOK_URL", "")
    js = f'window.__NOPE_CONFIG__ = {{ N8N_WEBHOOK_URL: "{webhook_url}" }};'
    return PlainTextResponse(js, media_type="application/javascript")

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
            "scam_flags_detected": len(evidence.scam_analysis.red_flags_detected) if evidence.scam_analysis else 0,
            "urls_checked": len(evidence.url_safety.urls_found) if evidence.url_safety else 0,
            "errors": evidence.errors,
        },
        "validation_passed": validation.is_valid,
        "response_time_seconds": round(elapsed, 1),
    }
    if verdict.scam_assessment:
        response["scam_assessment"] = {
            "is_likely_scam": verdict.scam_assessment.is_likely_scam,
            "scam_type": verdict.scam_assessment.scam_type,
            "scam_confidence": verdict.scam_assessment.scam_confidence,
            "red_flags": verdict.scam_assessment.red_flags,
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
@limiter.limit("10/minute")
def check(req: CheckRequest, request: Request):
    """Run the full verification pipeline (text only, JSON body)."""
    if len(req.claim) > MAX_CLAIM_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Claim is too long ({len(req.claim)} chars). Please keep it under {MAX_CLAIM_LENGTH} characters.",
        )
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
@limiter.limit("5/minute")
async def check_with_image(
    request: Request,
    claim: str = Form(...),
    file: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None),
):
    """Run the full verification pipeline with an optional image (multipart form)."""
    if len(claim) > MAX_CLAIM_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Claim is too long ({len(claim)} chars). Please keep it under {MAX_CLAIM_LENGTH} characters.",
        )
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
@limiter.limit("5/minute")
async def check_image_endpoint(
    request: Request,
    file: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None),
    context: Optional[str] = Form(None),
):
    """Analyze an image for AI generation, manipulation, or misuse."""
    if context and len(context) > MAX_CLAIM_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Context is too long ({len(context)} chars). Please keep it under {MAX_CLAIM_LENGTH} characters.",
        )
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


def _format_check_as_chat(result: dict) -> str:
    """Format a /check or /check-image result dict as a friendly markdown chat message."""
    lines = []
    emoji = result.get("emoji", "")
    label = result.get("label", "")
    lines.append(f"## {emoji} {label}\n")

    if result.get("explanation"):
        lines.append(result["explanation"] + "\n")

    # Scam assessment (text check)
    scam = result.get("scam_assessment")
    if scam and scam.get("is_likely_scam"):
        lines.append(f"**Scam alert ({scam.get('scam_type', 'unknown')}):** "
                      + ", ".join(scam.get("red_flags", [])) + "\n")

    # Image-specific fields
    if result.get("ai_generation_signals"):
        lines.append("**AI generation signals:** " + ", ".join(result["ai_generation_signals"]) + "\n")
    if result.get("manipulation_signals"):
        lines.append("**Manipulation signals:** " + ", ".join(result["manipulation_signals"]) + "\n")
    if result.get("context_analysis"):
        lines.append("**Context:** " + result["context_analysis"] + "\n")

    # Sources
    sources = result.get("sources", [])
    if sources:
        lines.append("**Sources:**")
        for s in sources[:5]:
            name = s.get("name", s.get("url", ""))
            url = s.get("url", "")
            lines.append(f"- [{name}]({url})" if url else f"- {name}")
        lines.append("")

    if result.get("educational_tip"):
        lines.append(f"💡 **Tip:** {result['educational_tip']}")

    return "\n".join(lines)


@app.post("/api/chat")
@limiter.limit("10/minute")
async def proxy_chat(request: Request):
    """Handle chat messages. Routes image requests through NOPE pipelines, proxies text to n8n."""
    raw_body = await request.body()
    if len(raw_body) > MAX_CHAT_BODY_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Request body too large ({len(raw_body)} bytes). Max: {MAX_CHAT_BODY_SIZE} bytes.",
        )

    try:
        payload = json.loads(raw_body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise HTTPException(status_code=400, detail="Invalid JSON body.")

    files = payload.get("files") or []
    chat_input = (payload.get("chatInput") or "").strip()

    # --- Image attached: run through NOPE pipelines directly ---
    if files:
        file_obj = files[0]
        image_b64 = file_obj.get("data")
        media_type = file_obj.get("type", "image/jpeg")
        if not image_b64:
            return JSONResponse({"output": "I couldn't read that image. Please try uploading again."})

        start = time.time()
        try:
            # If user provided text context, run the full claim+image pipeline
            if chat_input and chat_input != "Please analyze this image.":
                verdict, validation, evidence = check_claim(
                    chat_input, image_b64=image_b64, media_type=media_type
                )
                elapsed = time.time() - start
                try:
                    log_check(verdict, evidence, validation, response_time=elapsed)
                except Exception:
                    logger.exception("Failed to write audit log")
                result = _build_check_response(verdict, validation, evidence, elapsed)
            else:
                # Image-only: run image analysis pipeline
                image_bytes = base64.b64decode(image_b64)
                verdict, validation, evidence = check_image(
                    image_b64, media_type, chat_input
                )
                elapsed = time.time() - start
                try:
                    log_image_check(verdict, evidence, validation, user_context=chat_input, response_time=elapsed)
                except Exception:
                    logger.exception("Failed to write image audit log")
                result = {
                    "verdict": verdict.verdict.value,
                    "label": IMAGE_VERDICT_LABEL[verdict.verdict],
                    "emoji": IMAGE_VERDICT_EMOJI[verdict.verdict],
                    "explanation": verdict.explanation,
                    "ai_generation_signals": verdict.ai_generation_signals,
                    "manipulation_signals": verdict.manipulation_signals,
                    "context_analysis": verdict.context_analysis,
                    "sources": [{"name": s.name, "url": s.url, "snippet": s.snippet} for s in verdict.sources],
                    "educational_tip": verdict.educational_tip,
                }
        except Exception:
            logger.exception("Chat image pipeline failed")
            return JSONResponse({"output": "Something went wrong analyzing that image — please try again."})

        return JSONResponse({"output": _format_check_as_chat(result)})

    # --- Text only: proxy to n8n ---
    webhook_url = os.environ.get("N8N_WEBHOOK_URL", "")
    if not webhook_url:
        raise HTTPException(status_code=503, detail="N8N_WEBHOOK_URL not configured")

    headers = {"Content-Type": request.headers.get("content-type", "application/json")}
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(webhook_url, content=raw_body, headers=headers)

    return Response(content=resp.content, status_code=resp.status_code,
                    media_type=resp.headers.get("content-type"))


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import os
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
