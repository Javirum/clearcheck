"""ClearCheck — FastAPI server called by N8N chat workflow."""

import time

from fastapi import FastAPI
from pydantic import BaseModel

from src.agent import check_claim
from src.audit_log import log_check
from src.schemas import VerdictLevel

app = FastAPI(title="ClearCheck API")

VERDICT_EMOJI = {
    VerdictLevel.TRUE: "✅",
    VerdictLevel.MISLEADING: "⚠️",
    VerdictLevel.FALSE: "❌",
    VerdictLevel.UNCERTAIN: "❓",
}

VERDICT_LABEL = {
    VerdictLevel.TRUE: "Looks True",
    VerdictLevel.MISLEADING: "Misleading",
    VerdictLevel.FALSE: "This Is False",
    VerdictLevel.UNCERTAIN: "Not Enough Evidence",
}


class CheckRequest(BaseModel):
    claim: str


@app.post("/check")
def check(req: CheckRequest):
    """Run the full verification pipeline and return a formatted response."""
    start = time.time()
    verdict, validation, evidence = check_claim(req.claim.strip())
    elapsed = time.time() - start

    log_check(verdict, evidence, validation, response_time=elapsed)

    # Build sources list for the response
    sources = [
        {"name": s.name, "url": s.url, "snippet": s.snippet}
        for s in verdict.sources
    ]

    return {
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
            "errors": evidence.errors,
        },
        "validation_passed": validation.is_valid,
        "response_time_seconds": round(elapsed, 1),
    }


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
