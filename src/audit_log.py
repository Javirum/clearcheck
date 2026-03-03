"""SQLite audit logging for every NOPE verification."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from src.schemas import (
    GatheredEvidence,
    ImageEvidence,
    ImageValidationResult,
    ImageVerdict,
    ValidationResult,
    Verdict,
)

DB_PATH = Path(__file__).resolve().parent.parent / "nope_audit.db"


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    """Create the audit log table if it doesn't exist."""
    conn = _get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            claim TEXT NOT NULL,
            verdict TEXT NOT NULL,
            confidence REAL NOT NULL,
            explanation TEXT NOT NULL,
            sources TEXT NOT NULL,
            educational_tip TEXT NOT NULL,
            reasoning_chain TEXT NOT NULL,
            evidence_summary TEXT NOT NULL,
            validation_passed INTEGER NOT NULL,
            validation_issues TEXT NOT NULL,
            num_sources_consulted INTEGER NOT NULL,
            response_time_seconds REAL
        )
    """)
    conn.commit()
    conn.close()


def log_check(
    verdict: Verdict,
    evidence: GatheredEvidence,
    validation: ValidationResult,
    response_time: float | None = None,
) -> int:
    """Log a completed check to the audit database. Returns the row id."""
    init_db()

    num_sources = (
        len(evidence.pinecone_results)
        + len(evidence.tavily_results)
        + len(evidence.factcheck_results)
    )

    evidence_summary = {
        "pinecone_count": len(evidence.pinecone_results),
        "tavily_count": len(evidence.tavily_results),
        "factcheck_count": len(evidence.factcheck_results),
        "errors": evidence.errors,
    }

    conn = _get_connection()
    cursor = conn.execute(
        """
        INSERT INTO audit_log (
            timestamp, claim, verdict, confidence, explanation,
            sources, educational_tip, reasoning_chain,
            evidence_summary, validation_passed, validation_issues,
            num_sources_consulted, response_time_seconds
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now(timezone.utc).isoformat(),
            verdict.claim,
            verdict.verdict.value,
            verdict.confidence,
            verdict.explanation,
            json.dumps([s.model_dump() for s in verdict.sources]),
            verdict.educational_tip,
            verdict.reasoning_chain,
            json.dumps(evidence_summary),
            1 if validation.is_valid else 0,
            json.dumps(validation.issues),
            num_sources,
            response_time,
        ),
    )
    conn.commit()
    row_id = cursor.lastrowid
    conn.close()
    return row_id


def get_recent_checks(limit: int = 10) -> list[dict]:
    """Retrieve recent audit log entries."""
    init_db()
    conn = _get_connection()
    rows = conn.execute(
        "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


# ---------------------------------------------------------------------------
# Image audit log
# ---------------------------------------------------------------------------

def init_image_db() -> None:
    """Create the image audit log table if it doesn't exist."""
    conn = _get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS image_audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            description TEXT NOT NULL,
            verdict TEXT NOT NULL,
            confidence REAL NOT NULL,
            explanation TEXT NOT NULL,
            ai_generation_signals TEXT NOT NULL,
            manipulation_signals TEXT NOT NULL,
            context_analysis TEXT NOT NULL,
            sources TEXT NOT NULL,
            educational_tip TEXT NOT NULL,
            reasoning_chain TEXT NOT NULL,
            evidence_summary TEXT NOT NULL,
            validation_passed INTEGER NOT NULL,
            validation_issues TEXT NOT NULL,
            user_context TEXT,
            response_time_seconds REAL
        )
    """)
    conn.commit()
    conn.close()


def log_image_check(
    verdict: ImageVerdict,
    evidence: ImageEvidence,
    validation: ImageValidationResult,
    user_context: str = "",
    response_time: float | None = None,
) -> int:
    """Log a completed image check to the audit database. Returns the row id."""
    init_image_db()

    evidence_summary = {
        "reverse_search_count": len(evidence.reverse_search_results),
        "has_metadata": evidence.metadata is not None,
        "errors": evidence.errors,
    }

    conn = _get_connection()
    cursor = conn.execute(
        """
        INSERT INTO image_audit_log (
            timestamp, description, verdict, confidence, explanation,
            ai_generation_signals, manipulation_signals, context_analysis,
            sources, educational_tip, reasoning_chain,
            evidence_summary, validation_passed, validation_issues,
            user_context, response_time_seconds
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now(timezone.utc).isoformat(),
            verdict.description,
            verdict.verdict.value,
            verdict.confidence,
            verdict.explanation,
            json.dumps(verdict.ai_generation_signals),
            json.dumps(verdict.manipulation_signals),
            verdict.context_analysis,
            json.dumps([s.model_dump() for s in verdict.sources]),
            verdict.educational_tip,
            verdict.reasoning_chain,
            json.dumps(evidence_summary),
            1 if validation.is_valid else 0,
            json.dumps(validation.issues),
            user_context,
            response_time,
        ),
    )
    conn.commit()
    row_id = cursor.lastrowid
    conn.close()
    return row_id


def get_recent_image_checks(limit: int = 10) -> list[dict]:
    """Retrieve recent image audit log entries."""
    init_image_db()
    conn = _get_connection()
    rows = conn.execute(
        "SELECT * FROM image_audit_log ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]
