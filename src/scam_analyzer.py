"""Lightweight pattern-based scam text analyzer.

Runs before the LLM call to surface red flags in claim text. Designed to be
fast enough to execute as a parallel evidence source alongside Pinecone/Tavily.
"""

from __future__ import annotations

import re

from src.schemas import ScamAnalysisResult

# ---------------------------------------------------------------------------
# Pattern dictionaries — each maps a label to compiled regex patterns
# ---------------------------------------------------------------------------

URGENCY_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("Uses 'act now' pressure", re.compile(r"\bact\s+now\b", re.I)),
    ("Claims 'limited time' offer", re.compile(r"\blimited\s+time\b", re.I)),
    ("Says 'final notice'", re.compile(r"\bfinal\s+notice\b", re.I)),
    ("Threatens account suspension", re.compile(r"\baccount\s+(suspended|suspension|deactivat)", re.I)),
    ("Says 'immediate action required'", re.compile(r"\bimmediate(ly)?\b", re.I)),
    ("Claims offer 'expires today'", re.compile(r"\b(expires?\s+today|today\s+only|last\s+chance)\b", re.I)),
    ("Pressures with time deadline", re.compile(r"\b(within\s+\d+\s+hours?|in\s+the\s+next\s+\d+\s+minutes?)\b", re.I)),
    ("Warns of 'urgent' matter", re.compile(r"\burgent(ly)?\b", re.I)),
    ("Says 'don't delay'", re.compile(r"\b(don'?t\s+delay|time\s+is\s+running\s+out)\b", re.I)),
]

THREAT_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("Threatens arrest warrant", re.compile(r"\b(arrest\s+warrant|warrant\s+for\s+(your\s+)?arrest)\b", re.I)),
    ("Threatens legal action", re.compile(r"\b(legal\s+action|lawsuit|prosecut|sued)\b", re.I)),
    ("Threatens account freeze", re.compile(r"\b(account\s+frozen|freeze\s+your\s+account|account\s+will\s+be\s+closed)\b", re.I)),
    ("Threatens service disconnection", re.compile(r"\b(disconnected?|shut\s+off|terminat(e|ed|ion)\s+(your|service))\b", re.I)),
    ("Threatens with police", re.compile(r"\b(police|law\s+enforcement|authorities)\s+(will|have been|are)\b", re.I)),
    ("Claims identity has been compromised", re.compile(r"\b(identity|account|ssn|social\s+security)\s+(has\s+been\s+)?(compromised|stolen|used\s+fraudulently)\b", re.I)),
]

TOO_GOOD_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("Claims 'you've won' a prize", re.compile(r"\b(you('ve|\s+have)\s+won|congratulations.{0,20}(winner|won|selected))\b", re.I)),
    ("Promises guaranteed returns", re.compile(r"\bguaranteed\s+(returns?|income|profit|earnings)\b", re.I)),
    ("Mentions large dollar amount", re.compile(r"\$\s*[\d,]{4,}", re.I)),
    ("Claims 'risk-free' opportunity", re.compile(r"\brisk[- ]free\b", re.I)),
    ("Promises 'free' with hidden costs", re.compile(r"\b(free\s+(gift|vacation|cruise|trip|money|iphone|ipad))\b", re.I)),
    ("Promises specific high earnings", re.compile(r"\b(earn|make)\s+\$?\d[\d,]*\s*(per|a|every)\s*(day|week|month)\b", re.I)),
    ("Claims exclusive selection", re.compile(r"\b(you('ve|\s+have)\s+been\s+(selected|chosen)|exclusively?\s+selected)\b", re.I)),
]

PAYMENT_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("Requests gift card payment", re.compile(r"\bgift\s*cards?\b", re.I)),
    ("Requests wire transfer", re.compile(r"\b(wire\s+transfer|western\s+union|moneygram)\b", re.I)),
    ("Requests cryptocurrency payment", re.compile(r"\b(bitcoin|crypto(currency)?|btc|ethereum|bitcoin\s+atm)\b", re.I)),
    ("Requests Zelle/Venmo/CashApp", re.compile(r"\b(zelle|venmo|cash\s*app)\b", re.I)),
    ("Asks for prepaid debit card", re.compile(r"\bprepaid\s+(debit\s+)?card\b", re.I)),
    ("Requests money order", re.compile(r"\bmoney\s+order\b", re.I)),
]

IMPERSONATION_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("Claims to be IRS", re.compile(r"\b(irs|internal\s+revenue\s+service)\b", re.I)),
    ("Claims to be Social Security", re.compile(r"\b(social\s+security\s+(administration|office)|ssa)\b", re.I)),
    ("Claims to be Microsoft", re.compile(r"\b(microsoft|windows\s+(support|defender|security))\b", re.I)),
    ("Claims to be Apple Support", re.compile(r"\b(apple\s+(support|care|id)|icloud)\b", re.I)),
    ("Claims to be a bank", re.compile(r"\b(bank\s+of\s+america|wells\s+fargo|chase\s+bank|citibank)\b", re.I)),
    ("Claims to be Medicare", re.compile(r"\bmedicare\b", re.I)),
    ("Claims to be Amazon", re.compile(r"\bamazon\b", re.I)),
    ("Claims to be Geek Squad", re.compile(r"\bgeek\s+squad\b", re.I)),
    ("Claims to be a sheriff or DEA", re.compile(r"\b(sheriff|dea|drug\s+enforcement)\b", re.I)),
    ("Claims to be PayPal", re.compile(r"\bpaypal\b", re.I)),
    ("Requests remote access to your computer", re.compile(r"\b(remote\s+access|teamviewer|anydesk|install\s+(this\s+)?software)\b", re.I)),
    ("Unsolicited call about computer problem", re.compile(r"\b(called?\s+(about|to\s+say|saying)|phon(e|ed)\s+(about|saying))\b.*\b(virus|infect|computer|security)\b", re.I)),
]

INFO_HARVEST_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("Asks for Social Security number", re.compile(r"\b(social\s+security\s+number|ssn)\b", re.I)),
    ("Asks for password", re.compile(r"\b(password|passcode|pin\s*number|login\s+credentials)\b", re.I)),
    ("Asks for credit card number", re.compile(r"\b(credit\s+card\s+(number|info|details)|card\s+number|cvv|expir(y|ation)\s+date)\b", re.I)),
    ("Asks for bank account details", re.compile(r"\b(bank\s+account\s+(number|info|details)|routing\s+number|account\s+number)\b", re.I)),
    ("Asks to verify identity via link", re.compile(r"\b(verify\s+(your\s+)?(identity|account)|confirm\s+(your\s+)?(identity|account))\b", re.I)),
    ("Asks for Medicare number", re.compile(r"\bmedicare\s+(number|id|card)\b", re.I)),
]

# Category detection — maps scam type to its primary patterns
_CATEGORY_SIGNALS: dict[str, list[list[tuple[str, re.Pattern]]]] = {
    "tech_support": [IMPERSONATION_PATTERNS],
    "romance": [],  # detected via keyword heuristic below
    "government": [IMPERSONATION_PATTERNS, THREAT_PATTERNS],
    "grandparent": [],  # detected via keyword heuristic below
    "lottery": [TOO_GOOD_PATTERNS],
    "phishing": [INFO_HARVEST_PATTERNS, IMPERSONATION_PATTERNS],
    "investment": [TOO_GOOD_PATTERNS],
    "charity": [],  # detected via keyword heuristic below
}

_ROMANCE_KEYWORDS = re.compile(
    r"\b(oil\s+rig|military\s+(officer|deployed)|dating\s+(site|app)|online\s+(boyfriend|girlfriend|lover|partner|relationship)|"
    r"met\s+(someone|him|her)\s+online|customs?\s+fees?|stranded\s+(abroad|overseas)|inheritance.*legal\s+fees)\b",
    re.I,
)

_GRANDPARENT_KEYWORDS = re.compile(
    r"\b(grand(ma|pa|mother|father|child|son|daughter|kid)|nephew|niece|"
    r"bail\s+money|stranded|it'?s\s+me|don'?t\s+tell\s+(mom|dad|anyone))\b",
    re.I,
)

_CHARITY_KEYWORDS = re.compile(
    r"\b(donat(e|ion)|charity|humanitarian|relief\s+fund|veterans?\s+fund|"
    r"orphan|mission\s+trip|disaster\s+relief|go\s*fund\s*me)\b",
    re.I,
)

_TECH_SUPPORT_KEYWORDS = re.compile(
    r"\b(remote\s+access|virus|infected|computer\s+problem|tech(nical)?\s+support|"
    r"pop-?up\s+(warning|alert)|subscription\s+(expired|renewal)|auto[- ]?renewal)\b",
    re.I,
)


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def analyze_scam_patterns(text: str) -> ScamAnalysisResult:
    """Analyze text for scam indicators. Fast, regex-only — no LLM call."""
    red_flags: list[str] = []
    category_scores: dict[str, float] = {}

    # Run all pattern categories
    urgency_hits = _match_patterns(text, URGENCY_PATTERNS)
    threat_hits = _match_patterns(text, THREAT_PATTERNS)
    too_good_hits = _match_patterns(text, TOO_GOOD_PATTERNS)
    payment_hits = _match_patterns(text, PAYMENT_PATTERNS)
    impersonation_hits = _match_patterns(text, IMPERSONATION_PATTERNS)
    info_harvest_hits = _match_patterns(text, INFO_HARVEST_PATTERNS)

    red_flags.extend(urgency_hits)
    red_flags.extend(threat_hits)
    red_flags.extend(too_good_hits)
    red_flags.extend(payment_hits)
    red_flags.extend(impersonation_hits)
    red_flags.extend(info_harvest_hits)

    # Urgency score based on urgency + threat patterns
    urgency_count = len(urgency_hits) + len(threat_hits)
    urgency_score = min(urgency_count / 4.0, 1.0)

    # Detect scam category
    scam_type = _detect_category(
        text, urgency_hits, threat_hits, too_good_hits,
        payment_hits, impersonation_hits, info_harvest_hits,
    )

    # Compute overall scam likelihood
    total_flags = len(red_flags)
    if total_flags == 0:
        scam_likelihood = 0.0
    elif total_flags == 1:
        scam_likelihood = 0.2
    elif total_flags == 2:
        scam_likelihood = 0.4
    elif total_flags == 3:
        scam_likelihood = 0.6
    elif total_flags == 4:
        scam_likelihood = 0.75
    else:
        scam_likelihood = min(0.8 + (total_flags - 5) * 0.04, 0.95)

    # Boost likelihood for payment method red flags (very strong signal)
    if payment_hits:
        scam_likelihood = min(scam_likelihood + 0.15, 0.95)

    return ScamAnalysisResult(
        scam_likelihood=round(scam_likelihood, 2),
        scam_type=scam_type,
        red_flags_detected=red_flags,
        urgency_score=round(urgency_score, 2),
    )


def _match_patterns(text: str, patterns: list[tuple[str, re.Pattern]]) -> list[str]:
    """Return labels for all patterns that match in the text."""
    return [label for label, regex in patterns if regex.search(text)]


def _detect_category(
    text: str,
    urgency_hits: list[str],
    threat_hits: list[str],
    too_good_hits: list[str],
    payment_hits: list[str],
    impersonation_hits: list[str],
    info_harvest_hits: list[str],
) -> str | None:
    """Heuristic category detection based on pattern combinations."""
    # Keyword-based categories first (most specific)
    if _GRANDPARENT_KEYWORDS.search(text):
        return "grandparent"
    if _ROMANCE_KEYWORDS.search(text):
        return "romance"
    if _CHARITY_KEYWORDS.search(text):
        return "charity"

    # Tech support: impersonation + (urgency or remote access keywords)
    if impersonation_hits and _TECH_SUPPORT_KEYWORDS.search(text):
        return "tech_support"

    # Government: IRS/SSA/Medicare impersonation + threats
    gov_keywords = any("IRS" in h or "Social Security" in h or "Medicare" in h or "sheriff" in h or "DEA" in h for h in impersonation_hits)
    if gov_keywords and (threat_hits or urgency_hits):
        return "government"

    # Phishing: info harvesting + impersonation or urgency
    if info_harvest_hits and (impersonation_hits or urgency_hits):
        return "phishing"

    # Investment: too-good-to-be-true + financial keywords
    if too_good_hits and re.search(r"\b(invest|trading|forex|crypto|returns?|profit|portfolio)\b", text, re.I):
        return "investment"

    # Lottery/Prize: winning + fee request
    if too_good_hits and re.search(r"\b(won|winner|lottery|sweepstakes|prize|raffle)\b", text, re.I):
        return "lottery"

    # Tech support fallback
    if _TECH_SUPPORT_KEYWORDS.search(text) and (urgency_hits or payment_hits):
        return "tech_support"

    # Generic — if enough flags but no clear category
    total = len(urgency_hits) + len(threat_hits) + len(too_good_hits) + len(payment_hits) + len(impersonation_hits) + len(info_harvest_hits)
    if total >= 2:
        # Return the most likely based on which has most hits
        counts = {
            "phishing": len(info_harvest_hits) + len(impersonation_hits),
            "tech_support": len(impersonation_hits),
            "government": len(threat_hits),
            "investment": len(too_good_hits),
        }
        best = max(counts, key=counts.get)
        if counts[best] > 0:
            return best

    return None
