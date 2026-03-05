"""URL extraction and safety checking.

Extracts URLs from claim text, analyzes them for phishing/scam indicators,
and optionally checks against Google Safe Browsing API.
"""

from __future__ import annotations

import logging
import re
from urllib.parse import urlparse

import requests

from src.config import GOOGLE_SAFE_BROWSING_API_KEY
from src.schemas import URLSafetyResult, URLSafetyVerdict

logger = logging.getLogger("nope.url_safety")

# ---------------------------------------------------------------------------
# URL extraction
# ---------------------------------------------------------------------------

_URL_REGEX = re.compile(
    r"https?://[^\s<>\"')\]]+|"           # full URLs
    r"(?<!\w)www\.[^\s<>\"')\]]+|"         # www. prefixed
    r"(?<!\w)bit\.ly/[^\s<>\"')\]]+|"      # bit.ly shortlinks
    r"(?<!\w)tinyurl\.com/[^\s<>\"')\]]+|" # tinyurl shortlinks
    r"(?<!\w)t\.co/[^\s<>\"')\]]+"         # t.co shortlinks
    , re.I
)

# ---------------------------------------------------------------------------
# Suspicious TLD list
# ---------------------------------------------------------------------------

SUSPICIOUS_TLDS = {
    ".tk", ".ml", ".ga", ".cf", ".gq",   # Free TLDs heavily abused
    ".xyz", ".top", ".work", ".click",    # Cheap TLDs often used in phishing
    ".buzz", ".icu", ".cam", ".rest",
}

# ---------------------------------------------------------------------------
# Lookalike domain detection
# ---------------------------------------------------------------------------

_TRUSTED_BRANDS = [
    "paypal", "amazon", "microsoft", "apple", "google", "facebook",
    "netflix", "chase", "wellsfargo", "bankofamerica", "citibank",
    "usps", "fedex", "ups", "irs", "ssa",
]

# Common character substitutions used in lookalike domains
# Some chars map to multiple letters, so we try all variants
_CHAR_SUBS: list[tuple[str, str]] = [
    ("0", "o"), ("1", "l"), ("1", "i"), ("3", "e"), ("4", "a"),
    ("5", "s"), ("8", "b"), ("@", "a"),
]

SHORTENER_DOMAINS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly",
    "is.gd", "buff.ly", "rb.gy", "shorturl.at", "tiny.cc",
}


def extract_urls(text: str) -> list[str]:
    """Extract URLs from text."""
    urls = _URL_REGEX.findall(text)
    # Normalize: add scheme if missing
    normalized = []
    for url in urls:
        url = url.rstrip(".,;:!?")
        if not url.startswith("http"):
            url = "https://" + url
        normalized.append(url)
    return normalized


# ---------------------------------------------------------------------------
# Pattern-based URL checks
# ---------------------------------------------------------------------------

def _check_suspicious_tld(parsed: urlparse) -> str | None:
    hostname = parsed.hostname or ""
    for tld in SUSPICIOUS_TLDS:
        if hostname.endswith(tld):
            return f"Suspicious TLD: {tld} — commonly used in phishing"
    return None


def _check_ip_based_url(parsed: urlparse) -> str | None:
    hostname = parsed.hostname or ""
    # Check for IP address as hostname
    if re.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", hostname):
        return "URL uses IP address instead of domain name — often a phishing sign"
    return None


def _check_lookalike_domain(parsed: urlparse) -> str | None:
    hostname = (parsed.hostname or "").lower()

    # Generate multiple normalized variants (since 1->l and 1->i are both valid)
    variants = {hostname}
    for fake_char, real_char in _CHAR_SUBS:
        new_variants = set()
        for v in variants:
            if fake_char in v:
                new_variants.add(v.replace(fake_char, real_char))
        variants.update(new_variants)

    for brand in _TRUSTED_BRANDS:
        # Brand appears in subdomain or path but isn't the real domain
        if brand in hostname and not hostname.endswith(f"{brand}.com") and not hostname.endswith(f"{brand}.gov"):
            return f"Possible lookalike domain impersonating {brand}"
        # Check normalized variants for character substitution attacks
        for variant in variants:
            if variant != hostname and brand in variant:
                return f"Possible lookalike domain using character substitution for {brand}"
    return None


def _check_shortened_url(parsed: urlparse) -> str | None:
    hostname = (parsed.hostname or "").lower()
    for shortener in SHORTENER_DOMAINS:
        if hostname == shortener or hostname.endswith("." + shortener):
            return f"Shortened URL ({shortener}) — real destination is hidden"
    return None


def _check_url_patterns(url: str) -> list[str]:
    """Run all pattern-based checks on a URL. Returns list of threat descriptions."""
    parsed = urlparse(url)
    threats = []

    for check in [_check_suspicious_tld, _check_ip_based_url, _check_lookalike_domain, _check_shortened_url]:
        result = check(parsed)
        if result:
            threats.append(result)

    # Excessively long URL (common in phishing)
    if len(url) > 200:
        threats.append("Unusually long URL — often used to hide malicious destination")

    # Multiple redirects embedded in URL
    if url.count("http") > 1:
        threats.append("URL contains embedded redirect — possible phishing technique")

    return threats


# ---------------------------------------------------------------------------
# URL expansion (resolve shortened links)
# ---------------------------------------------------------------------------

def _expand_url(url: str) -> str | None:
    """Attempt to expand a shortened URL by following redirects."""
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()

    is_shortened = any(hostname == s or hostname.endswith("." + s) for s in SHORTENER_DOMAINS)
    if not is_shortened:
        return None

    try:
        resp = requests.head(url, allow_redirects=True, timeout=5,
                             headers={"User-Agent": "ClearCheck/1.0"})
        final_url = resp.url
        if final_url != url:
            return final_url
    except Exception as e:
        logger.warning("Failed to expand URL %s: %s", url, e)
    return None


# ---------------------------------------------------------------------------
# Google Safe Browsing API
# ---------------------------------------------------------------------------

def _check_safe_browsing(urls: list[str]) -> dict[str, list[str]]:
    """Check URLs against Google Safe Browsing API. Returns {url: [threats]}."""
    if not GOOGLE_SAFE_BROWSING_API_KEY or not urls:
        return {}

    api_url = f"https://safebrowsing.googleapis.com/v4/threatMatches:find?key={GOOGLE_SAFE_BROWSING_API_KEY}"

    body = {
        "client": {"clientId": "clearcheck", "clientVersion": "1.0"},
        "threatInfo": {
            "threatTypes": ["MALWARE", "SOCIAL_ENGINEERING", "UNWANTED_SOFTWARE", "POTENTIALLY_HARMFUL_APPLICATION"],
            "platformTypes": ["ANY_PLATFORM"],
            "threatEntryTypes": ["URL"],
            "threatEntries": [{"url": u} for u in urls],
        },
    }

    try:
        resp = requests.post(api_url, json=body, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning("Google Safe Browsing API error: %s", e)
        return {}

    results: dict[str, list[str]] = {}
    for match in data.get("matches", []):
        url = match.get("threat", {}).get("url", "")
        threat_type = match.get("threatType", "UNKNOWN")
        threat_label = {
            "MALWARE": "Malware distribution site",
            "SOCIAL_ENGINEERING": "Phishing/social engineering site",
            "UNWANTED_SOFTWARE": "Distributes unwanted software",
            "POTENTIALLY_HARMFUL_APPLICATION": "Potentially harmful application",
        }.get(threat_type, f"Flagged by Safe Browsing: {threat_type}")
        results.setdefault(url, []).append(threat_label)

    return results


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def check_urls(text: str) -> URLSafetyResult | None:
    """Extract URLs from text and analyze them for safety. Returns None if no URLs found."""
    urls = extract_urls(text)
    if not urls:
        return None

    # Expand shortened URLs
    expanded_map: dict[str, str | None] = {}
    for url in urls:
        expanded_map[url] = _expand_url(url)

    # Build full list of URLs to check (original + expanded)
    all_urls = list(urls)
    for expanded in expanded_map.values():
        if expanded and expanded not in all_urls:
            all_urls.append(expanded)

    # Google Safe Browsing check
    safe_browsing_results = _check_safe_browsing(all_urls)

    # Build per-URL results
    verdicts: list[URLSafetyVerdict] = []
    any_unsafe = False

    for url in urls:
        threats: list[str] = []

        # Pattern-based checks
        threats.extend(_check_url_patterns(url))

        # Check expanded URL too
        expanded = expanded_map.get(url)
        if expanded:
            threats.extend(_check_url_patterns(expanded))

        # Safe Browsing results
        if url in safe_browsing_results:
            threats.extend(safe_browsing_results[url])
        if expanded and expanded in safe_browsing_results:
            threats.extend(safe_browsing_results[expanded])

        is_safe = len(threats) == 0
        if not is_safe:
            any_unsafe = True

        details = ""
        if expanded:
            details = f"Shortened URL expands to: {expanded}"

        verdicts.append(URLSafetyVerdict(
            url=url,
            is_safe=is_safe,
            threats=threats,
            expanded_url=expanded,
            details=details,
        ))

    return URLSafetyResult(
        urls_found=urls,
        results=verdicts,
        any_unsafe=any_unsafe,
    )
