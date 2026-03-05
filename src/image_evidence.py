"""Image evidence gathering: metadata extraction and reverse image search."""

from __future__ import annotations

import io
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

import requests
from PIL import Image, UnidentifiedImageError
from PIL.ExifTags import TAGS

Image.MAX_IMAGE_PIXELS = 50_000_000  # ~7000x7000 max — prevent decompression bombs
from tavily import TavilyClient

from src.config import SERPAPI_API_KEY, TAVILY_API_KEY
from src.retry import retry
from src.schemas import ImageEvidence, ImageMetadata, ReverseImageResult

logger = logging.getLogger("nope.image_evidence")


# ---------------------------------------------------------------------------
# Metadata extraction via Pillow
# ---------------------------------------------------------------------------

def extract_metadata(image_bytes: bytes) -> ImageMetadata:
    try:
        img = Image.open(io.BytesIO(image_bytes))
    except UnidentifiedImageError as e:
        logger.error("Cannot identify image format: %s", e)
        raise ValueError(f"Unrecognized image format: {e}") from e

    exif_data: dict = {}
    raw_exif = img.getexif()
    if raw_exif:
        for tag_id, value in raw_exif.items():
            tag_name = TAGS.get(tag_id, str(tag_id))
            try:
                if isinstance(value, bytes):
                    value = value.hex()
                exif_data[tag_name] = value
            except Exception:
                exif_data[tag_name] = str(value)

    return ImageMetadata(
        format=img.format,
        width=img.width,
        height=img.height,
        exif=exif_data,
        file_size_bytes=len(image_bytes),
    )


# ---------------------------------------------------------------------------
# Reverse image search (SerpAPI Google Lens or Tavily fallback)
# ---------------------------------------------------------------------------

def reverse_image_search(
    image_b64: str, user_context: str, image_url: str | None = None,
) -> list[ReverseImageResult]:
    if SERPAPI_API_KEY and image_url:
        return _serpapi_google_lens(image_url)
    return _tavily_fallback(user_context)


@retry(
    max_attempts=2,
    transient_exceptions=(ConnectionError, TimeoutError, requests.ConnectionError, requests.Timeout),
)
def _serpapi_google_lens(image_url: str) -> list[ReverseImageResult]:
    """Use SerpAPI Google Lens to find visually similar images."""
    params = {
        "engine": "google_lens",
        "url": image_url,
        "api_key": SERPAPI_API_KEY,
    }
    resp = requests.get(
        "https://serpapi.com/search", params=params, timeout=15
    )
    resp.raise_for_status()
    data = resp.json()

    results: list[ReverseImageResult] = []
    for match in data.get("visual_matches", [])[:5]:
        results.append(
            ReverseImageResult(
                title=match.get("title", ""),
                url=match.get("link", ""),
                content=match.get("snippet", match.get("title", "")),
                score=match.get("position", 0),
            )
        )
    return results


@retry(max_attempts=2, transient_exceptions=(ConnectionError, TimeoutError))
def _tavily_fallback(user_context: str) -> list[ReverseImageResult]:
    """Fall back to Tavily text search using user-provided context."""
    if not user_context:
        return []

    if not TAVILY_API_KEY:
        logger.warning("Tavily API key not configured, skipping reverse search fallback")
        return []

    client = TavilyClient(api_key=TAVILY_API_KEY)
    response = client.search(
        query=f"image verification: {user_context}",
        max_results=5,
        search_depth="advanced",
    )

    return [
        ReverseImageResult(
            title=r.get("title", ""),
            url=r.get("url", ""),
            content=r.get("content", ""),
            score=r.get("score", 0.0),
        )
        for r in response.get("results", [])
    ]


# ---------------------------------------------------------------------------
# Parallel evidence gathering
# ---------------------------------------------------------------------------

def gather_image_evidence(
    image_bytes: bytes, image_b64: str, user_context: str, image_url: str | None = None,
) -> ImageEvidence:
    """Run metadata extraction and reverse search in parallel."""
    evidence = ImageEvidence()
    errors: list[str] = []

    def _metadata():
        try:
            return extract_metadata(image_bytes)
        except ValueError as e:
            errors.append(f"Metadata extraction error: {e}")
            logger.error("Image metadata extraction failed: %s", e)
            return None
        except Exception as e:
            errors.append(f"Metadata extraction error: {e}")
            logger.exception("Unexpected error extracting image metadata")
            return None

    def _reverse_search():
        try:
            return reverse_image_search(image_b64, user_context, image_url=image_url)
        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else "unknown"
            errors.append(f"Reverse image search HTTP error ({status}): {e}")
            logger.error("Reverse image search HTTP %s: %s", status, e)
            return []
        except (requests.ConnectionError, requests.Timeout) as e:
            errors.append(f"Reverse image search connection error: {e}")
            logger.error("Reverse image search connection failed: %s", e)
            return []
        except Exception as e:
            errors.append(f"Reverse image search error: {e}")
            logger.exception("Unexpected error in reverse image search")
            return []

    with ThreadPoolExecutor(max_workers=2) as executor:
        metadata_future = executor.submit(_metadata)
        search_future = executor.submit(_reverse_search)

        try:
            evidence.metadata = metadata_future.result(timeout=20)
        except FuturesTimeoutError:
            errors.append("Metadata extraction timed out")
            logger.error("Metadata extraction exceeded 20s timeout")

        try:
            evidence.reverse_search_results = search_future.result(timeout=20)
        except FuturesTimeoutError:
            errors.append("Reverse image search timed out")
            logger.error("Reverse image search exceeded 20s timeout")

    evidence.errors = errors
    return evidence
