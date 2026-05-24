import os
from typing import Dict, List, Optional

import requests

SERPAPI_ENDPOINT = "https://serpapi.com/search.json"


def _is_public_http_url(image_ref: str) -> bool:
    return isinstance(image_ref, str) and image_ref.startswith(("http://", "https://"))


def _require_public_image_url(image_ref: str) -> str:
    if _is_public_http_url(image_ref):
        return image_ref

    raise ValueError(
        "Google Lens reverse-image search requires a public image URL. "
        "Local files are not directly supported by the Google Lens search API wrapper used here."
    )


def search_google_lens_visual_matches(
    image_ref: str,
    api_key: Optional[str] = None,
    limit: int = 10,
    hl: str = "en",
    gl: str = "us",
) -> Dict:
    """Search Google Lens for an image and return the raw SerpApi payload.

    This helper uses SerpApi's Google Lens engine, which expects a public image URL.
    """
    image_url = _require_public_image_url(image_ref)
    serpapi_api_key = _require_serpapi_api_key(api_key)
    params = {
        "engine": "google_lens",
        "url": image_url,
        "api_key": serpapi_api_key,
        "hl": hl,
        "gl": gl,
        "no_cache": "true",
    }

    response = requests.get(SERPAPI_ENDPOINT, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()

    if payload.get("error"):
        raise RuntimeError(f"SerpApi Google Lens error: {payload['error']}")

    visual_matches = payload.get("visual_matches") or []
    payload["visual_matches"] = visual_matches[: max(limit, 0)]
    return payload


def _require_serpapi_api_key(api_key: Optional[str] = None) -> str:
    serpapi_api_key = api_key or os.getenv("SERPAPI_API_KEY")
    if not serpapi_api_key:
        raise RuntimeError("Set SERPAPI_API_KEY before calling SerpApi search.")
    return serpapi_api_key


def search_google_images(
    query: str,
    api_key: Optional[str] = None,
    limit: int = 10,
    hl: str = "en",
    gl: str = "us",
    safe: str = "active",
) -> Dict:
    """Search Google Images through SerpApi and return the raw payload."""
    serpapi_api_key = _require_serpapi_api_key(api_key)
    payload: Dict = {"images_results": []}
    start = 0

    while len(payload["images_results"]) < max(limit, 0):
        params = {
            "engine": "google_images",
            "q": query,
            "api_key": serpapi_api_key,
            "hl": hl,
            "gl": gl,
            "safe": safe,
            "ijn": start // 100,
        }
        response = requests.get(SERPAPI_ENDPOINT, params=params, timeout=30)
        response.raise_for_status()
        page_payload = response.json()

        if page_payload.get("error"):
            raise RuntimeError(f"SerpApi Google Images error: {page_payload['error']}")

        page_results = page_payload.get("images_results") or []
        if not page_results:
            break

        payload["images_results"].extend(page_results)
        if len(page_results) < 100:
            break
        start += 100

    payload["images_results"] = payload["images_results"][: max(limit, 0)]
    return payload


def extract_visual_match_image_urls(payload: Dict, limit: int = 10) -> List[str]:
    """Extract the visually similar image URLs from a SerpApi Google Lens payload."""
    image_urls: List[str] = []

    for match in payload.get("visual_matches") or []:
        image_url = match.get("image")
        if image_url and image_url not in image_urls:
            image_urls.append(image_url)
        if len(image_urls) >= limit:
            break

    return image_urls


def fetch_similar_image_urls_from_google_lens(
    image_ref: str,
    api_key: Optional[str] = None,
    limit: int = 10,
    hl: str = "en",
    gl: str = "us",
) -> List[str]:
    """Return only similar-image URLs for a public image reference."""
    payload = search_google_lens_visual_matches(
        image_ref=image_ref,
        api_key=api_key,
        limit=limit,
        hl=hl,
        gl=gl,
    )
    return extract_visual_match_image_urls(payload, limit=limit)


def extract_google_image_result_urls(payload: Dict, limit: int = 10) -> List[str]:
    """Extract original image URLs from a SerpApi Google Images payload."""
    image_urls: List[str] = []

    for result in payload.get("images_results") or []:
        image_url = result.get("original") or result.get("image") or result.get("thumbnail")
        if image_url and image_url not in image_urls:
            image_urls.append(image_url)
        if len(image_urls) >= limit:
            break

    return image_urls


def fetch_google_image_search_urls(
    query: str,
    api_key: Optional[str] = None,
    limit: int = 10,
    hl: str = "en",
    gl: str = "us",
    safe: str = "active",
) -> List[str]:
    """Return image result URLs for a text query through SerpApi Google Images."""
    payload = search_google_images(
        query=query,
        api_key=api_key,
        limit=limit,
        hl=hl,
        gl=gl,
        safe=safe,
    )
    return extract_google_image_result_urls(payload, limit=limit)
