"""
Direct Visual Question Answering helpers for Promptomatix.

This module intentionally avoids the optimization/configuration flow. It is a
small smoke-test path for sending one image plus one question to a vision model.
"""

import base64
import mimetypes
import os
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Union
from urllib.parse import urlparse, urlunparse
from urllib.request import Request, urlopen


ImageInput = Union[str, os.PathLike]


DEFAULT_VQA_SYSTEM_PROMPT = (
    "Answer the user's visual question using only evidence from the image. "
    "If the image does not contain enough information, say that explicitly."
)


def answer_visual_question(
    image: Union[ImageInput, Sequence[ImageInput]],
    prompt: str,
    *,
    model_name: Optional[str] = None,
    model_provider: Optional[str] = None,
    model_api_key: Optional[str] = None,
    model_api_base: Optional[str] = None,
    system_prompt: Optional[str] = DEFAULT_VQA_SYSTEM_PROMPT,
    temperature: float = 0.0,
    max_tokens: int = 512,
    timeout: float = 60.0,
) -> str:
    """Ask a VLM a question about one or more images.

    Args:
        image: Local image path, image URL, data URL, or a sequence of those.
        prompt: The visual question/instruction to answer.
        model_name: Vision-capable model name. Defaults to ``VQA_MODEL`` or
            ``OPTIMIZER_MODEL`` env vars, then provider-specific defaults.
        model_provider: ``openai``, ``local``, ``databricks``, ``togetherai``,
            ``gemini``, or ``anthropic``. Defaults to ``VQA_MODEL_PROVIDER`` or
            ``OPTIMIZER_PROVIDER``, then ``openai``.
        model_api_key: API key. For local OpenAI-compatible endpoints this can
            be any non-empty value; ``EMPTY`` is used by default.
        model_api_base: Base URL for OpenAI-compatible/local endpoints.
        system_prompt: Optional system instruction.
        temperature: Model sampling temperature.
        max_tokens: Maximum answer tokens.
        timeout: Request timeout in seconds.

    Returns:
        The model's text answer.
    """
    if not prompt or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")

    images = _coerce_images(image)
    if not images:
        raise ValueError("image must contain at least one image path, URL, or data URL")

    provider = (model_provider or os.getenv("VQA_MODEL_PROVIDER") or os.getenv("OPTIMIZER_PROVIDER") or "openai").lower()
    model = model_name or os.getenv("VQA_MODEL") or os.getenv("OPTIMIZER_MODEL") or _default_model_for_provider(provider)

    if provider in {"openai", "local", "databricks", "togetherai"}:
        api_base = (
            model_api_base
            or os.getenv("VQA_API_BASE")
            or os.getenv("OPTIMIZER_API_BASE")
            or os.getenv("LOCAL_VLM_API_BASE")
            or _chat_completions_url_to_base(os.getenv("LOCAL_VLM_URL"))
        )
        api_key = (
            model_api_key
            or os.getenv("VQA_API_KEY")
            or os.getenv("OPTIMIZER_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or ("EMPTY" if provider == "local" or api_base else None)
        )
        return _call_openai_compatible_vqa(
            prompt=prompt,
            images=images,
            model_name=model,
            api_key=api_key,
            api_base=api_base,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    if provider == "gemini":
        api_key = model_api_key or os.getenv("VQA_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        return _call_gemini_vqa(
            prompt=prompt,
            images=images,
            model_name=model,
            api_key=api_key,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if provider == "anthropic":
        api_key = model_api_key or os.getenv("VQA_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        return _call_anthropic_vqa(
            prompt=prompt,
            images=images,
            model_name=model,
            api_key=api_key,
            api_base=model_api_base or os.getenv("VQA_API_BASE"),
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    raise ValueError(f"Unsupported VQA model_provider: {provider}")


def image_to_data_url(image: ImageInput) -> str:
    """Convert a local path, image URL, or data URL into a base64 data URL."""
    image_str = str(image)
    if _is_data_url(image_str):
        return image_str

    image_bytes = _read_image_bytes(image_str)
    media_type = _guess_media_type(image_str, image_bytes)
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def _call_openai_compatible_vqa(
    *,
    prompt: str,
    images: Sequence[ImageInput],
    model_name: str,
    api_key: Optional[str],
    api_base: Optional[str],
    system_prompt: Optional[str],
    temperature: float,
    max_tokens: int,
    timeout: float,
) -> str:
    if not api_key:
        raise ValueError("A VQA/OpenAI-compatible API key is required. Set VQA_API_KEY, OPTIMIZER_API_KEY, or OPENAI_API_KEY.")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("The openai package is required for OpenAI-compatible VQA calls.") from exc

    client_kwargs = {"api_key": api_key, "timeout": timeout}
    base_url = _normalize_openai_base_url(api_base)
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": build_openai_vision_content(prompt, images)})

    response = client.chat.completions.create(
        model=_clean_model_name(model_name),
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return _extract_openai_message_text(response)


def _call_gemini_vqa(
    *,
    prompt: str,
    images: Sequence[ImageInput],
    model_name: str,
    api_key: Optional[str],
    system_prompt: Optional[str],
    temperature: float,
    max_tokens: int,
) -> str:
    if not api_key:
        raise ValueError("A Gemini API key is required. Set VQA_API_KEY, GEMINI_API_KEY, or GOOGLE_API_KEY.")

    try:
        import google.generativeai as genai
        from PIL import Image
    except ImportError as exc:
        raise ImportError("google-generativeai and Pillow are required for Gemini VQA calls.") from exc

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=_clean_model_name(model_name),
        system_instruction=system_prompt,
        generation_config={
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        },
    )

    contents: List[Any] = [prompt]
    for img in images:
        image_bytes = _read_image_bytes_or_data_url(str(img))
        contents.append(Image.open(BytesIO(image_bytes)))

    response = model.generate_content(contents)
    return getattr(response, "text", "").strip()


def _call_anthropic_vqa(
    *,
    prompt: str,
    images: Sequence[ImageInput],
    model_name: str,
    api_key: Optional[str],
    api_base: Optional[str],
    system_prompt: Optional[str],
    temperature: float,
    max_tokens: int,
    timeout: float,
) -> str:
    if not api_key:
        raise ValueError("An Anthropic API key is required. Set VQA_API_KEY or ANTHROPIC_API_KEY.")

    try:
        import anthropic
    except ImportError as exc:
        raise ImportError("The anthropic package is required for Anthropic VQA calls.") from exc

    client_kwargs = {"api_key": api_key, "timeout": timeout}
    if api_base:
        client_kwargs["base_url"] = api_base
    client = anthropic.Anthropic(**client_kwargs)

    content = [{"type": "text", "text": prompt}]
    for img in images:
        image_str = str(img)
        image_bytes = _read_image_bytes_or_data_url(image_str)
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": _guess_media_type(image_str, image_bytes),
                    "data": base64.b64encode(image_bytes).decode("ascii"),
                },
            }
        )

    kwargs = {
        "model": _clean_model_name(model_name),
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": content}],
    }
    if system_prompt:
        kwargs["system"] = system_prompt

    response = client.messages.create(**kwargs)
    return "\n".join(getattr(block, "text", "") for block in response.content).strip()


def build_openai_vision_content(prompt: str, images: Sequence[ImageInput]) -> List[dict]:
    """Build OpenAI-compatible multimodal chat message content."""
    content = [{"type": "text", "text": prompt}]
    for img in images:
        image_str = str(img)
        if _is_url(image_str) or _is_data_url(image_str):
            image_url = image_str
        else:
            image_url = image_to_data_url(image_str)
        content.append({"type": "image_url", "image_url": {"url": image_url}})
    return content


def _coerce_images(image: Union[ImageInput, Sequence[ImageInput]]) -> List[ImageInput]:
    if isinstance(image, (str, os.PathLike)):
        return [image]
    if isinstance(image, Iterable):
        return list(image)
    raise TypeError("image must be a path, URL, data URL, or a sequence of those")


def _read_image_bytes_or_data_url(image: str) -> bytes:
    if _is_data_url(image):
        _, data = image.split(",", 1)
        return base64.b64decode(data)
    return _read_image_bytes(image)


def _read_image_bytes(image: str) -> bytes:
    if _is_url(image):
        request = Request(image, headers={"User-Agent": "promptomatix-vqa/0.1"})
        with urlopen(request, timeout=30) as response:
            return response.read()

    path = Path(image).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    return path.read_bytes()


def _guess_media_type(image: str, image_bytes: Optional[bytes] = None) -> str:
    if _is_data_url(image):
        return image.split(";", 1)[0].replace("data:", "", 1)

    media_type, _ = mimetypes.guess_type(str(image))
    if media_type and media_type.startswith("image/"):
        return media_type

    if image_bytes:
        if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if image_bytes.startswith(b"\xff\xd8"):
            return "image/jpeg"
        if image_bytes.startswith(b"GIF87a") or image_bytes.startswith(b"GIF89a"):
            return "image/gif"
        if image_bytes[:12].endswith(b"WEBP"):
            return "image/webp"

    return "image/jpeg"


def _extract_openai_message_text(response: Any) -> str:
    content = response.choices[0].message.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(getattr(item, "text", "")))
        return "\n".join(part for part in parts if part).strip()
    return str(content).strip()


def _default_model_for_provider(provider: str) -> str:
    if provider == "gemini":
        return "gemini-1.5-flash"
    if provider == "anthropic":
        return "claude-3-5-sonnet-latest"
    if provider == "local":
        return "Qwen/Qwen2-VL-7B-Instruct"
    return "gpt-4o-mini"


def _clean_model_name(model_name: str) -> str:
    for prefix in ("openai/", "gemini/", "anthropic/"):
        if model_name.startswith(prefix):
            return model_name[len(prefix):]
    return model_name


def _normalize_openai_base_url(api_base: Optional[str]) -> Optional[str]:
    if not api_base:
        return None
    return _chat_completions_url_to_base(api_base) or api_base


def _chat_completions_url_to_base(url: Optional[str]) -> Optional[str]:
    if not url:
        return None

    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return url

    path = parsed.path.rstrip("/")
    if path.endswith("/chat/completions"):
        path = path[: -len("/chat/completions")]
    return urlunparse(parsed._replace(path=path or "", params="", query="", fragment=""))


def _is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"}


def _is_data_url(value: str) -> bool:
    return value.startswith("data:image/")
