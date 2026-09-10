import base64
import logging
import re
import time
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from config import OPENROUTER_BASE_URL, OPENROUTER_HEADERS


def _sanitize_headers_ascii(headers: dict) -> dict:
    """HTTP header values must be ASCII-safe for some transports."""
    safe = {}
    for k, v in (headers or {}).items():
        key = str(k).encode("ascii", "ignore").decode("ascii")
        val = str(v).encode("ascii", "ignore").decode("ascii")
        safe[key] = val
    return safe


def _normalize_text_quotes(text: str) -> str:
    if text is None:
        return ""
    # Replace common smart punctuation that can trigger strict encoders.
    return (
        str(text)
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
    )

def get_llm_client(api_key: str, base_url: str = OPENROUTER_BASE_URL) -> OpenAI:
    """Initializes the OpenAI client pointing to any supported LLM provider."""
    return OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

def get_openrouter_client(api_key: str) -> OpenAI:
    """Initializes the OpenAI client pointing to OpenRouter."""
    return get_llm_client(api_key, OPENROUTER_BASE_URL)

def _safe_extract_chat_content(response) -> str | None:
    """Extract text content from a chat completion response."""
    if response is None:
        return None
    choices = getattr(response, "choices", None)
    if not choices or not isinstance(choices, (list, tuple)) or len(choices) == 0:
        return None
    choice = choices[0]
    if choice is None:
        return None
    message = getattr(choice, "message", None)
    if message is None:
        if isinstance(choice, dict):
            msg = choice.get("message")
            if isinstance(msg, dict):
                return _extract_message_text(msg.get("content"))
        return None
    content = getattr(message, "content", None)
    return _extract_message_text(content)

def call_openrouter(
    client: OpenAI,
    model: str,
    prompt: str,
    system_prompt: str = "You vary your tone based on instructions.",
    temperature: float = 0.5,
    max_tokens: int = 1200,
    extra_headers: dict | None = None,
    max_retries: int = 3,
    backoff_factor: float = 1.5,
) -> str:
    """Execute a chat completion request with automatic retries."""
    headers_to_send = extra_headers if extra_headers is not None else OPENROUTER_HEADERS
    safe_headers = _sanitize_headers_ascii(headers_to_send) if headers_to_send else None

    safe_system_prompt = _normalize_text_quotes(system_prompt)
    safe_prompt = _normalize_text_quotes(prompt)

    req_kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": safe_system_prompt},
            {"role": "user", "content": safe_prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if safe_headers:
        req_kwargs["extra_headers"] = safe_headers

    last_error = "Unknown error"

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(**req_kwargs)
            content = _safe_extract_chat_content(response)

            if content and content.strip():
                return content.strip()

            last_error = "Model response contained empty choices or null message content."
            if attempt < max_retries:
                time.sleep(backoff_factor ** attempt)
                continue

        except (APIConnectionError, APITimeoutError) as net_err:
            last_error = f"Connection/timeout error ({net_err.__class__.__name__}): {net_err}"
            if attempt < max_retries:
                time.sleep(backoff_factor ** attempt)
                continue

        except RateLimitError as rate_err:
            last_error = f"Rate limit exceeded (HTTP 429): {rate_err}"
            if attempt < max_retries:
                time.sleep(max(2.0, (backoff_factor ** attempt) * 2))
                continue

        except InternalServerError as srv_err:
            last_error = f"Provider internal server error (HTTP 5xx): {srv_err}"
            if attempt < max_retries:
                time.sleep(backoff_factor ** attempt)
                continue

        except UnicodeEncodeError:
            try:
                ascii_system_prompt = safe_system_prompt.encode("ascii", "ignore").decode("ascii")
                ascii_prompt = safe_prompt.encode("ascii", "ignore").decode("ascii")
                ascii_model = str(model).encode("ascii", "ignore").decode("ascii") or str(model)

                fallback_kwargs = {
                    "model": ascii_model,
                    "messages": [
                        {"role": "system", "content": ascii_system_prompt},
                        {"role": "user", "content": ascii_prompt}
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if safe_headers:
                    fallback_kwargs["extra_headers"] = safe_headers

                fallback_resp = client.chat.completions.create(**fallback_kwargs)
                fb_content = _safe_extract_chat_content(fallback_resp)
                if fb_content and fb_content.strip():
                    return fb_content.strip()
                last_error = "Model response contained empty content during ASCII fallback."
            except Exception as fb_err:
                last_error = f"ASCII encoding fallback failed: {fb_err}"

            if attempt < max_retries:
                time.sleep(backoff_factor ** attempt)
                continue

        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            if attempt < max_retries:
                time.sleep(backoff_factor ** attempt)
                continue

    return f"Error calling Main Text Model ({model}) after {max_retries} attempts: {last_error}"

def encode_image(image_bytes: bytes) -> str:
    """Converts image bytes to base64 string."""
    return base64.b64encode(image_bytes).decode("utf-8")


def _extract_message_text(content) -> str:
    """Normalize provider-specific content payloads into plain text."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                txt = item.get("text")
                if txt:
                    parts.append(str(txt))
            elif item:
                parts.append(str(item))
        return "\n".join(parts).strip()
    return "" if content is None else str(content).strip()


def _looks_incomplete(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    if len(t) < 120:
        return True
    # Common symptom of truncation: abrupt cut without terminal punctuation.
    if not t.endswith((".", "!", "?", "]", ")", "`", "\"", "'")):
        return True
    return False


def _coherent_caption(text: str) -> str:
    """Keep model output as one paragraph so it remains one retrieval unit."""
    text = (text or "").strip()
    text = text.replace("\r\n", "\n")
    text = re.sub(r"(?m)^\s*[-*•]\s+", "", text)
    text = re.sub(r"(?m)^\s*(?:caption|description)\s*:\s*", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def generate_image_caption(
    client: OpenAI,
    vision_model: str,
    image_bytes: bytes,
    context: str,
    image_media_type: str = "image/jpeg",
    extra_headers: dict | None = None,
    max_retries: int = 3,
) -> str:
    """Generate a caption for an image and nearby text context."""
    headers_to_send = extra_headers if extra_headers is not None else OPENROUTER_HEADERS
    safe_headers = _sanitize_headers_ascii(headers_to_send) if headers_to_send else None
    base64_image = encode_image(image_bytes)
    try:
        prompt = f"""
        You are an expert scientific image transcriber.
        Here is the text found near the image below in a scientific paper:
        ---
        {context}
        ---
        Look closely at the provided image. Write one coherent, self-contained paragraph
        describing what the image or table shows, including its figure/table number when
        confidently identifiable and the key data, trends, variables, and relationships.
        Do not use headings, bullets, numbered lists, line breaks, or Markdown.
        """

        last_error = "Unknown error"
        for attempt in range(1, max_retries + 1):
            try:
                req_kwargs = {
                    "model": vision_model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{image_media_type};base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    "temperature": 0.3,
                    "max_tokens": 1000,
                }
                if safe_headers:
                    req_kwargs["extra_headers"] = safe_headers

                response = client.chat.completions.create(**req_kwargs)
                text = _safe_extract_chat_content(response)

                if not text or text.strip().lower() in {"none", ""}:
                    last_error = "empty or null response from vision model"
                    time.sleep(1.5 * attempt)
                    continue

                # Check if output was cut off
                choices = getattr(response, "choices", None)
                choice = choices[0] if (choices and len(choices) > 0) else None
                finish_reason = getattr(choice, "finish_reason", None) if choice else None
                if finish_reason == "length" or _looks_incomplete(text):
                    cont_kwargs = {
                        "model": vision_model,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:{image_media_type};base64,{base64_image}"
                                        }
                                    }
                                ]
                            },
                            {"role": "assistant", "content": text},
                            {"role": "user", "content": "Continue from exactly where you stopped. Do not repeat earlier lines."}
                        ],
                        "temperature": 0.2,
                        "max_tokens": 500,
                    }
                    if safe_headers:
                        cont_kwargs["extra_headers"] = safe_headers

                    try:
                        continuation = client.chat.completions.create(**cont_kwargs)
                        cont_text = _safe_extract_chat_content(continuation)
                        if cont_text:
                            text = f"{text}\n{cont_text}".strip()
                    except Exception:
                        pass

                return _coherent_caption(text)
            except Exception as retry_error:
                last_error = str(retry_error)
                time.sleep(1.5 * attempt)

        return f"[Image transcription failed after {max_retries} attempts: {last_error}]"
    except Exception as e:
        return f"[Image transcription failed: {e}]"


def condense_query_with_history(
    client: OpenAI,
    model: str,
    chat_history: list,
    latest_query: str,
    extra_headers: dict | None = None,
) -> str:
    """Rewrite follow-up question into a standalone search query."""
    if not chat_history or not (latest_query or "").strip():
        return latest_query

    # Use the last few messages for query reformulation
    recent_history = chat_history[-4:]
    formatted_turns = []
    for msg in recent_history:
        role = str(msg.get("role", "user")).capitalize()
        content = str(msg.get("content", ""))[:400].strip()
        if content:
            formatted_turns.append(f"{role}: {content}")

    if not formatted_turns:
        return latest_query

    history_str = "\n".join(formatted_turns)
    prompt = f"""Given the following conversation history and a follow-up question from a researcher reading academic papers, rephrase the follow-up question to be a complete, standalone search query.
Include necessary context such as specific paper names, model names, algorithms, or experimental setups referenced earlier.
Do NOT answer the question. Do NOT include quotes or prefixes like "Standalone query:". Only return the rephrased search query.
If the question is already complete and standalone, return it exactly as is.

Conversation History:
{history_str}

Follow-up Question:
{latest_query}

Standalone Query:"""

    try:
        headers_to_send = extra_headers if extra_headers is not None else OPENROUTER_HEADERS
        safe_headers = _sanitize_headers_ascii(headers_to_send) if headers_to_send else None
        safe_prompt = _normalize_text_quotes(prompt)
        req_kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a concise search query reformulation assistant."},
                {"role": "user", "content": safe_prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 90,
        }
        if safe_headers:
            req_kwargs["extra_headers"] = safe_headers

        response = client.chat.completions.create(**req_kwargs)
        raw_text = _safe_extract_chat_content(response)
        if not raw_text:
            return latest_query
        rewritten = raw_text.strip()
        rewritten = re.sub(r'^(?:Standalone\s*Query\s*:\s*|Query\s*:\s*)', '', rewritten, flags=re.IGNORECASE).strip().strip('"\'')
        return rewritten if rewritten else latest_query
    except Exception:
        return latest_query

