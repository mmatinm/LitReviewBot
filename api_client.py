import base64
from openai import OpenAI
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

def get_openrouter_client(api_key: str) -> OpenAI:
    """Initializes the OpenAI client pointing to OpenRouter."""
    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
    )

def call_openrouter(
    client: OpenAI,
    model: str,
    prompt: str,
    system_prompt: str = "You vary your tone based on instructions.",
    temperature: float = 0.5,
    max_tokens: int = 1200,
) -> str:
    """Standard call to a Text Model on OpenRouter."""
    try:
        safe_headers = _sanitize_headers_ascii(OPENROUTER_HEADERS)
        safe_system_prompt = _normalize_text_quotes(system_prompt)
        safe_prompt = _normalize_text_quotes(prompt)

        response = client.chat.completions.create(
            extra_headers=safe_headers,
            model=model,
            messages=[
                {"role": "system", "content": safe_system_prompt},
                {"role": "user", "content": safe_prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
    except UnicodeEncodeError:
        # Hard fallback for environments/transports that still enforce ASCII.
        try:
            safe_headers = _sanitize_headers_ascii(OPENROUTER_HEADERS)
            ascii_system_prompt = _normalize_text_quotes(system_prompt).encode("ascii", "ignore").decode("ascii")
            ascii_prompt = _normalize_text_quotes(prompt).encode("ascii", "ignore").decode("ascii")
            ascii_model = str(model).encode("ascii", "ignore").decode("ascii") or str(model)

            response = client.chat.completions.create(
                extra_headers=safe_headers,
                model=ascii_model,
                messages=[
                    {"role": "system", "content": ascii_system_prompt},
                    {"role": "user", "content": ascii_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error calling Main Text Model: {e}"
    except Exception as e:
        return f"Error calling Main Text Model: {e}"

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

def generate_image_caption(
    client: OpenAI,
    vision_model: str,
    image_bytes: bytes,
    context: str,
    image_media_type: str = "image/jpeg",
) -> str:
    """Send image and surrounding text context to Vision Model to generate a caption."""
    base64_image = encode_image(image_bytes)
    try:
        prompt = f"""
        You are an expert scientific image transcriber. 
        Here is the text found near the image below in a scientific paper:
        ---
        {context}
        ---
        Look closely at the provided image. Does it correspond to a specific Figure or Table mentioned in the context?
        If so, name it (e.g., 'Figure 1'). Then, provide a detailed but concise textual caption/description of what the image/table shows, 
        including any key data points, trends, or variables.
        """

        last_error = None
        for _ in range(2):
            try:
                response = client.chat.completions.create(
                    extra_headers=OPENROUTER_HEADERS,
                    model=vision_model,
                    messages=[
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
                    temperature=0.3,
                    max_tokens=1000
                )

                choice = response.choices[0]
                text = _extract_message_text(choice.message.content)

                if not text or text.strip().lower() == "none":
                    last_error = "empty response from model"
                    continue

                finish_reason = getattr(choice, "finish_reason", None)
                if finish_reason == "length" or _looks_incomplete(text):
                    # Ask the model to continue so outputs are not cut mid-sentence.
                    continuation = client.chat.completions.create(
                        extra_headers=OPENROUTER_HEADERS,
                        model=vision_model,
                        messages=[
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
                        temperature=0.2,
                        max_tokens=500
                    )
                    cont_text = _extract_message_text(continuation.choices[0].message.content)
                    if cont_text:
                        text = f"{text}\n{cont_text}".strip()

                return text
            except Exception as retry_error:
                last_error = retry_error

        return f"[Image transcription failed: {last_error}]"
    except Exception as e:
        return f"[Image transcription failed: {e}]"
