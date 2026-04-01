import base64
from openai import OpenAI
from config import OPENROUTER_BASE_URL, OPENROUTER_HEADERS

def get_openrouter_client(api_key: str) -> OpenAI:
    """Initializes the OpenAI client pointing to OpenRouter."""
    return OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
    )

def call_openrouter(client: OpenAI, model: str, prompt: str, system_prompt: str = "You vary your tone based on instructions.", temperature: float = 0.5) -> str:
    """Standard call to a Text Model on OpenRouter."""
    try:
        response = client.chat.completions.create(
            extra_headers=OPENROUTER_HEADERS,
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error calling Main Text Model: {e}"

def encode_image(image_bytes: bytes) -> str:
    """Converts image bytes to base64 string."""
    return base64.b64encode(image_bytes).decode("utf-8")

def generate_image_caption(client: OpenAI, vision_model: str, image_bytes: bytes, context: str) -> str:
    """Send image and surrounding text context to Vision Model to generate a caption."""
    try:
        base64_image = encode_image(image_bytes)
        prompt = f"""
        You are an expert scientific image transcriber. 
        Here is the text found near the image below:
        ---
        {context}
        ---
        Look closely at the provided image. Does it correspond to a specific Figure or Table mentioned in the context?
        If so, name it (e.g., 'Figure 1'). Then, provide a detailed but concise textual caption/description of what the image/table shows, 
        including any key data points, trends, or variables.
        """
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
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            temperature=0.3,
            max_tokens=600
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[Image transcription failed: {e}]"
