PROVIDERS = {
    "OpenRouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "key_url": "https://openrouter.ai/keys",
        "headers": {
            "HTTP-Referer": "https://localhost:8501",
            "X-Title": "LitReviewBot",
        },
        "text_models": [
            "nvidia/nemotron-3-ultra-550b-a55b:free",
            "minimax/minimax-m3:free",
            "nvidia/nemotron-3-super-120b-a12b:free",
            "google/gemma-4-31b-it:free",
        ],
        "image_models": [
            "minimax/minimax-m3:free",
            "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
            "google/gemma-4-31b-it:free",
        ],
    },
    "AvalAI": {
        "base_url": "https://api.avalai.ir/v1",
        "key_url": "https://chat.avalai.ir/platform/home",
        "headers": {},
        "text_models": [
            "gemini-2.5-flash-lite",
            "gpt-4.1-nano",
        ],
        "image_models": [
            "gemini-2.5-flash-lite",
            "gpt-4.1-nano",
        ],
    },
    "Hormouz AI": {
        "base_url": "https://api.hormouz.net/v1",
        "key_url": "https://hormouz.net",
        "headers": {},
        "text_models": [
            "nvidia/nemotron-3-ultra-550b-a55b:free",
            "google/gemma-4-31b-it:free",
        ],
        "image_models": [
            "google/gemma-4-31b-it:free",
        ],
    },
}

# Backward compatibility aliases
TEXT_MODELS = PROVIDERS["OpenRouter"]["text_models"]
IMAGE_MODELS = PROVIDERS["OpenRouter"]["image_models"]
OPENROUTER_BASE_URL = PROVIDERS["OpenRouter"]["base_url"]
OPENROUTER_HEADERS = PROVIDERS["OpenRouter"]["headers"]

