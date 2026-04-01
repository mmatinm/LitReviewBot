VISION_MODELS = [
    "nvidia/nemotron-nano-12b-v2-vl:free",
    "google/gemma-3-27b-it:free",
    "google/gemma-3-4b-it:free",
    "google/lyria-3-pro-preview",
    "google/gemma-3-12b-it:free",
    "google/lyria-3-clip-preview",
]

TEXT_MODELS = [
    "qwen/qwen3.6-plus-preview:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "stepfun/step-3.5-flash:free",
]

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://localhost:8501", 
    "X-Title": "LitReviewBot"
}
