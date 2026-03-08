"""
VLM Client for Sprint 3 — Self-Aware Failure Mining.
Supports two backends:
  - OpenRouter  (default): OpenAI-compatible API, access to many vision models
  - Gemini:                Google Generative AI API

Usage:
    # OpenRouter (recommended — use any vision model)
    client = VLMClient(api_key="sk-or-...", backend="openrouter",
                       model_name="google/gemini-2.0-flash-exp:free")

    # Gemini direct
    client = VLMClient(api_key="AIza...", backend="gemini")

    response = client.query("path/to/image.jpg", "What do you see?")
    parsed   = client.query_structured("path/to/image.jpg", FAILURE_PROMPT)
"""
import os
import time
import json
import base64
from pathlib import Path
from PIL import Image

# ── OpenRouter / OpenAI-compatible backend ────────────────────────────────────
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# ── Gemini backend ─────────────────────────────────────────────────────────────
try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False


def _image_to_b64(image_path, max_size=1024):
    """Resize image and encode as base64 JPEG string for API calls."""
    img = Image.open(image_path).convert("RGB")
    img.thumbnail((max_size, max_size))
    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class VLMClient:
    """
    Unified VLM client supporting OpenRouter and Gemini backends.

    Args:
        api_key:     API key. Defaults to OPENROUTER_API_KEY or GEMINI_API_KEY env var.
        backend:     'openrouter' (default) or 'gemini'.
        model_name:  Model identifier.
                     OpenRouter examples:
                       'google/gemini-2.0-flash-exp:free'   (free tier on OpenRouter)
                       'google/gemini-flash-1.5-8b'
                       'meta-llama/llama-3.2-11b-vision-instruct:free'
                       'anthropic/claude-3-haiku'
                     Gemini examples:
                       'gemini-2.0-flash'
        max_retries: Retry count on transient failures.
        retry_delay: Base delay between retries (seconds, exponential backoff).
        site_url:    (OpenRouter only) Your site URL shown in OR dashboard.
    """

    OPENROUTER_BASE = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        api_key=None,
        backend="openrouter",
        model_name=None,
        max_retries=3,
        retry_delay=2.0,
        site_url="https://prosit2-vision",
    ):
        self.backend = backend.lower()
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        if self.backend == "openrouter":
            if not HAS_OPENAI:
                raise ImportError(
                    "openai package not installed. Run: pip install openai"
                )
            self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
            if not self.api_key:
                raise ValueError(
                    "OpenRouter API key required. "
                    "Set OPENROUTER_API_KEY env var or pass api_key."
                )
            self.model_name = model_name or "google/gemini-2.0-flash-exp:free"
            self._client = OpenAI(
                base_url=self.OPENROUTER_BASE,
                api_key=self.api_key,
                default_headers={
                    "HTTP-Referer": site_url,
                    "X-Title": "PROSIT2-FailureMiner",
                },
            )

        elif self.backend == "gemini":
            if not HAS_GENAI:
                raise ImportError(
                    "google-generativeai not installed. "
                    "Run: pip install google-generativeai"
                )
            self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
            if not self.api_key:
                raise ValueError(
                    "Gemini API key required. "
                    "Set GEMINI_API_KEY env var or pass api_key."
                )
            self.model_name = model_name or "gemini-2.0-flash"
            genai.configure(api_key=self.api_key)
            self._genai_model = genai.GenerativeModel(self.model_name)

        else:
            raise ValueError(f"Unknown backend '{backend}'. Use 'openrouter' or 'gemini'.")

        print(f"VLMClient ready: backend={self.backend}  model={self.model_name}")

    # ── Core query ────────────────────────────────────────────────────────────

    def query(self, image_path, prompt, temperature=0.1):
        """
        Send an image + text prompt to the VLM.

        Args:
            image_path:  Path to image file.
            prompt:      Text prompt.
            temperature: Sampling temperature (lower = more deterministic).

        Returns:
            str: The VLM's text response, or '[VLM_ERROR] ...' on failure.
        """
        for attempt in range(self.max_retries):
            try:
                if self.backend == "openrouter":
                    return self._query_openrouter(image_path, prompt, temperature)
                else:
                    return self._query_gemini(image_path, prompt, temperature)
            except Exception as e:
                wait = self.retry_delay * (attempt + 1)
                if attempt < self.max_retries - 1:
                    print(f"  VLM attempt {attempt+1} failed: {e}. Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"  VLM failed after {self.max_retries} attempts: {e}")
                    return f"[VLM_ERROR] {str(e)}"

    def _query_openrouter(self, image_path, prompt, temperature):
        """OpenRouter / OpenAI-compatible vision call."""
        b64 = _image_to_b64(image_path)
        response = self._client.chat.completions.create(
            model=self.model_name,
            temperature=temperature,
            max_tokens=512,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        return response.choices[0].message.content

    def _query_gemini(self, image_path, prompt, temperature):
        """Native Gemini API call."""
        img = Image.open(image_path)
        response = self._genai_model.generate_content(
            [prompt, img],
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=512,
            ),
        )
        return response.text

    # ── Structured (JSON) query ───────────────────────────────────────────────

    def query_structured(self, image_path, prompt, temperature=0.1):
        """
        Query the VLM and parse JSON from the response.

        Returns:
            dict parsed from JSON, or {'raw_response': raw_text} on parse failure.
        """
        raw = self.query(image_path, prompt, temperature=temperature)
        if raw.startswith("[VLM_ERROR]"):
            return {"error": raw}
        try:
            if "```json" in raw:
                json_str = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                json_str = raw.split("```")[1].split("```")[0].strip()
            else:
                json_str = raw.strip()
            return json.loads(json_str)
        except (json.JSONDecodeError, IndexError):
            return {"raw_response": raw}

    # ── Batch query ───────────────────────────────────────────────────────────

    def batch_query(self, image_paths, prompt, delay=1.2, temperature=0.1):
        """
        Query VLM for multiple images with the same prompt.

        Args:
            image_paths: List of image file paths.
            prompt:      Prompt sent with each image.
            delay:       Seconds between queries (rate limit protection).

        Returns:
            List of (image_path, response_text) tuples.
        """
        results = []
        for i, img_path in enumerate(image_paths):
            print(f"  [{i+1}/{len(image_paths)}] {Path(img_path).name}")
            resp = self.query(img_path, prompt, temperature=temperature)
            results.append((str(img_path), resp))
            if i < len(image_paths) - 1:
                time.sleep(delay)
        return results
