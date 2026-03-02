"""
VLM Client for Sprint 3 — Self-Aware Failure Mining.
Wraps the Google Gemini API for image+text reasoning.

Usage:
    from src.vlm.vlm_client import VLMClient
    client = VLMClient(api_key="YOUR_KEY")
    response = client.query("path/to/image.jpg", "What objects do you see?")
"""
import os
import time
import json
import base64
from pathlib import Path

try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

from PIL import Image


class VLMClient:
    """
    Thin wrapper around Google Gemini Vision API.

    Args:
        api_key: Gemini API key. If None, reads from GEMINI_API_KEY env var.
        model_name: Gemini model to use.
        max_retries: Number of retries on transient errors.
        retry_delay: Seconds between retries.
    """
    def __init__(self, api_key=None, model_name='gemini-2.0-flash',
                 max_retries=3, retry_delay=2.0):
        if not HAS_GENAI:
            raise ImportError(
                "google-generativeai is not installed. "
                "Install with: pip install google-generativeai"
            )

        self.api_key = api_key or os.environ.get('GEMINI_API_KEY', '')
        if not self.api_key:
            raise ValueError(
                "Gemini API key required. Set GEMINI_API_KEY env var or pass api_key."
            )

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def query(self, image_path, prompt, temperature=0.3):
        """
        Send an image + text prompt to the VLM.

        Args:
            image_path: Path to the image file.
            prompt: Text prompt to send with the image.
            temperature: Sampling temperature (lower = more deterministic).

        Returns:
            response_text: The VLM's text response.
        """
        img = Image.open(image_path)

        for attempt in range(self.max_retries):
            try:
                response = self.model.generate_content(
                    [prompt, img],
                    generation_config=genai.types.GenerationConfig(
                        temperature=temperature,
                        max_output_tokens=1024,
                    )
                )
                return response.text
            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"  VLM query failed (attempt {attempt+1}): {e}")
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    print(f"  VLM query failed after {self.max_retries} attempts: {e}")
                    return f"[VLM_ERROR] {str(e)}"

    def query_structured(self, image_path, prompt, temperature=0.1):
        """
        Send an image + prompt and parse JSON from the response.

        Args:
            image_path: Path to the image file.
            prompt: Text prompt that instructs JSON output.
            temperature: Sampling temperature.

        Returns:
            parsed: dict/list parsed from JSON, or raw text if parsing fails.
        """
        raw = self.query(image_path, prompt, temperature=temperature)

        # Try to extract JSON from the response
        try:
            # Handle markdown code blocks
            if '```json' in raw:
                json_str = raw.split('```json')[1].split('```')[0].strip()
            elif '```' in raw:
                json_str = raw.split('```')[1].split('```')[0].strip()
            else:
                json_str = raw.strip()
            return json.loads(json_str)
        except (json.JSONDecodeError, IndexError):
            return {'raw_response': raw}

    def batch_query(self, image_paths, prompt, delay=1.0, temperature=0.3):
        """
        Query the VLM for multiple images with the same prompt.

        Args:
            image_paths: List of image file paths.
            prompt: Text prompt to send with each image.
            delay: Seconds between queries (rate limiting).

        Returns:
            results: List of (image_path, response_text) tuples.
        """
        results = []
        for i, img_path in enumerate(image_paths):
            print(f"  Querying VLM [{i+1}/{len(image_paths)}]: {Path(img_path).name}")
            response = self.query(img_path, prompt, temperature=temperature)
            results.append((str(img_path), response))
            if i < len(image_paths) - 1:
                time.sleep(delay)
        return results
