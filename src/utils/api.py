"""API utilities for GHC Benchmark - OpenRouter interactions."""

import time
import requests
import numpy as np
from .. import config

REQUEST_DELAY = 2.0
ERROR_BACKOFF = [10, 30, 60, 120, 300]


def query_model(model: str, prompt: str, system_prompt: str, max_retries: int = 5) -> str:
    """Query an LLM via OpenRouter with aggressive retry logic."""
    time.sleep(REQUEST_DELAY)
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {config.API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/dncolomer/ghc-benchmark",
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 800,
                },
                timeout=90
            )
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            elif response.status_code == 429:
                wait_time = ERROR_BACKOFF[min(attempt, len(ERROR_BACKOFF)-1)]
                print(f"Rate limited (429), waiting {wait_time}s... (attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
            elif response.status_code == 401:
                wait_time = ERROR_BACKOFF[min(attempt, len(ERROR_BACKOFF)-1)]
                print(f"Auth error (401), waiting {wait_time}s... (attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                wait_time = ERROR_BACKOFF[min(attempt, len(ERROR_BACKOFF)-1)]
                print(f"Error {response.status_code}: {response.text[:100]}, waiting {wait_time}s...")
                time.sleep(wait_time)
        except Exception as e:
            wait_time = ERROR_BACKOFF[min(attempt, len(ERROR_BACKOFF)-1)]
            print(f"Attempt {attempt + 1}/{max_retries} failed: {e}, waiting {wait_time}s...")
            time.sleep(wait_time)
    return ""


def query_judgment_model(chunk_a: str, chunk_b: str) -> float:
    """Query LLM to judge linearity score between two text chunks."""
    import re
    
    time.sleep(REQUEST_DELAY)
    
    prompt = config.LINEARITY_JUDGE_PROMPT.format(
        NONLINEAR=config.LINEARITY_EXAMPLE_NONLINEAR,
        LINEAR=config.LINEARITY_EXAMPLE_LINEAR,
        CHUNK_A=chunk_a,
        CHUNK_B=chunk_b
    )
    
    for attempt in range(5):
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {config.API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/dncolomer/ghc-benchmark",
                },
                json={
                    "model": config.LINEARITY_JUDGE_MODEL,
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant that evaluates text linearity."},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 200,
                },
                timeout=90
            )
            if response.status_code == 200:
                content = response.json()["choices"][0]["message"]["content"]
                match = re.search(r'\b(\d{1,3})\b', content)
                if match:
                    score = int(match.group(1))
                    return min(max(score, 0), 100)
            elif response.status_code == 429:
                wait_time = ERROR_BACKOFF[min(attempt, len(ERROR_BACKOFF)-1)]
                print(f"Rate limited (429), waiting {wait_time}s...")
                time.sleep(wait_time)
            elif response.status_code == 401:
                wait_time = ERROR_BACKOFF[min(attempt, len(ERROR_BACKOFF)-1)]
                print(f"Auth error (401), waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                wait_time = ERROR_BACKOFF[min(attempt, len(ERROR_BACKOFF)-1)]
                print(f"Error {response.status_code}, waiting {wait_time}s...")
                time.sleep(wait_time)
        except Exception as e:
            wait_time = ERROR_BACKOFF[min(attempt, len(ERROR_BACKOFF)-1)]
            print(f"Attempt {attempt + 1} failed: {e}, waiting {wait_time}s...")
            time.sleep(wait_time)
    return 50


def get_embedding(text: str) -> np.ndarray:
    """Get text embedding via OpenRouter with retry logic."""
    time.sleep(REQUEST_DELAY)
    
    for attempt in range(5):
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {config.API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "sentence-transformers/all-MiniLM-L6-v2",
                    "input": text,
                },
                timeout=120
            )
            if response.status_code == 200:
                data = response.json()
                return np.array(data["data"][0]["embedding"])
            elif response.status_code in [429, 401]:
                wait_time = ERROR_BACKOFF[min(attempt, len(ERROR_BACKOFF)-1)]
                print(f"Embedding error ({response.status_code}), waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                wait_time = ERROR_BACKOFF[min(attempt, len(ERROR_BACKOFF)-1)]
                print(f"Embedding error {response.status_code}, waiting {wait_time}s...")
                time.sleep(wait_time)
        except Exception as e:
            wait_time = ERROR_BACKOFF[min(attempt, len(ERROR_BACKOFF)-1)]
            print(f"Embedding attempt {attempt+1} failed: {e}, waiting {wait_time}s...")
            time.sleep(wait_time)
    
    return np.zeros(384)
