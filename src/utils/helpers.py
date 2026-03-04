"""Helper utilities for GHC Benchmark."""

import os
import json
from typing import Optional, List
from .. import config


def load_transcript(filepath: str = None) -> str:
    """Load transcript file content."""
    if filepath is None:
        filepath = config.TRANSCRIPT_FILE
    
    if not os.path.exists(filepath):
        print(f"Transcript file not found: {filepath}")
        return ""
    
    with open(filepath, 'r') as f:
        return f.read().strip()


def load_and_chunk_transcript(filepath: str = None, num_chunks: int = None) -> List[str]:
    """Load and chunk transcript file."""
    if filepath is None:
        filepath = config.TRANSCRIPT_FILE
    if num_chunks is None:
        num_chunks = config.NUM_CHUNKS
    
    if not os.path.exists(filepath):
        print(f"Transcript file not found: {filepath}")
        return []
    
    with open(filepath, 'r') as f:
        full_text = f.read().strip()
    
    chunk_size = len(full_text) // num_chunks
    chunks = []
    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size if i < num_chunks - 1 else len(full_text)
        
        paragraph_end = full_text.rfind('.', start, end)
        if paragraph_end > start + 50:
            end = paragraph_end + 1
        
        chunks.append(full_text[start:end].strip())
    
    return chunks


def get_transcript_chunks() -> List[str]:
    """Lazy-load transcript chunks."""
    if not hasattr(get_transcript_chunks, '_chunks'):
        get_transcript_chunks._chunks = load_and_chunk_transcript()
    return get_transcript_chunks._chunks


def load_results(reset: bool = False) -> dict:
    """Load existing results without overwriting."""
    if reset and os.path.exists(config.SCORES_FILE):
        os.remove(config.SCORES_FILE)
    elif os.path.exists(config.SCORES_FILE):
        with open(config.SCORES_FILE, "r") as f:
            return json.load(f)
    return {}


def save_results(results: dict):
    """Save results preserving existing entries."""
    os.makedirs(os.path.dirname(config.SCORES_FILE) if config.SCORES_FILE.startswith('results') else "results", exist_ok=True)
    with open(config.SCORES_FILE, "w") as f:
        json.dump(results, f, indent=2)


def load_samples(reset: bool = False) -> dict:
    """Load existing samples."""
    if reset:
        return {}
    if os.path.exists(config.SAMPLES_FILE):
        with open(config.SAMPLES_FILE, "r") as f:
            return json.load(f)
    return {}


def save_samples(samples: dict):
    """Save samples preserving existing entries."""
    os.makedirs(os.path.dirname(config.SAMPLES_FILE) if config.SAMPLES_FILE.startswith('results') else "results", exist_ok=True)
    with open(config.SAMPLES_FILE, "w") as f:
        json.dump(samples, f, indent=2)
