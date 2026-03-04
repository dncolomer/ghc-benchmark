"""Metrics and scoring functions for GHC Benchmark."""

import time
import random
import numpy as np
from typing import List, Dict
from .api import query_judgment_model, get_embedding
from .helpers import get_transcript_chunks
from .. import config


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)


def calculate_score(llm_outputs: List[str], transcript_chunks: List[str] = None) -> float:
    """Calculate similarity score between LLM outputs and transcript chunks."""
    if transcript_chunks is None:
        transcript_chunks = get_transcript_chunks()
    
    if not llm_outputs or not transcript_chunks:
        return 0.0
    
    all_texts = llm_outputs + transcript_chunks
    
    embeddings = []
    batch_size = 10
    for i in range(0, len(all_texts), batch_size):
        batch = all_texts[i:i+batch_size]
        for text in batch:
            emb = get_embedding(text)
            embeddings.append(emb)
        time.sleep(0.2)
    
    llm_embeds = embeddings[:len(llm_outputs)]
    transcript_embeds = embeddings[len(llm_outputs):]
    
    similarities = []
    for _ in range(config.NUM_COMPARISONS):
        llm_idx = random.randint(0, len(llm_embeds) - 1)
        trans_idx = random.randint(0, len(transcript_embeds) - 1)
        
        sim = cosine_similarity(llm_embeds[llm_idx], transcript_embeds[trans_idx])
        similarities.append(sim)
    
    return np.mean(similarities)


def calculate_linearity_score(text: str) -> float:
    """Calculate linearity score for a single text by chunking and evaluating transitions."""
    if not text or len(text.strip()) < 100:
        return 50.0
    
    chunk_size = len(text) // config.NUM_LINEARITY_CHUNKS
    chunks = []
    for i in range(config.NUM_LINEARITY_CHUNKS):
        start = i * chunk_size
        end = start + chunk_size if i < config.NUM_LINEARITY_CHUNKS - 1 else len(text)
        
        sentence_end = text.rfind('.', start, end)
        if sentence_end > start + 50:
            end = sentence_end + 1
        
        chunks.append(text[start:end].strip())
    
    if len(chunks) < 2:
        return 50.0
    
    scores = []
    for i in range(len(chunks) - 1):
        score = query_judgment_model(chunks[i], chunks[i + 1])
        scores.append(score)
        time.sleep(0.5)
    
    return float(np.mean(scores))


def calculate_cluster_score(text: str, linear_threshold: float = None) -> Dict:
    """Calculate cluster-based linearity score.
    
    Looks at clusters of consecutive linear transitions.
    - Single large cluster (>5 linear transitions) = LLM-like (robotic, sustained)
    - Many small clusters = human-like (fragmented, jumpy)
    """
    if linear_threshold is None:
        linear_threshold = config.LINEARITY_THRESHOLD
    
    if not text or len(text.strip()) < 100:
        return {"cluster_score": 50.0, "max_cluster": 0, "clusters": [], "all_transitions": []}
    
    chunk_size = len(text) // config.NUM_LINEARITY_CHUNKS
    chunks = []
    for i in range(config.NUM_LINEARITY_CHUNKS):
        start = i * chunk_size
        end = start + chunk_size if i < config.NUM_LINEARITY_CHUNKS - 1 else len(text)
        
        sentence_end = text.rfind('.', start, end)
        if sentence_end > start + 50:
            end = sentence_end + 1
        
        chunks.append(text[start:end].strip())
    
    if len(chunks) < 2:
        return {"cluster_score": 50.0, "max_cluster": 0, "clusters": [], "all_transitions": []}
    
    transitions = []
    for i in range(len(chunks) - 1):
        score = query_judgment_model(chunks[i], chunks[i + 1])
        is_linear = score >= linear_threshold
        transitions.append({"from_idx": i, "to_idx": i+1, "score": score, "is_linear": is_linear})
        time.sleep(0.5)
    
    clusters = []
    current_cluster = {"type": "linear" if transitions[0]["is_linear"] else "non_linear", "length": 1}
    
    for t in transitions[1:]:
        if t["is_linear"] == (current_cluster["type"] == "linear"):
            current_cluster["length"] += 1
        else:
            clusters.append(current_cluster)
            current_cluster = {"type": "linear" if t["is_linear"] else "non_linear", "length": 1}
    clusters.append(current_cluster)
    
    linear_clusters = [c for c in clusters if c["type"] == "linear"]
    non_linear_clusters = [c for c in clusters if c["type"] == "non_linear"]
    
    max_linear_cluster = max([c["length"] for c in linear_clusters], default=0)
    
    cluster_score = 0
    if max_linear_cluster >= 5:
        cluster_score = 90 + min((max_linear_cluster - 5) * 2, 10)
    elif max_linear_cluster >= 3:
        cluster_score = 60 + (max_linear_cluster - 3) * 10
    else:
        cluster_score = 20 + max_linear_cluster * 10
    
    cluster_score = min(cluster_score, 100)
    
    return {
        "cluster_score": float(cluster_score),
        "max_cluster": max_linear_cluster,
        "num_linear_clusters": len(linear_clusters),
        "num_non_linear_clusters": len(non_linear_clusters),
        "clusters": clusters,
        "all_transitions": transitions
    }
