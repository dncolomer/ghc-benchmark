#!/usr/bin/env python3
"""Calibration test: test linearity and cluster metrics on human and LLM samples."""

import sys
import os
import json
sys.path.insert(0, os.path.dirname(__file__))

from benchmark import calculate_linearity_score, calculate_cluster_score

with open("transcript_cleaned.txt", "r") as f:
    transcript = f.read().strip()

with open("results/samples.json", "r") as f:
    samples = json.load(f)

print("=" * 60)
print("LINEARITY & CLUSTER CALIBRATION TEST")
print("=" * 60)

# Split transcript into chunks and test each
transcript_paras = [p.strip() for p in transcript.split('\n\n') if p.strip()]

test_chunks = []
# Take several substantial paragraphs
for para in transcript_paras[2:8]:  # Skip very short ones
    if len(para) > 500:
        test_chunks.append(("HUMAN", para[:2500]))
        if len(test_chunks) >= 4:
            break

# Add LLM samples
llm_model = "google/gemini-2.0-flash-001"
if llm_model in samples:
    for i, sample in enumerate(samples[llm_model]["completion"][:2] + samples[llm_model]["zero_shot"][:2]):
        test_chunks.append((f"LLM-{i+1}", sample[:2500]))

print(f"\nTesting {len(test_chunks)} text samples...\n")

results = []
for label, text in test_chunks:
    print(f"Testing {label} ({len(text)} chars)...")
    linearity_score = calculate_linearity_score(text)
    cluster_result = calculate_cluster_score(text)
    
    results.append({
        "label": label,
        "linearity": linearity_score,
        "cluster": cluster_result["cluster_score"],
        "max_cluster": cluster_result["max_cluster"],
        "num_linear_clusters": cluster_result["num_linear_clusters"],
    })
    
    print(f"  Linearity Score: {linearity_score:.1f}")
    print(f"  Cluster Score:   {cluster_result['cluster_score']:.1f} (max cluster: {cluster_result['max_cluster']}, linear clusters: {cluster_result['num_linear_clusters']})")
    print()

print("=" * 60)
print("SUMMARY")
print("=" * 60)

print("\nHuman samples:")
for r in results:
    if "HUMAN" in r["label"]:
        print(f"  {r['label']}: linearity={r['linearity']:.1f}, cluster={r['cluster']:.1f}, max_cluster={r['max_cluster']}")

print("\nLLM samples:")
for r in results:
    if "LLM" in r["label"]:
        print(f"  {r['label']}: linearity={r['linearity']:.1f}, cluster={r['cluster']:.1f}, max_cluster={r['max_cluster']}")

human_linearity = [r["linearity"] for r in results if "HUMAN" in r["label"]]
llm_linearity = [r["linearity"] for r in results if "LLM" in r["label"]]
human_cluster = [r["cluster"] for r in results if "HUMAN" in r["label"]]
llm_cluster = [r["cluster"] for r in results if "LLM" in r["label"]]

print("\nAverages:")
print(f"  Human Linearity:  {sum(human_linearity)/len(human_linearity):.1f}")
print(f"  LLM Linearity:    {sum(llm_linearity)/len(llm_linearity):.1f}")
print(f"  Human Cluster:    {sum(human_cluster)/len(human_cluster):.1f}")
print(f"  LLM Cluster:      {sum(llm_cluster)/len(llm_cluster):.1f}")