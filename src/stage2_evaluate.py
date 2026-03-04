"""Stage 2: Evaluate LLM samples with benchmarks.

This stage runs the three benchmarks:
1. Similarity Benchmark: Embedding-based similarity to human transcript
2. Linearity Index: Sequential flow measurement (0-100)
3. Cluster Score: Sustained thinking patterns (>5 = LLM-like)
"""

import random
from . import config
from .utils.helpers import load_samples, save_samples, load_results, save_results, get_transcript_chunks
from .utils.metrics import calculate_score, calculate_linearity_score, calculate_cluster_score


def run_linearity_benchmark(generations: list) -> dict:
    """Run linearity benchmark on a list of LLM generations."""
    print(f"\n  Running linearity benchmark on {len(generations)} generations...")
    
    all_scores = []
    cluster_scores = []
    
    for i, generation in enumerate(generations):
        if len(generation) < 100:
            print(f"    Generation {i+1}/{len(generations)}: Skipping (too short)")
            continue
        
        print(f"    Generation {i+1}/{len(generations)}...", end=" ")
        
        linearity_score = calculate_linearity_score(generation)
        cluster_result = calculate_cluster_score(generation)
        
        all_scores.append(linearity_score)
        cluster_scores.append(cluster_result["cluster_score"])
        
        print(f"linearity: {linearity_score:.1f}, cluster: {cluster_result['cluster_score']:.1f}")
    
    if not all_scores:
        return {
            "linearly_index": {"raw_score": 50.0, "num_evaluated": 0},
            "cluster_index": {"raw_score": 50.0, "num_evaluated": 0}
        }
    
    return {
        "linearly_index": {
            "raw_score": float(sum(all_scores) / len(all_scores)),
            "num_evaluated": len(all_scores),
        },
        "cluster_index": {
            "raw_score": float(sum(cluster_scores) / len(cluster_scores)),
            "num_evaluated": len(cluster_scores),
        }
    }


def run_evaluate(models=None, reset: bool = False):
    """Run evaluation benchmarks on existing samples.
    
    Args:
        models: List of model names to evaluate (None = all)
        reset: Whether to reset existing scores
    """
    samples = load_samples()
    results = load_results(reset=reset)
    
    if not samples:
        print("No samples found. Run stage 1 first: python benchmark.py run --stage generate")
        return
    
    if models is None:
        models = list(samples.keys())
    
    print(f"Evaluating {len(models)} models...")
    
    for model in models:
        if model not in samples:
            print(f"  Warning: No samples for {model}")
            continue
        
        if model in results and "balanced_score" in results[model] and not reset:
            print(f"\nSkipping {model} - already evaluated")
            continue
        
        model_name = model.split("/")[-1]
        model_samples = samples[model]
        
        print(f"\n{'='*50}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*50}")
        
        completion_outputs = model_samples.get("completion", [])
        zero_shot_outputs = model_samples.get("zero_shot", [])
        
        model_results = {}
        
        transcript_chunks = get_transcript_chunks()
        
        if completion_outputs:
            completion_score = calculate_score(completion_outputs, transcript_chunks)
            completion_normalized = (completion_score + 1) / 2 * 100
            model_results["completion"] = {
                "raw_score": float(completion_score),
                "normalized_score": float(completion_normalized),
                "num_outputs": len(completion_outputs),
            }
            
            completion_sample_count = min(config.SAMPLES_PER_MODEL, len(completion_outputs))
            model_results["completion_samples"] = random.sample(completion_outputs, completion_sample_count)
        
        if zero_shot_outputs:
            zero_shot_score = calculate_score(zero_shot_outputs, transcript_chunks)
            zero_shot_normalized = (zero_shot_score + 1) / 2 * 100
            model_results["zero_shot"] = {
                "raw_score": float(zero_shot_score),
                "normalized_score": float(zero_shot_normalized),
                "num_outputs": len(zero_shot_outputs),
            }
            
            zero_shot_sample_count = min(config.SAMPLES_PER_MODEL, len(zero_shot_outputs))
            model_results["zero_shot_samples"] = random.sample(zero_shot_outputs, zero_shot_sample_count)
        
        all_generations = completion_outputs + zero_shot_outputs
        if all_generations:
            linearity_result = run_linearity_benchmark(all_generations)
            model_results["linearly_index"] = linearity_result["linearly_index"]
            model_results["cluster_index"] = linearity_result["cluster_index"]
        
        if "completion" in model_results and "zero_shot" in model_results:
            balanced = (model_results["completion"]["normalized_score"] + model_results["zero_shot"]["normalized_score"]) / 2
            model_results["balanced_score"] = float(balanced)
        
        if "linearly_index" in model_results and "balanced_score" in model_results:
            inverted_linearity = 100 - model_results["linearly_index"]["raw_score"]
            if "cluster_index" in model_results:
                inverted_cluster = 100 - model_results["cluster_index"]["raw_score"]
                combined = (model_results["balanced_score"] + inverted_linearity + inverted_cluster) / 3
            else:
                combined = (model_results["balanced_score"] + inverted_linearity) / 2
            model_results["combined_score"] = float(combined)
        
        results[model] = model_results
        
        save_results(results)
        
        if "balanced_score" in model_results:
            print(f"\n{model_name}:")
            print(f"  Completion:   {model_results['completion']['normalized_score']:.2f}%")
            print(f"  Zero-shot:    {model_results['zero_shot']['normalized_score']:.2f}%")
            print(f"  Balanced:     {model_results['balanced_score']:.2f}%")
            if "linearly_index" in model_results:
                print(f"  Linearity:    {model_results['linearly_index']['raw_score']:.2f}%")
                print(f"  Cluster:     {model_results['cluster_index']['raw_score']:.2f}%")
                print(f"  Combined:     {model_results['combined_score']:.2f}%")
    
    print("\n" + "="*50)
    print("EVALUATION COMPLETE")
    print("="*50)
    print(f"Scores saved to: {config.SCORES_FILE}")
    
    return results


if __name__ == "__main__":
    run_evaluate()
