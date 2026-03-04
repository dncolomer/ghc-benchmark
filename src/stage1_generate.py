"""Stage 1: Generate LLM samples via OpenRouter.

This stage generates completions and zero-shot responses from various LLM models.
"""

import time
import random
from . import config
from .utils.api import query_model
from .utils.helpers import get_transcript_chunks, load_samples, save_samples, load_results, save_results


def run_completion_benchmark(model: str, model_name: str) -> list:
    """Run completion variant: uses transcript chunks as starting points."""
    print(f"\n  Running completion variant...")
    
    transcript_chunks = get_transcript_chunks()
    llm_outputs = []
    
    for i in range(config.NUM_QUERIES):
        chunk_idx = i % len(transcript_chunks)
        starting_point = transcript_chunks[chunk_idx]
        
        prompt = f"""{config.SYSTEM_PROMPT}

{starting_point}

Continue thinking out loud about this concept:"""
        
        print(f"    Query {i+1}/{config.NUM_QUERIES}...", end=" ")
        output = query_model(model, prompt, config.SYSTEM_PROMPT)
        
        if output:
            llm_outputs.append(output)
            print("✓")
        else:
            print("✗ (failed - will retry on next run)")
        
        time.sleep(3)
    
    return llm_outputs


def run_zero_shot_benchmark(model: str, model_name: str) -> list:
    """Run zero-shot variant: no transcript chunk context."""
    print(f"\n  Running zero-shot variant...")
    
    zero_shot_prompts = [
        "What is Grover's algorithm and how does it work?",
        "Explain quantum search algorithms.",
        "How does quantum computing speed up search problems?",
        "What is the difference between classical and quantum search?",
        "Explain superposition and how it relates to searching.",
    ]
    
    llm_outputs = []
    
    for i in range(config.NUM_QUERIES):
        prompt_idx = i % len(zero_shot_prompts)
        
        print(f"    Query {i+1}/{config.NUM_QUERIES}...", end=" ")
        output = query_model(model, config.ZERO_SHOT_PROMPT, config.ZERO_SHOT_PROMPT)
        
        if output:
            llm_outputs.append(output)
            print("✓")
        else:
            print("✗ (failed - will retry on next run)")
        
        time.sleep(3)
    
    return llm_outputs


def run_generate(models=None, reset: bool = False):
    """Run sample generation for all models.
    
    Args:
        models: List of model names to generate for (None = all)
        reset: Whether to reset existing samples
    """
    if models is None:
        models = config.MODELS
    
    samples = load_samples(reset=reset)
    results = load_results(reset=reset)
    
    for model in models:
        model_name = model.split("/")[-1]
        
        if model in samples and not reset:
            completion_samples = samples[model].get("completion", [])
            zero_shot_samples = samples[model].get("zero_shot", [])
            if len(completion_samples) >= config.NUM_QUERIES and len(zero_shot_samples) >= config.NUM_QUERIES:
                print(f"\nSkipping {model} - already have {len(completion_samples)} completion + {len(zero_shot_samples)} zero-shot")
                continue
        
        print(f"\n{'='*50}")
        print(f"Generating samples for: {model}")
        print(f"{'='*50}")
        
        completion_outputs = run_completion_benchmark(model, model_name)
        zero_shot_outputs = run_zero_shot_benchmark(model, model_name)
        
        samples[model] = {
            "completion": completion_outputs,
            "zero_shot": zero_shot_outputs,
        }
        
        save_samples(samples)
        
        print(f"\n  Saved {len(completion_outputs)} completion + {len(zero_shot_outputs)} zero-shot samples")
        
        print("  Waiting 10s before next model...")
        time.sleep(10)
    
    print("\n" + "="*50)
    print("SAMPLE GENERATION COMPLETE")
    print("="*50)
    print(f"Samples saved to: {config.SAMPLES_FILE}")
    
    return samples


if __name__ == "__main__":
    run_generate()