"""Stage 3: Generate reports and visualizations.

This stage generates:
1. PNG charts: authenticity, linearity, cluster, combined
2. Markdown report: benchmark summary
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
from . import config
from .utils.helpers import load_results


def generate_charts(results: dict):
    """Generate visualization charts."""
    os.makedirs(config.CHARTS_DIR, exist_ok=True)
    
    if not results:
        print("No results to plot")
        return
    
    models = [m.split("/")[-1] for m in results.keys()]
    
    balanced_scores = []
    linearity_scores = []
    cluster_scores = []
    
    for m in results.keys():
        if "balanced_score" in results[m]:
            balanced_scores.append(results[m]["balanced_score"])
        elif "completion" in results[m]:
            balanced_scores.append(results[m]["completion"]["normalized_score"])
        else:
            balanced_scores.append(0)
        
        if "linearly_index" in results[m]:
            linearity_scores.append(results[m]["linearly_index"]["raw_score"])
        else:
            linearity_scores.append(0)
        
        if "cluster_index" in results[m]:
            cluster_scores.append(results[m]["cluster_index"]["raw_score"])
        else:
            cluster_scores.append(0)
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    colors1 = plt.cm.viridis(np.linspace(0.2, 0.8, len(models)))
    ax1 = axes[0, 0]
    bars1 = ax1.bar(x, balanced_scores, width, color=colors1, edgecolor='black', linewidth=1.2)
    ax1.set_ylabel('Similarity Score (%)', fontsize=11)
    ax1.set_title('Human Thinking Authenticity\n(Balanced: Completion + Zero-shot)', fontsize=12)
    ax1.set_ylim(0, 100)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=35, ha='right', fontsize=9)
    for bar, score in zip(bars1, balanced_scores):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{score:.1f}', ha='center', va='bottom', fontsize=8)
    
    colors2 = plt.cm.plasma(np.linspace(0.2, 0.8, len(models)))
    ax2 = axes[0, 1]
    bars2 = ax2.bar(x, linearity_scores, width, color=colors2, edgecolor='black', linewidth=1.2)
    ax2.set_ylabel('Linearity Score', fontsize=11)
    ax2.set_title('Linearity Index\n(0-100, higher = more linear)', fontsize=12)
    ax2.set_ylim(0, 100)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=35, ha='right', fontsize=9)
    for bar, score in zip(bars2, linearity_scores):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{score:.1f}', ha='center', va='bottom', fontsize=8)
    
    colors3 = plt.cm.inferno(np.linspace(0.2, 0.8, len(models)))
    ax3 = axes[1, 0]
    bars3 = ax3.bar(x, cluster_scores, width, color=colors3, edgecolor='black', linewidth=1.2)
    ax3.set_ylabel('Cluster Score', fontsize=11)
    ax3.set_title('Cluster Score\n(0-100, >5 consecutive linear = LLM-like)', fontsize=12)
    ax3.set_ylim(0, 100)
    ax3.set_xticks(x)
    ax3.set_xticklabels(models, rotation=35, ha='right', fontsize=9)
    for bar, score in zip(bars3, cluster_scores):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{score:.1f}', ha='center', va='bottom', fontsize=8)
    
    combined_scores = []
    for i, m in enumerate(results.keys()):
        if "combined_score" in results[m]:
            combined_scores.append(results[m]["combined_score"])
        else:
            combined_scores.append((balanced_scores[i] + linearity_scores[i]) / 2)
    
    colors4 = plt.cm.coolwarm(np.linspace(0.2, 0.8, len(models)))
    ax4 = axes[1, 1]
    bars4 = ax4.bar(x, combined_scores, width, color=colors4, edgecolor='black', linewidth=1.2)
    ax4.set_ylabel('Combined Score', fontsize=11)
    ax4.set_title('Combined Score\n(Authenticity + Linearity) / 2', fontsize=12)
    ax4.set_ylim(0, 100)
    ax4.set_xticks(x)
    ax4.set_xticklabels(models, rotation=35, ha='right', fontsize=9)
    for bar, score in zip(bars4, combined_scores):
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{score:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f"{config.CHARTS_DIR}/benchmark_overview.png", dpi=150, bbox_inches='tight')
    print(f"Chart saved to {config.CHARTS_DIR}/benchmark_overview.png")
    
    sorted_by_combined = sorted(results.items(), key=lambda x: -x[1].get("combined_score", x[1].get("balanced_score", 0)))
    
    fig2, ax = plt.subplots(figsize=(12, 6))
    model_names = [m.split("/")[-1] for m, _ in sorted_by_combined]
    scores = [d.get("combined_score", d.get("balanced_score", 0)) for _, d in sorted_by_combined]
    
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(model_names)))
    bars = ax.barh(model_names[::-1], scores[::-1], color=colors[::-1], edgecolor='black')
    ax.set_xlim(0, 100)
    ax.set_xlabel('Combined Score', fontsize=12)
    ax.set_title('GHC Benchmark Results - Ranked by Combined Score', fontsize=14)
    
    for bar, score in zip(bars, scores[::-1]):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2.,
                f'{score:.1f}%', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{config.CHARTS_DIR}/ranked_results.png", dpi=150, bbox_inches='tight')
    print(f"Chart saved to {config.CHARTS_DIR}/ranked_results.png")


def generate_markdown_report(results: dict):
    """Generate markdown report."""
    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    
    report = """# GHC Benchmark Report

## Genuine Human Cognition Score

This benchmark measures how closely AI Chain of Thought matches real human thinking patterns.

### Methodology

**Three Benchmarks:**

1. **Authenticity (Similarity)** - Embedding-based similarity to human transcript
2. **Linearity Index** - Sequential flow measurement (0-100, higher = more linear)
3. **Cluster Score** - Sustained thinking patterns (>5 consecutive linear = LLM-like)

### Results

| Model | Completion | Zero-shot | Balanced | Linearity | Cluster | Combined |
|-------|------------|-----------|----------|-----------|---------|----------|
"""
    
    sorted_results = sorted(results.items(), key=lambda x: -x[1].get("combined_score", x[1].get("balanced_score", 0)))
    
    for model, data in sorted_results:
        model_name = model.split("/")[-1]
        completion = data.get("completion", {}).get("normalized_score", 0)
        zero_shot = data.get("zero_shot", {}).get("normalized_score", 0)
        balanced = data.get("balanced_score", 0)
        linearity = data.get("linearly_index", {}).get("raw_score", 0)
        cluster = data.get("cluster_index", {}).get("raw_score", 0)
        combined = data.get("combined_score", 0)
        
        report += f"| {model_name} | {completion:.1f}% | {zero_shot:.1f}% | {balanced:.1f}% | {linearity:.1f} | {cluster:.1f} | {combined:.1f}% |\n"
    
    report += """
### Key Findings

- **Linearity**: LLMs tend to produce more linear, structured thinking than human transcripts
- **Cluster**: Human thinking is fragmented with many topic switches; LLMs have sustained linear progressions
- **Authenticity**: Models like Gemini and Claude show highest similarity to human thinking patterns

### Interpretation

- **High Combined Score** = Better at mimicking genuine human cognition
- **High Linearity** = More robotic, structured (opposite of human)
- **High Cluster** = Sustained thinking chains (LLM-like)
- **High Authenticity** = Closer match to human transcript patterns

Generated by GHC Benchmark Pipeline
"""
    
    filepath = f"{config.REPORTS_DIR}/report.md"
    with open(filepath, "w") as f:
        f.write(report)
    
    print(f"Report saved to {filepath}")


def run_report():
    """Generate all reports and visualizations."""
    results = load_results()
    
    if not results:
        print("No results found. Run stages 1 and 2 first.")
        return
    
    print("Generating reports...")
    
    generate_charts(results)
    generate_markdown_report(results)
    
    print("\n" + "="*50)
    print("REPORT GENERATION COMPLETE")
    print("="*50)


if __name__ == "__main__":
    run_report()
