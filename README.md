# GHC Benchmark - Genuine Human Cognition Score

A benchmarking pipeline that measures how closely AI Chain of Thought matches real human thinking patterns.

## What is GHC?

**Genuine Human Cognition (GHC)** is a benchmark that evaluates how authentically AI mimics human thinking. Modern AI is trained on polished, edited text—but real human cognition is messy, non-linear, and full of false starts. This benchmark measures that gap.

## Three Benchmarks

| Benchmark | What it measures |
|-----------|-----------------|
| **Authenticity (Similarity)** | Embedding-based similarity to real human transcripts |
| **Linearity Index** | Sequential flow measurement (0-100, higher = more robotic) |
| **Cluster Score** | Sustained thinking patterns (>5 consecutive linear = LLM-like) |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline
python benchmark.py run --stage all

# Or run individual stages
python benchmark.py run --stage generate   # Generate LLM samples
python benchmark.py run --stage evaluate    # Run benchmarks
python benchmark.py run --stage report      # Generate reports
```

## Configuration

Edit `src/config.py` to customize:

- `MODELS` - List of OpenRouter models to evaluate
- `NUM_QUERIES` - Queries per model (default: 20)
- `LINEARITY_JUDGE_MODEL` - Model used for linearity evaluation

```bash
# Use specific models
python benchmark.py run --stage all --models "openai/gpt-4o-mini" "google/gemini-2.0-flash-001"

# Reset and rerun
python benchmark.py run --stage all --reset
```

## Output

Results are saved to:
- `results/scores.json` - Full benchmark scores
- `results/samples.json` - Generated LLM outputs
- `results/charts/` - Visualization charts
- `results/reports/report.md` - Markdown report

## API Key

Set your OpenRouter API key via environment variable:

```bash
export OPENROUTER_API_KEY="your-api-key"
```

Or edit the default in `src/config.py` (not recommended for production).

## About

Built by [Uncertain Systems](https://uncertain.systems) as part of the open educational stack. Part of the GHC (Genuine Human Cognition) dataset initiative—collecting real human thinking to train AI that truly understands human thought.

## License

MIT
