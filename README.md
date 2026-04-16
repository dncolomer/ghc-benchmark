# GHC Benchmark — Genuine Human Cognition Benchmark

**Metacognition Track** | [Google DeepMind x Kaggle "Measuring Progress Toward AGI"](https://www.kaggle.com/competitions/kaggle-measuring-agi)

The first metacognition benchmark built on a formally grounded single-prompt think-aloud protocol.

## Core Premise

Multi-turn evaluation of metacognition in LLMs is mathematically confounded. The two-phase approach (solve, then review) produces a different probability distribution than single-prompt joint generation — the analysis step leaks the model's generic "critique any text" capability rather than measuring genuine self-monitoring.

GHC uses single-prompt think-aloud protocols (Ericsson & Simon, 1993) where reasoning and self-monitoring are jointly optimized under one causal prefix. The trace structure — not the final answer — is what gets scored.

## Tasks

| # | Task | Items | What It Measures |
|---|------|-------|------------------|
| 1 | CoT Linearity Analysis | 30 | Trace non-linearity: backtracking, revisions, genuine self-doubt |
| 2 | Zoo Planning + Monitoring | 30 | Route planning + spontaneous constraint verification |
| 3 | Verbal Traces (Game of 24) | 30 | Metacognitive richness: subgoals, stuck moments, strategy shifts |
| 4 | Self-Interrogation Loop | 30 | Self-critique depth and whether it improves the answer |
| 5 | Effort Calibration | 30 | Predicted vs actual difficulty, trace length adaptation |

**150 items total** (default), all procedurally generated with verified ground truths. Each task runs in ~10-12 minutes.

### Configuration

Each task file has two constants at the top for easy tuning:

```python
N_ITEMS = 30  # Number of items to evaluate. Max available: 60-80 depending on task.
N_JOBS = 4    # Parallel workers for evaluation.
```

For quick iteration, keep `N_ITEMS = 30`. For statistically robust, publishable results, increase to 60-80. The data generators support the full range natively — no other code changes needed.

## Structure

```
tasks/
  task1_cot_linearity.py       # Paste into Kaggle task notebook
  task2_zoo_planning.py
  task3_verbal_traces.py
  task4_self_interrogation.py
  task5_effort_calibration.py
  ghc_data.py                  # Shared data generators (reference)
docs/
  index.html                   # Project website
writeup.md                     # Competition writeup
ghc_benchmark.py               # Monolith version (backup)
```

## Usage

Each task file is self-contained. To run on Kaggle:

1. Go to https://www.kaggle.com/benchmarks/tasks/new
2. Paste the contents of a task file
3. Run — it creates the task + leaderboard results
4. Group tasks into a benchmark

## Built With

- [kaggle-benchmarks SDK](https://github.com/Kaggle/kaggle-benchmarks)

## References

- Ericsson & Simon (1993). *Protocol Analysis: Verbal Reports as Data*
- Patel et al. (2021). The Zoo Task. *Psychological Assessment*
- Wurgaft et al. (2025). arXiv:2505.23931
- Google DeepMind (2026). Measuring Progress Toward AGI: A Cognitive Taxonomy

## Team

**Uncertain Systems** (e/acc)

## License

MIT
