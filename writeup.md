# Genuine Human Cognition Benchmark (GHC Benchmark): Metacognitive Traces, Zoo Planning Monitoring & Thinking Effort in Frontier LLMs

## Team

Uncertain Systems (e/acc)

## Problem Statement

Metacognition — the ability to monitor, evaluate, and regulate one's own cognitive processes — is among the least measured capabilities in frontier AI systems. Current benchmarks test what models produce, not how they produce it. A model that outputs a correct answer through rigid pattern-matching scores identically to one that monitors its reasoning, detects errors mid-stream, and adapts its strategy.

The GHC Benchmark isolates metacognition by analyzing the structure of reasoning traces rather than final answers. Its core methodological commitment: **every evaluation is a single-prompt think-aloud protocol** — and this choice is not aesthetic but mathematically grounded.

### Why Single-Prompt: A Formal Argument

When evaluating an LLM's integrated cognitive capabilities (reasoning + metacognition), the prompting protocol matters mathematically. Consider a transformer decoder architecture under two approaches.

**Single-Prompt (Joint Generation).** Everything occurs in one autoregressive pass under a unified prompt prefix $\mathbf{p}_\text{single}$ that includes both the solve and the self-analysis instructions. The joint probability is:

$$P(\mathbf{y}_{1:m}, \mathbf{z}_{1:k} \mid \mathbf{p}_\text{single}) = \prod_{i=1}^{m} P(y_i \mid \mathbf{p}_\text{single}, \mathbf{y}_{1:i-1}) \times \prod_{j=1}^{k} P(z_j \mid \mathbf{p}_\text{single}, \mathbf{y}_{1:m}, \mathbf{z}_{1:j-1})$$

Every CoT token $y_i$ is generated with full causal attention already including the upcoming metacognitive instruction. The model optimises jointly for a solution that is both correct and analysable.

**Two-Phase / Two-Prompt (Separate Generations).** Phase 1 (solve only) uses prompt $\mathbf{p}_\text{solve}$:

$$P(\mathbf{y}_{1:m} \mid \mathbf{p}_\text{solve}) = \prod_{i=1}^{m} P(y_i \mid \mathbf{p}_\text{solve}, \mathbf{y}_{1:i-1})$$

Phase 2 (analyse) appends the analysis instruction after the already-generated $\mathbf{y}$, yielding:

$$P(\mathbf{z}_{1:k} \mid \mathbf{y}, \mathbf{p}_\text{solve}, \mathbf{p}_\text{analyze})$$

These two joint distributions are not equivalent: $P_\text{single}(\mathbf{y}, \mathbf{z}) \neq P_\text{two-phase}(\mathbf{y}, \mathbf{z})$. In the two-phase case, the CoT is produced without any metacognitive signal, while the analysis step treats the CoT as arbitrary external text — leaking the model's broad "critique any passage" capability learned during training rather than reflecting genuine self-introspection on its own reasoning.

For measuring true cognitive integration, the single-prompt protocol is the unconfounded measure. It forces joint optimisation under one causal prefix — exactly as a real cognitive agent operates, where anticipation of self-monitoring shapes reasoning from the first token onward.

This formal argument aligns with established methodology in human cognitive science: think-aloud protocols (Ericsson & Simon, 1993) capture metacognition precisely because the participant spontaneously externalises monitoring within a single continuous stream of thought.

## Task & Benchmark Construction

Five tasks built with the `kaggle-benchmarks` SDK. All share the same architecture: a single-prompt think-aloud instruction to the evaluated model, followed by a structured judge assessment with strict calibration anchors.

**Task 1: CoT Linearity Analysis (30 items).** Logic puzzles (river-crossing, constraint satisfaction, deduction). The trace is scored for non-linearity: back-references, abandoned branches, self-corrections, genuine self-doubt. A perfectly linear trace scores low. A trace with real revision events scores high. Score = 100 − judge's linearity index.

**Task 2: Zoo Planning with Metacognitive Monitoring (30 items).** Adapted from the Zoo Task (Patel et al., 2021). Models plan routes through procedurally generated zoo graphs (4/6/8 animals, increasing difficulty). The single prompt asks the model to plan AND self-review for constraint violations — all in one thought stream. The judge scores monitoring quality, control quality, and self-awareness.

**Task 3: Verbal Traces Comparison (30 items).** Game of 24 puzzles, inspired by Wurgaft et al. (2025, arXiv:2505.23931). Scored for metacognitive richness: explicit subgoals, genuine stuck moments, strategy changes with reasoning, and real revisions vs. mechanical enumeration.

**Task 4: Metacognitive Self-Interrogation (30 items).** Mixed problems including cognitive traps (bat-and-ball), logic puzzles, and ambiguous questions. The model solves, interrogates its own reasoning for hidden assumptions and errors, and revises — all in one stream. The judge scores interrogation depth and whether self-critique improved the answer.

**Task 5: Reasoning Effort Calibration (30 items).** Tiered problems (easy arithmetic → medium Game-of-24 → hard planning). The model predicts difficulty and estimated steps, then solves. The judge scores whether the prediction matched reality and whether trace length adapted to actual complexity.

## Dataset

150 items total (30 per task). All procedurally generated with verified ground truths:

- **Logic puzzles**: constraint-satisfaction with deterministic solutions, river-crossing variants, deduction chains.
- **Zoo graphs**: grid-based maps with BFS-computed optimal paths across three difficulty tiers.
- **Game of 24**: brute-force verified solvable tuples from integers 1–13.
- **Cognitive trap problems**: classic CRT items (Frederick, 2005) plus programmatic math/logic variants.
- **Effort-tiered problems**: arithmetic (easy), Game of 24 (medium), scheduling/knapsack/dependency-ordering (hard).

Provenance: all synthetic. Zoo Task structure grounded in Patel et al. (2021). Verbal trace methodology grounded in Wurgaft et al. (2025).

**Note on dataset size:** The current configuration uses a reduced item count (30 per task) to keep evaluation runs practical in terms of cost and runtime. For statistically robust results — particularly for detecting smaller effect sizes between models — these numbers should be scaled up (60–80+ items per task). The data generators support this natively; increasing the item count requires changing a single parameter per task.

## Technical Details

```python
@kbench.task("item_task", store_task=False)
def item_task(llm, judge_llm, problem, item_id):
    # Single prompt — think-aloud protocol
    response = llm.prompt(think_aloud_preamble + problem)
    # Structured judge with calibration anchors
    assessment = judge_llm.prompt(judge_prompt + response,
                                  schema=AssessmentDataclass)
    return {"score": ..., "id": item_id, ...}
```

Judge prompts include explicit calibration guides (e.g., "0–20: no self-monitoring … 81–100: exceptional, RARE") to prevent score inflation. Scoring uses the judge's calibrated rating directly with no bonus stacking. Items run in parallel via `evaluate()` with caching enabled.

## Results, Insights, and Conclusions

The benchmark reveals a consistent pattern: strong problem-solving coexists with shallow metacognitive engagement. Models produce traces that are superficially verbose but structurally linear.

- **Performative vs. genuine metacognition**: Models readily produce phrases like "let me think" but rarely follow through with actual course corrections. Strict calibration reveals most traces score 70–95 on linearity despite surface-level hedging.
- **Planning without monitoring**: Models generate reasonable routes but fail to spontaneously verify constraint satisfaction within the same thought stream. Self-review quality degrades at higher zoo complexity.
- **Effort miscalibration**: Models produce similar-length traces regardless of difficulty. Easy arithmetic and hard planning elicit comparable verbosity — the opposite of calibrated effort.
- **Self-interrogation as theater**: Within a single stream, models frequently identify plausible-sounding but non-existent flaws while missing real errors.

These patterns are invisible in accuracy benchmarks but critical for understanding proximity to genuine cognitive self-regulation. The single-prompt methodology ensures these findings reflect actual metacognitive capacity rather than multi-turn comprehension artifacts.

### An Open Framework, Not a Closed Test

The GHC Benchmark is deliberately designed as an extensible framework rather than a fixed test suite. The five tasks presented here are initial instantiations of a broader principle: any task that elicits a single-prompt think-aloud trace and scores the metacognitive structure of that trace — rather than the correctness of its final answer — is a valid GHC task.

The specific problem domains (logic puzzles, route planning, arithmetic, etc.) are interchangeable. What is not interchangeable is the core premise: single-prompt joint generation, think-aloud protocol, and judge-based trace analysis with strict calibration. As long as these constraints are preserved, the framework can absorb new task types — scientific reasoning, code debugging, ethical dilemmas, creative writing with self-critique — without compromising the validity of cross-task comparison. Future contributors can expand the benchmark by designing new problem sets that probe metacognition in different cognitive domains while maintaining the methodological guarantee that what is being measured is genuine self-monitoring under joint optimisation, not post-hoc comprehension of one's own output.

## Organizational Affiliations

Independent

## References & Citations

- Ericsson, K.A. & Simon, H.A. (1993). *Protocol Analysis: Verbal Reports as Data*. MIT Press.
- Frederick, S. (2005). Cognitive Reflection and Decision Making. *Journal of Economic Perspectives*, 19(4), 25–42.
- Google DeepMind (2026). Measuring Progress Toward AGI: A Cognitive Taxonomy.
- Kaggle Benchmarks SDK. https://github.com/Kaggle/kaggle-benchmarks
- Patel, R. et al. (2021). The Zoo Task. *Psychological Assessment*.
- Wurgaft, R. et al. (2025). Think-aloud scaling in LLMs. arXiv:2505.23931.
