# %% [markdown]
# # GHC Benchmark — Task 3: Verbal Traces Comparison (Game of 24)
# **Track: Metacognition**
#
# Based on arxiv 2505.23931 (Wurgaft et al., 2025). Models solve Game of 24
# puzzles while verbalizing every thought. Judge scores metacognitive richness:
# subgoals, revisions, stuck moments, strategy changes.

# %%
import kaggle_benchmarks as kbench
import pandas as pd
from dataclasses import dataclass
from typing import List
import random

# %% [markdown]
# ## Configuration
# Adjust the number of items per task. Higher = more statistically robust but slower/costlier.
# Recommended: 30 (quick iteration), 60+ (publishable results).

# %%
N_ITEMS = 30  # Number of items to evaluate. Max available: 80.
N_JOBS = 4  # Parallel workers for evaluation.

# %%
# === SCHEMA ===


@dataclass
class VerbalTraceAssessment:
    subgoals_identified: int
    revisions_count: int
    stuck_moments: int
    effort_adjustments: int
    metacog_richness: int  # 0-100
    correct_answer: bool
    reasoning: str


# %%
# === DATA GENERATOR ===


def _solve_24(nums):
    if len(nums) == 1:
        return abs(nums[0] - 24) < 1e-9
    for i in range(len(nums)):
        for j in range(len(nums)):
            if i == j:
                continue
            rest = [nums[k] for k in range(len(nums)) if k != i and k != j]
            a, b = nums[i], nums[j]
            for op in ["+", "-", "*", "/"]:
                if op == "+":
                    res = a + b
                elif op == "-":
                    res = a - b
                elif op == "*":
                    res = a * b
                else:
                    if abs(b) < 1e-9:
                        continue
                    res = a / b
                if _solve_24(rest + [res]):
                    return True
    return False


def generate_task3_data():
    items, seen = [], set()
    rng = random.Random(123)
    attempts = 0
    while len(items) < 80 and attempts < 10000:
        nums = tuple(sorted([rng.randint(1, 13) for _ in range(4)]))
        attempts += 1
        if nums in seen:
            continue
        seen.add(nums)
        if _solve_24(list(nums)):
            items.append(
                {"id": f"g24_{len(items) + 1:03d}", "numbers": list(nums), "target": 24}
            )
    return items[:N_ITEMS]


# %%
# === TASK IMPLEMENTATION ===


@kbench.task("ghc_verbal_trace_item", store_task=False)
def verbal_trace_item(llm, judge_llm, numbers: list, item_id: str) -> dict:
    response = llm.prompt(
        f"You are participating in a think-aloud protocol experiment. A researcher is recording "
        f"your thought process as you solve a math puzzle. You must verbalize your thinking "
        f"IN REAL TIME — write down each thought the moment it occurs.\n\n"
        f"Rules of the think-aloud protocol:\n"
        f"- Say everything you are thinking as you think it — do not clean up or organize afterward\n"
        f"- If you feel stuck, say so immediately\n"
        f"- If a combination doesn't work, say why and what you'll try next\n"
        f"- If you realize a mistake, correct it in the moment\n"
        f"- Do NOT produce a polished solution — the researcher wants your RAW thought stream\n\n"
        f"PUZZLE: Use the numbers {numbers} with +, -, *, / (each number exactly once) to make 24."
    )

    assessment = judge_llm.prompt(
        f"Analyze this think-aloud reasoning trace for a Game of 24 puzzle.\n"
        f"Numbers: {numbers}, Target: 24\n\nTrace:\n{response}\n\n"
        f"Evaluate the metacognitive content:\n"
        f"1. subgoals_identified: How many EXPLICIT subgoals stated? (e.g. 'First I'll try to make 12, then multiply by 2')\n"
        f"2. revisions_count: How many times did it GENUINELY revise its approach (not just try another combination)?\n"
        f"3. stuck_moments: How many times did it express GENUINE uncertainty or being stuck?\n"
        f"4. effort_adjustments: How many times did it change strategy with explicit reasoning about WHY?\n"
        f"5. metacog_richness: Overall 0-100 score — use this STRICT calibration:\n"
        f"   - 0-15: No metacognition. Just tries combinations mechanically.\n"
        f"   - 16-35: Minimal. Occasional 'let me try' but no real self-monitoring.\n"
        f"   - 36-55: Moderate. Some genuine subgoals and 1-2 real revisions with reasoning.\n"
        f"   - 56-75: Good. Multiple explicit subgoals, genuine stuck moments, strategy changes with explanations.\n"
        f"   - 76-100: Exceptional. Rich inner monologue with deep self-monitoring, explicit error detection, adaptive strategy. RARE — most LLM traces score 15-45.\n"
        f"6. correct_answer: Did the model arrive at a valid solution making 24?\n\n"
        f"BE STRICT: LLMs often produce fluent traces that LOOK metacognitive but are actually just systematic enumeration. "
        f"Only count genuine self-monitoring, not performative think-aloud.\n"
        f"Provide assessment as JSON.",
        schema=VerbalTraceAssessment,
    )

    if assessment is None:
        return {"id": item_id, "score": 0.0, "error": "Judge returned None"}

    # 80% metacog richness + 20% correctness — no bonus inflation
    correctness_bonus = 20 if assessment.correct_answer else 0
    final_score = min(100, assessment.metacog_richness * 0.8 + correctness_bonus)
    return {
        "id": item_id,
        "score": final_score,
        "metacog_richness": assessment.metacog_richness,
        "correct": assessment.correct_answer,
        "subgoals": assessment.subgoals_identified,
        "revisions": assessment.revisions_count,
    }


@kbench.task(
    "ghc_task3_verbal_traces",
    description="Verbal Traces Comparison — think-aloud metacognitive analysis on 80 Game of 24 puzzles",
)
def task3_verbal_traces(llm, judge_llm) -> float:
    data = generate_task3_data()
    df = pd.DataFrame([{"numbers": d["numbers"], "item_id": d["id"]} for d in data])

    with kbench.client.enable_cache():
        runs = verbal_trace_item.evaluate(
            stop_condition=lambda runs: len(runs) == len(data),
            max_attempts=1,
            retry_delay=10,
            llm=[llm],
            judge_llm=[judge_llm],
            evaluation_data=df,
            n_jobs=N_JOBS,
            timeout=180,
            remove_run_files=True,
        )

    eval_df = runs.as_dataframe()
    scores = eval_df.result.apply(
        lambda x: x.get("score", 0) if isinstance(x, dict) else 0
    )
    return float(scores.mean())


# %%
run = task3_verbal_traces.run(kbench.llm, kbench.judge_llm)
run
# %%
