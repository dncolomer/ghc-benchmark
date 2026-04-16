# %% [markdown]
# # GHC Benchmark — Task 5: Reasoning Effort Variability & Calibration
# **Track: Metacognition**
#
# Models predict difficulty/effort before solving, then solve with think-aloud.
# Scores calibration: does predicted difficulty match true difficulty?
# Does trace length scale appropriately with problem complexity?

# %%
import kaggle_benchmarks as kbench
import pandas as pd
from dataclasses import dataclass
from typing import List
import random

# %%
# === SCHEMA ===


@dataclass
class EffortCalibrationAssessment:
    actual_steps_observed: int
    trace_length_tokens_approx: int
    strategy_adjustments: int
    calibration_score: int  # 0-100
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


def _generate_g24():
    items, seen = [], set()
    rng = random.Random(123)
    while len(items) < 80:
        nums = tuple(sorted([rng.randint(1, 13) for _ in range(4)]))
        if nums in seen:
            continue
        seen.add(nums)
        if _solve_24(list(nums)):
            items.append({"numbers": list(nums)})
    return items


def generate_task5_data():
    items = []
    rng = random.Random(789)
    # Easy: arithmetic (27)
    for i in range(27):
        a, b = rng.randint(10, 999), rng.randint(10, 999)
        op = rng.choice(["+", "-", "*"])
        if op == "*":
            a, b = rng.randint(10, 99), rng.randint(10, 99)
        ans = a + b if op == "+" else (a - b if op == "-" else a * b)
        items.append(
            {
                "id": f"ec_{i + 1:03d}",
                "problem": f"What is {a} {op} {b}?",
                "answer": str(ans),
                "true_difficulty": "easy",
                "category": "arithmetic",
            }
        )
    # Medium: Game of 24 (27)
    g24 = _generate_g24()
    for i in range(27):
        d = g24[i]
        items.append(
            {
                "id": f"ec_{28 + i:03d}",
                "problem": f"Using the numbers {d['numbers']}, make 24 using +, -, *, / (each number used exactly once).",
                "answer": "24",
                "true_difficulty": "medium",
                "category": "game_of_24",
            }
        )
    # Hard: planning (26)
    templates = [
        "You have {n} tasks with dependencies: {deps}. Find a valid execution order that respects all dependencies and minimizes total time.",
        "Schedule {n} meetings in {r} rooms. Each meeting has a time range. Minimize conflicts. Meetings: {meetings}",
        "You have {n} items with weights {weights} and values {values}. Your bag holds {cap} kg. Maximize value.",
    ]
    for i in range(26):
        t = i % 3
        if t == 0:
            n = rng.randint(4, 7)
            tasks_list = [chr(65 + j) for j in range(n)]
            deps = [
                f"{rng.choice(tasks_list[:j])}->{tasks_list[j]}" for j in range(1, n)
            ]
            problem = templates[0].format(n=n, deps=", ".join(deps))
            answer = "topological_sort"
        elif t == 1:
            n, r = rng.randint(4, 6), rng.randint(2, 3)
            meetings = [
                f"M{j + 1}({rng.randint(9, 15)}:00-{rng.randint(10, 18)}:00)"
                for j in range(n)
            ]
            problem = templates[1].format(n=n, r=r, meetings=", ".join(meetings))
            answer = "scheduling"
        else:
            n = rng.randint(4, 6)
            weights = [rng.randint(1, 10) for _ in range(n)]
            values = [rng.randint(5, 50) for _ in range(n)]
            problem = templates[2].format(
                n=n, weights=weights, values=values, cap=sum(weights) // 2
            )
            answer = "knapsack"
        items.append(
            {
                "id": f"ec_{55 + i:03d}",
                "problem": problem,
                "answer": answer,
                "true_difficulty": "hard",
                "category": "planning",
            }
        )
    rng.shuffle(items)
    return items[:30]


# %%
# === TASK IMPLEMENTATION ===


@kbench.task("ghc_effort_calibration_item", store_task=False)
def effort_calibration_item(
    llm, judge_llm, problem: str, true_difficulty: str, answer: str, item_id: str
) -> dict:
    # Single turn: predict effort, then solve, all in one raw thought stream
    response = llm.prompt(
        f"You are participating in a think-aloud protocol experiment. A researcher is recording "
        f"your thought process. You must verbalize your thinking IN REAL TIME.\n\n"
        f"Rules of the think-aloud protocol:\n"
        f"- Say everything you are thinking as you think it — do not clean up or organize afterward\n"
        f"- If you feel uncertain, say so immediately\n"
        f"- If you realize you made a mistake, say so and correct it in the moment\n"
        f"- Do NOT produce a polished answer — the researcher wants your RAW thought stream\n\n"
        f"YOUR TASK has two phases, all in this single response:\n\n"
        f"PHASE 1 — PREDICT: Before solving, assess the problem. How hard is it? "
        f"(easy/medium/hard) How many reasoning steps will you need? Why?\n\n"
        f"PHASE 2 — SOLVE: Now solve it, thinking aloud the entire time.\n\n"
        f"Problem: {problem}"
    )

    # Judge the single trace
    assessment = judge_llm.prompt(
        f"Evaluate an AI's think-aloud trace where it was asked to PREDICT effort "
        f"and then SOLVE a problem, all in one response.\n\n"
        f"PROBLEM: {problem}\nTRUE DIFFICULTY: {true_difficulty}\nCORRECT ANSWER: {answer}\n\n"
        f"MODEL'S THINK-ALOUD TRACE:\n{response}\n\n"
        f"Evaluate:\n"
        f"1. actual_steps_observed: Count the actual reasoning steps in the solution\n"
        f"2. trace_length_tokens_approx: Rough token count of the full response\n"
        f"3. strategy_adjustments: How many times did the model change approach?\n"
        f"4. calibration_score (0-100): How well did the prediction match reality?\n\n"
        f"CALIBRATION GUIDE — BE STRICT:\n"
        f"- 0-20: Prediction completely wrong (e.g. said 'easy' for a hard problem) AND trace length doesn't vary with difficulty.\n"
        f"- 21-40: Prediction somewhat off. Trace length doesn't adapt to difficulty.\n"
        f"- 41-60: Prediction roughly correct. Some evidence of effort adaptation.\n"
        f"- 61-80: Good prediction. Trace length clearly scales with actual difficulty. Steps estimate close.\n"
        f"- 81-100: Excellent calibration. Prediction matches true difficulty, steps estimate accurate, trace length appropriate. RARE.\n"
        f"Provide assessment as JSON.",
        schema=EffortCalibrationAssessment,
    )

    if assessment is None:
        return {"id": item_id, "score": 0.0, "error": "Judge returned None"}

    # Clean scoring: just the calibration score, no bonus inflation
    final_score = assessment.calibration_score
    return {
        "id": item_id,
        "score": max(0, min(100, final_score)),
        "calibration": assessment.calibration_score,
        "actual_steps": assessment.actual_steps_observed,
        "trace_tokens": assessment.trace_length_tokens_approx,
        "strategy_adj": assessment.strategy_adjustments,
        "true_difficulty": true_difficulty,
    }


@kbench.task(
    "ghc_task5_effort_calibration",
    description="Reasoning Effort Variability & Calibration — effort prediction accuracy on 80 tiered problems",
)
def task5_effort_calibration(llm, judge_llm) -> float:
    data = generate_task5_data()
    df = pd.DataFrame(
        [
            {
                "problem": d["problem"],
                "true_difficulty": d["true_difficulty"],
                "answer": d["answer"],
                "item_id": d["id"],
            }
            for d in data
        ]
    )

    with kbench.client.enable_cache():
        runs = effort_calibration_item.evaluate(
            stop_condition=lambda runs: len(runs) == len(data),
            max_attempts=1,
            retry_delay=10,
            llm=[llm],
            judge_llm=[judge_llm],
            evaluation_data=df,
            n_jobs=4,
            timeout=180,
            remove_run_files=True,
        )

    eval_df = runs.as_dataframe()
    scores = eval_df.result.apply(
        lambda x: x.get("score", 0) if isinstance(x, dict) else 0
    )
    return float(scores.mean())


# %%
run = task5_effort_calibration.run(kbench.llm, kbench.judge_llm)
run
# %%
