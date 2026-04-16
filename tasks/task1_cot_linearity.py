# %% [markdown]
# # GHC Benchmark — Task 1: CoT Linearity Analysis
# **Track: Metacognition**
#
# Measures how non-linear (metacognitive) a model's chain-of-thought reasoning is.
# Linear traces = step-by-step with no revision. Non-linear traces = backtracking,
# self-doubt, abandoned branches, revisions — indicators of genuine metacognition.

# %%
import kaggle_benchmarks as kbench
import pandas as pd
from dataclasses import dataclass
from typing import List

# %% [markdown]
# ## Data & Schema
# Paste the contents of ghc_data.py here (or import if uploaded as dataset).
# For brevity we inline only what Task 1 needs.

# %%
# === INLINE DATA (copy from ghc_data.py: generate_task1_data + LinearityAssessment) ===
import random
from math import comb


@dataclass
class LinearityAssessment:
    linearity_index: int
    back_references_count: int
    abandoned_branches_count: int
    self_doubt_phrases_count: int
    revision_count: int
    monitoring_phrases: List[str]
    reasoning: str


def _river_crossing_puzzles():
    items = []
    items.append(
        {
            "id": "rc_001",
            "puzzle": "A farmer needs to cross a river with a fox, a goose, and a bag of beans. The boat can carry only the farmer and one item at a time. If left alone, the fox will eat the goose, and the goose will eat the beans. How can the farmer get everything across safely?",
            "category": "river_crossing",
            "difficulty": "medium",
        }
    )
    for i, (a, b, c, r1, r2) in enumerate(
        [
            (
                "wolf",
                "sheep",
                "cabbage",
                "wolf will eat the sheep",
                "sheep will eat the cabbage",
            ),
            (
                "cat",
                "mouse",
                "cheese",
                "cat will eat the mouse",
                "mouse will eat the cheese",
            ),
            (
                "lion",
                "antelope",
                "grass",
                "lion will eat the antelope",
                "antelope will eat the grass",
            ),
            (
                "hawk",
                "chicken",
                "corn",
                "hawk will eat the chicken",
                "chicken will eat the corn",
            ),
        ]
    ):
        items.append(
            {
                "id": f"rc_{i + 2:03d}",
                "puzzle": f"A farmer needs to cross a river with a {a}, a {b}, and a bag of {c}. The boat can carry only the farmer and one item. If left alone, the {r1}, and the {r2}. How can the farmer get everything across safely?",
                "category": "river_crossing",
                "difficulty": "medium",
            }
        )
    return items


def _constraint_puzzles():
    items = []
    names_pool = [
        "Alice",
        "Bob",
        "Carol",
        "Dave",
        "Eve",
        "Frank",
        "Grace",
        "Hank",
        "Ivy",
        "Jack",
    ]
    colors_pool = [
        "red",
        "blue",
        "green",
        "yellow",
        "white",
        "orange",
        "purple",
        "pink",
    ]
    for i in range(20):
        rng = random.Random(42 + i)
        n = rng.choice([3, 4])
        names = rng.sample(names_pool, n)
        colors = rng.sample(colors_pool, n)
        assignment = list(zip(names, colors))
        rng.shuffle(assignment)
        houses = {
            j + 1: {"name": assignment[j][0], "color": assignment[j][1]}
            for j in range(n)
        }
        clues = []
        for j in range(n):
            pos = j + 1
            name = houses[pos]["name"]
            color = houses[pos]["color"]
            if j == 0:
                clues.append(f"{name} lives in the leftmost house.")
            elif j == n - 1:
                clues.append(f"{name} lives in the rightmost house.")
            else:
                clues.append(f"{name} lives in house number {pos}.")
            if rng.random() > 0.5:
                clues.append(f"The {color} house is house number {pos}.")
        rng.shuffle(clues)
        clue_text = "\n".join(f"{k + 1}. {c}" for k, c in enumerate(clues))
        items.append(
            {
                "id": f"cp_{i + 1:03d}",
                "puzzle": f"There are {n} houses in a row, numbered 1 to {n} from left to right. Each house has a unique color and is occupied by a different person.\nPeople: {', '.join(names)}\nColors: {', '.join(colors)}\n\nClues:\n{clue_text}\n\nDetermine who lives in which colored house.",
                "category": "constraint_satisfaction",
                "difficulty": rng.choice(["easy", "medium", "hard"]),
            }
        )
    return items


def _logic_deduction_puzzles():
    templates = [
        (
            "Three friends — {a}, {b}, and {c} — each ordered a different drink: coffee, tea, and juice. {a} didn't order coffee. {b} didn't order tea. {c} ordered juice. What did each person order?",
            "easy",
        ),
        (
            "Four students scored differently on a test. {a} scored higher than {b}. {c} scored lower than {b} but higher than {d}. {a} did not get the highest score. Rank all four students from highest to lowest.",
            "medium",
        ),
        (
            "In a row of five seats, {a} sits to the left of {b}. {c} sits at one end. {d} sits between {a} and {c}. {e} sits next to {b}. Determine the seating order from left to right.",
            "hard",
        ),
        (
            "{a} says '{b} is lying.' {b} says '{c} is lying.' {c} says 'Both {a} and {b} are lying.' If exactly one person is telling the truth, who is it? Explain your reasoning step by step.",
            "medium",
        ),
        (
            "A hat puzzle: {a}, {b}, and {c} each wear a hat that is either red or blue. Each can see the others' hats but not their own. {a} says 'I don't know my hat color.' {b} says 'I don't know either.' {c} then correctly guesses their own hat color. If there is at least one red hat, what color is {c}'s hat and why?",
            "hard",
        ),
    ]
    names_pool = [
        "Alice",
        "Bob",
        "Carol",
        "Dave",
        "Eve",
        "Frank",
        "Grace",
        "Hank",
        "Ivy",
        "Jack",
        "Kim",
        "Leo",
        "Mia",
        "Noah",
        "Olivia",
        "Pete",
        "Quinn",
        "Rosa",
        "Sam",
        "Tina",
    ]
    items = []
    for i in range(35):
        rng = random.Random(100 + i)
        tmpl, diff = templates[i % len(templates)]
        names = rng.sample(names_pool, 5)
        items.append(
            {
                "id": f"ld_{i + 1:03d}",
                "puzzle": tmpl.format(
                    a=names[0], b=names[1], c=names[2], d=names[3], e=names[4]
                ),
                "category": "logic_deduction",
                "difficulty": diff,
            }
        )
    return items


def generate_task1_data():
    items = (
        _river_crossing_puzzles() + _constraint_puzzles() + _logic_deduction_puzzles()
    )
    random.Random(42).shuffle(items)
    return items[:30]


# %%
# === TASK IMPLEMENTATION ===


@kbench.task("ghc_cot_linearity_item", store_task=False)
def cot_linearity_item(llm, judge_llm, puzzle: str, item_id: str) -> dict:
    """Evaluate metacognitive trace linearity on a single puzzle."""
    response = llm.prompt(
        f"You are participating in a think-aloud protocol experiment. A researcher is recording "
        f"your thought process as you solve a puzzle. You must verbalize your thinking IN REAL TIME "
        f"— write down each thought the moment it occurs, exactly as it comes to mind.\n\n"
        f"Rules of the think-aloud protocol:\n"
        f"- Say everything you are thinking as you think it — do not clean up or organize afterward\n"
        f"- If you feel uncertain, say so immediately (e.g. 'hmm, I'm not sure about this...')\n"
        f"- If you realize you made a mistake, say so and correct it in the moment\n"
        f"- If you want to abandon an approach, explain why before switching\n"
        f"- Do NOT produce a polished final answer — the researcher wants your RAW thought stream\n\n"
        f"Puzzle: {puzzle}"
    )

    assessment = judge_llm.prompt(
        f"Analyze the following reasoning trace for metacognitive indicators. "
        f"Assess how LINEAR vs NON-LINEAR the reasoning is.\n\n"
        f"A LINEAR trace goes step-by-step without backtracking, revision, or self-doubt.\n"
        f"A NON-LINEAR trace includes back-references, abandoned branches, self-corrections, "
        f"monitoring phrases like 'wait', 'actually', 'let me reconsider', 'hmm', etc.\n\n"
        f"CALIBRATION GUIDE for linearity_index:\n"
        f"- 90-100: Perfectly linear. No revisions, no backtracking, no self-doubt at all.\n"
        f"- 70-89: Mostly linear with minor hedging (e.g. 'let me check') but no real course corrections.\n"
        f"- 50-69: Mixed. Some genuine revisions or abandoned approaches, but structure is still forward-moving.\n"
        f"- 30-49: Substantially non-linear. Multiple real backtrack events, strategy changes, or error corrections.\n"
        f"- 0-29: Highly non-linear. Extensive self-monitoring, many abandoned paths, genuine struggle visible.\n\n"
        f"BE STRICT: Most LLM traces are highly linear (70-95) even when they use phrases like 'let me think'. "
        f"Only count GENUINE revisions where the model actually changes its answer or approach, not performative hedging.\n\n"
        f"Reasoning trace:\n{response}\n\n"
        f"Provide your assessment as JSON.",
        schema=LinearityAssessment,
    )

    if assessment is None:
        return {"id": item_id, "score": 0.0, "error": "Judge returned None"}

    # Score = 100 - linearity (so non-linear = higher metacognition score)
    # No bonus — rely purely on judge calibration
    score = 100 - assessment.linearity_index
    return {
        "id": item_id,
        "score": max(0, min(100, score)),
        "linearity_index": assessment.linearity_index,
        "back_refs": assessment.back_references_count,
        "abandoned": assessment.abandoned_branches_count,
        "self_doubt": assessment.self_doubt_phrases_count,
        "revisions": assessment.revision_count,
    }


@kbench.task(
    "ghc_task1_cot_linearity",
    description="CoT Linearity Analysis — measures metacognitive trace non-linearity across 60 logic puzzles",
)
def task1_cot_linearity(llm, judge_llm) -> float:
    data = generate_task1_data()
    df = pd.DataFrame([{"puzzle": d["puzzle"], "item_id": d["id"]} for d in data])

    with kbench.client.enable_cache():
        runs = cot_linearity_item.evaluate(
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
run = task1_cot_linearity.run(kbench.llm, kbench.judge_llm)
run
# %%
