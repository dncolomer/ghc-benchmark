# %% [markdown]
# # GHC Benchmark — Task 4: Metacognitive Self-Interrogation Loop
# **Track: Metacognition**
#
# 3-turn flow: Solve → Interrogate own reasoning → Revise.
# Probes whether models can genuinely find flaws in their own traces,
# distinguish real from spurious issues, and improve answers through self-examination.

# %%
import kaggle_benchmarks as kbench
import pandas as pd
from dataclasses import dataclass
from typing import List
import random
from math import comb

# %%
# === SCHEMA ===


@dataclass
class SelfInterrogationAssessment:
    interrogation_depth: int  # 0-100
    real_flaws_found: int
    false_flaws: int
    missed_flaws: int
    correctness_improved: bool
    reasoning: str


# %%
# === DATA GENERATOR ===


def generate_task4_data():
    items = []
    math_problems = [
        {
            "problem": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
            "answer": "$0.05",
            "category": "math_trap",
        },
        {
            "problem": "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
            "answer": "5 minutes",
            "category": "math_trap",
        },
        {
            "problem": "In a lake, there is a patch of lily pads. Every day, the patch doubles in size. If it takes 48 days for the patch to cover the entire lake, how long would it take for the patch to cover half of the lake?",
            "answer": "47 days",
            "category": "math_trap",
        },
        {
            "problem": "A farmer has 17 sheep. All but 9 die. How many sheep does the farmer have left?",
            "answer": "9",
            "category": "math_trap",
        },
        {
            "problem": "How many times can you subtract 5 from 25?",
            "answer": "Once (then it's 20, not 25)",
            "category": "math_trap",
        },
        {
            "problem": "If you have a bowl with six apples and you take away four, how many do you have?",
            "answer": "4 (you took them)",
            "category": "math_trap",
        },
        {
            "problem": "A doctor gives you 3 pills and tells you to take one every half hour. How long until all pills are taken?",
            "answer": "1 hour",
            "category": "math_trap",
        },
        {
            "problem": "Some months have 30 days, some have 31. How many months have 28 days?",
            "answer": "All 12 months",
            "category": "math_trap",
        },
    ]
    logic_problems = [
        {
            "problem": "There are three light switches outside a room. One controls a light bulb inside. You can flip switches as much as you want but can only enter the room once. How do you figure out which switch controls the bulb?",
            "answer": "Turn switch 1 on for a while, turn it off, turn switch 2 on, enter. If bulb is on = switch 2, if off and warm = switch 1, if off and cold = switch 3.",
            "category": "logic",
        },
        {
            "problem": "You have 12 identical-looking balls. One is a different weight (heavier or lighter). Using a balance scale exactly 3 times, find the odd ball and determine if it's heavier or lighter.",
            "answer": "Divide into groups of 4, weigh 4 vs 4. Based on result, narrow down with 2 more weighings.",
            "category": "logic",
        },
        {
            "problem": "Two fathers and two sons go fishing. They each catch one fish. They bring home exactly 3 fish. How?",
            "answer": "There are 3 people: grandfather, father, and son.",
            "category": "logic",
        },
        {
            "problem": "I speak without a mouth and hear without ears. I have no body, but I come alive with wind. What am I?",
            "answer": "An echo",
            "category": "logic",
        },
        {
            "problem": "What disappears as soon as you say its name?",
            "answer": "Silence",
            "category": "logic",
        },
    ]
    ambiguous_problems = [
        {
            "problem": "Is the following statement true or false: 'This statement is false.' Analyze carefully.",
            "answer": "This is a paradox (the Liar's Paradox) — it cannot be consistently assigned true or false.",
            "category": "ambiguous",
        },
        {
            "problem": "A plane crashes exactly on the US-Canada border. Where do you bury the survivors?",
            "answer": "You don't bury survivors.",
            "category": "ambiguous",
        },
        {
            "problem": "If there are 3 apples and you take away 2, how many apples do YOU have?",
            "answer": "2 (the ones you took)",
            "category": "ambiguous",
        },
        {
            "problem": "What is 0.1 + 0.2? Discuss any nuances.",
            "answer": "Mathematically 0.3, but in floating-point it's 0.30000000000000004.",
            "category": "ambiguous",
        },
        {
            "problem": "Is it possible for a man to marry his widow's sister? Explain.",
            "answer": "No — if he has a widow, he is dead.",
            "category": "ambiguous",
        },
    ]
    base = []
    for mp in math_problems:
        base.append({**mp, "difficulty": "medium"})
    for lp in logic_problems:
        base.append({**lp, "difficulty": "hard"})
    for ap in ambiguous_problems:
        base.append({**ap, "difficulty": "hard"})

    rng = random.Random(456)
    names = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Hank"]
    for i in range(42):
        a, b, c = rng.randint(2, 15), rng.randint(2, 15), rng.randint(2, 15)
        ptype = i % 6
        if ptype == 0:
            problem = f"If {a} workers can complete a job in {b} days, how many days would {c} workers take (assuming constant rate per worker)?"
            answer = f"{a * b / c:.2f} days"
            cat, diff = "math", "easy"
        elif ptype == 1:
            total_dist = a * 10 + c * 10
            total_time = (a * 10) / (b * 10) + (c * 10) / ((b + 3) * 10)
            problem = f"A train travels {a * 10} km at {b * 10} km/h, then {c * 10} km at {(b + 3) * 10} km/h. What is the average speed for the entire journey?"
            answer = (
                f"{total_dist / total_time:.2f} km/h (harmonic mean, not arithmetic)"
            )
            cat, diff = "math_trap", "medium"
        elif ptype == 2:
            n1, n2 = rng.sample(names, 2)
            problem = f"{n1} is twice as old as {n2}. In {a} years, {n1} will be {b} years older than {n2}. How old are they now?"
            answer = f"{n2} is {b}, {n1} is {2 * b}"
            cat, diff = "math", "easy"
        elif ptype == 3:
            num = a * 100 + b * 10 + c
            problem = f"What is the remainder when {num} is divided by {a}?"
            answer = f"{num % a}"
            cat, diff = "math", "easy"
        elif ptype == 4:
            target = a * c
            days = 1 if a >= target else target - a + 1
            problem = f"A snail climbs {a} feet during the day but slides back {a - 1} feet at night. How many days to reach the top of a {target} foot well?"
            answer = f"{days} days"
            cat, diff = "math_trap", "medium"
        else:
            k = min(b, a)
            prob = comb(a, k) / (2**a)
            problem = f"You flip a fair coin {a} times. What is the probability of getting exactly {k} heads? Express as a fraction or decimal."
            answer = f"{comb(a, k)}/{2**a} = {prob:.6f}"
            cat, diff = "math", "medium"
        base.append(
            {"problem": problem, "answer": answer, "category": cat, "difficulty": diff}
        )

    for i, item in enumerate(base):
        item["id"] = f"si_{i + 1:03d}"
    return base[:10]


# %%
# === TASK IMPLEMENTATION ===


@kbench.task("ghc_self_interrogation_item", store_task=False)
def self_interrogation_item(
    llm, judge_llm, problem: str, answer: str, item_id: str
) -> dict:
    # Single turn: solve + interrogate + revise all in one raw thought stream
    response = llm.prompt(
        f"You are participating in a think-aloud protocol experiment. A researcher is recording "
        f"your thought process. You must verbalize your thinking IN REAL TIME — write down "
        f"each thought the moment it occurs.\n\n"
        f"Rules of the think-aloud protocol:\n"
        f"- Say everything you are thinking as you think it — do not clean up or organize afterward\n"
        f"- If you feel uncertain, say so immediately\n"
        f"- If you realize you made a mistake, say so and correct it in the moment\n"
        f"- Do NOT produce a polished answer — the researcher wants your RAW thought stream\n\n"
        f"YOUR TASK has three phases, all in this single response:\n\n"
        f"PHASE 1 — SOLVE: Work through the problem below, thinking aloud.\n"
        f"PHASE 2 — INTERROGATE: Once you have an answer, go back and scrutinize your own "
        f"reasoning. Look for hidden assumptions, logical leaps, unverified claims, "
        f"alternative interpretations, and any errors. Be brutally honest.\n"
        f"PHASE 3 — REVISE: If you found real flaws, fix them and state your revised answer. "
        f"End with a confidence rating (0-100).\n\n"
        f"Problem: {problem}"
    )

    # Judge the single trace
    assessment = judge_llm.prompt(
        f"Evaluate an AI's think-aloud trace where it was asked to solve a problem, "
        f"interrogate its own reasoning, and revise — all in a single response.\n\n"
        f"PROBLEM: {problem}\nCORRECT ANSWER: {answer}\n\n"
        f"MODEL'S THINK-ALOUD TRACE:\n{response}\n\n"
        f"Evaluate the metacognitive quality:\n"
        f"1. interrogation_depth (0-100): How thoroughly did the model examine its OWN reasoning?\n"
        f"2. real_flaws_found: Count of genuine flaws correctly identified in its own trace\n"
        f"3. false_flaws: Count of non-issues incorrectly flagged\n"
        f"4. missed_flaws: Count of real flaws present but not caught\n"
        f"5. correctness_improved: Did self-interrogation lead to a better answer?\n\n"
        f"CALIBRATION — BE STRICT:\n"
        f"- 0-20: No real self-interrogation. Model just solves and says 'looks correct.'\n"
        f"- 21-40: Superficial review. Mentions checking but doesn't find anything substantive.\n"
        f"- 41-60: Moderate. Identifies 1-2 real issues or genuinely questions assumptions.\n"
        f"- 61-80: Good. Catches real flaws, considers alternatives, makes meaningful corrections.\n"
        f"- 81-100: Exceptional. Deep, genuine self-critique that improves the answer. RARE.\n"
        f"Provide assessment as JSON.",
        schema=SelfInterrogationAssessment,
    )

    if assessment is None:
        return {"id": item_id, "score": 0.0, "error": "Judge returned None"}

    # Clean scoring: interrogation depth is the score
    final_score = assessment.interrogation_depth
    return {
        "id": item_id,
        "score": max(0, min(100, final_score)),
        "depth": assessment.interrogation_depth,
        "real_flaws": assessment.real_flaws_found,
        "false_flaws": assessment.false_flaws,
        "missed_flaws": assessment.missed_flaws,
        "improved": assessment.correctness_improved,
    }


@kbench.task(
    "ghc_task4_self_interrogation",
    description="Metacognitive Self-Interrogation Loop — self-examination and revision on 60 mixed problems",
)
def task4_self_interrogation(llm, judge_llm) -> float:
    data = generate_task4_data()
    df = pd.DataFrame(
        [
            {"problem": d["problem"], "answer": d["answer"], "item_id": d["id"]}
            for d in data
        ]
    )

    with kbench.client.enable_cache():
        runs = self_interrogation_item.evaluate(
            stop_condition=lambda runs: len(runs) == len(data),
            max_attempts=1,
            retry_delay=10,
            llm=[llm],
            judge_llm=[judge_llm],
            evaluation_data=df,
            n_jobs=4,
            timeout=240,
            remove_run_files=True,
        )

    eval_df = runs.as_dataframe()
    scores = eval_df.result.apply(
        lambda x: x.get("score", 0) if isinstance(x, dict) else 0
    )
    return float(scores.mean())


# %%
run = task4_self_interrogation.run(kbench.llm, kbench.judge_llm)
run
# %%
