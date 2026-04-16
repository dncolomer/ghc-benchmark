# %% [markdown]
# # GHC Benchmark — Task 2: Zoo Planning with Metacognitive Monitoring
# **Track: Metacognition**
#
# Adapted from The Zoo Task (Patel et al., 2021). Models plan efficient routes
# through a zoo graph to feed animals, then self-review for violations.
# Probes metacognitive monitoring (error detection) and control (revision).

# %%
import kaggle_benchmarks as kbench
import pandas as pd
from dataclasses import dataclass
from typing import List
import random
from collections import deque

# %%
# === SCHEMA ===


@dataclass
class ZooMetacogAssessment:
    monitoring_quality: int  # 0-100
    control_quality: int  # 0-100
    self_awareness: int  # 0-100
    false_positives: int
    missed_issues: int
    reasoning: str


# %%
# === DATA GENERATOR ===


def generate_zoo_graph(num_animals, seed_val):
    rng = random.Random(seed_val)
    if num_animals <= 4:
        rows, cols = 3, 3
    elif num_animals <= 6:
        rows, cols = 3, 4
    else:
        rows, cols = 4, 4
    nodes = [(r, c) for r in range(rows) for c in range(cols)]
    start, finish = (0, 0), (rows - 1, cols - 1)
    available = [n for n in nodes if n != start and n != finish]
    rng.shuffle(available)
    animal_names_pool = [
        "Lions",
        "Elephants",
        "Giraffes",
        "Penguins",
        "Monkeys",
        "Zebras",
        "Bears",
        "Flamingos",
        "Wolves",
        "Pandas",
        "Tigers",
        "Seals",
        "Otters",
        "Parrots",
        "Turtles",
    ]
    animals_selected = rng.sample(animal_names_pool, num_animals)
    animal_positions = {
        animal: available[i] for i, animal in enumerate(animals_selected)
    }
    adjacency = {n: [] for n in nodes}
    for r in range(rows):
        for c in range(cols):
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    adjacency[(r, c)].append((nr, nc))
    target = frozenset(animals_selected)
    pos_to_animal = {v: k for k, v in animal_positions.items()}
    initial_visited = (
        frozenset([pos_to_animal[start]]) if start in pos_to_animal else frozenset()
    )
    queue = deque([(start, initial_visited, [start])])
    seen = {(start, initial_visited)}
    optimal_path = None
    while queue:
        curr, visited_animals, path = queue.popleft()
        if curr == finish and visited_animals == target:
            optimal_path = path
            break
        for neighbor in adjacency[curr]:
            new_visited = (
                visited_animals | frozenset([pos_to_animal[neighbor]])
                if neighbor in pos_to_animal
                else visited_animals
            )
            state = (neighbor, new_visited)
            if state not in seen:
                seen.add(state)
                queue.append((neighbor, new_visited, path + [neighbor]))
    node_labels = {start: "Start", finish: "Finish"}
    for animal, pos in animal_positions.items():
        node_labels[pos] = f"{animal} Cage"
    jc = 0
    for n in nodes:
        if n not in node_labels:
            jc += 1
            node_labels[n] = f"Junction-{jc}"
    lines = [
        f"Zoo Map ({rows}x{cols} grid):",
        f"Start: {node_labels[start]} at position {start}",
        f"Finish: {node_labels[finish]} at position {finish}",
        f"\nAnimals to feed: {', '.join(animals_selected)}",
        f"\nPaths (bidirectional connections):",
    ]
    seen_edges = set()
    for n in nodes:
        for nb in adjacency[n]:
            edge = tuple(sorted([n, nb]))
            if edge not in seen_edges:
                seen_edges.add(edge)
                lines.append(f"  {node_labels[n]} <-> {node_labels[nb]}")
    lines.append(f"\nGrid layout (row, col):")
    for r in range(rows):
        row_str = [f"[{node_labels[(r, c)][:8].ljust(8)}]" for c in range(cols)]
        lines.append("  " + " -- ".join(row_str))
        if r < rows - 1:
            lines.append("  " + "    ".join(["     |    "] * cols))
    zoo_text = "\n".join(lines)
    opt_text = (
        " -> ".join(node_labels[n] for n in optimal_path) if optimal_path else "No path"
    )
    opt_length = len(optimal_path) if optimal_path else -1
    return {
        "zoo_text": zoo_text,
        "animals": animals_selected,
        "num_animals": num_animals,
        "optimal_path": opt_text,
        "optimal_length": opt_length,
        "adjacency": {str(k): [str(v) for v in vs] for k, vs in adjacency.items()},
        "animal_positions": {k: str(v) for k, v in animal_positions.items()},
        "node_labels": {str(k): v for k, v in node_labels.items()},
    }


def generate_task2_data():
    items = []
    idx = 0
    for num_animals, count in [(4, 24), (6, 24), (8, 22)]:
        for i in range(count):
            zoo = generate_zoo_graph(num_animals, seed_val=idx * 1000 + i)
            if zoo["optimal_length"] > 0:
                items.append(
                    {
                        "id": f"zoo_{idx + 1:03d}",
                        **zoo,
                        "difficulty": {4: "easy", 6: "medium", 8: "hard"}[num_animals],
                    }
                )
                idx += 1
    return items[:30]


# %%
# === TASK IMPLEMENTATION ===


@kbench.task("ghc_zoo_planning_item", store_task=False)
def zoo_planning_item(
    llm,
    judge_llm,
    zoo_text: str,
    animals: list,
    optimal_length: int,
    item_id: str,
    num_animals: int,
) -> dict:
    # Single turn: think-aloud planning + self-review in one shot
    response = llm.prompt(
        f"You are participating in a think-aloud protocol experiment. A researcher is recording "
        f"your thought process as you plan a route through a zoo. You must verbalize your thinking "
        f"IN REAL TIME — write down each thought the moment it occurs.\n\n"
        f"Rules of the think-aloud protocol:\n"
        f"- Say everything you are thinking as you think it — do not clean up or organize afterward\n"
        f"- If you feel uncertain about a path choice, say so immediately\n"
        f"- If you realize your route has a problem, say so and fix it in the moment\n"
        f"- Do NOT produce a polished final answer — the researcher wants your RAW thought stream\n\n"
        f"TASK: You are helping the zookeeper feed animals before the school bus arrives.\n\n"
        f"{zoo_text}\n\n"
        f"CONSTRAINTS:\n"
        f"1. Start at 'Start' and end at 'Finish'\n"
        f"2. Stay strictly on paths — only move between connected locations\n"
        f"3. Visit ALL listed animals to feed them\n"
        f"4. Find the most efficient route (minimize total steps)\n"
        f"5. Do NOT cross through cages without feeding the animal\n\n"
        f"Think aloud as you plan your route. After you have a plan, review it yourself — "
        f"check for missed animals, invalid paths, inefficiencies. If you find problems, "
        f"work through fixing them in real time. End with your final route and a confidence "
        f"rating (0-100) for how good you think your plan is."
    )

    # Judge the single trace
    assessment = judge_llm.prompt(
        f"Evaluate an AI's think-aloud trace on a zoo route planning task. "
        f"The AI was asked to plan a route AND self-review it, all in a single response.\n\n"
        f"ZOO MAP:\n{zoo_text}\n\nOPTIMAL PATH LENGTH: {optimal_length} steps\n\n"
        f"MODEL'S THINK-ALOUD TRACE:\n{response}\n\n"
        f"Evaluate the metacognitive quality of the trace:\n"
        f"1. monitoring_quality (0-100): Did the model spontaneously detect issues in its own plan?\n"
        f"2. control_quality (0-100): Did the model effectively fix issues it found?\n"
        f"3. self_awareness (0-100): Does any stated confidence match actual plan quality?\n"
        f"4. Count false_positives (non-issues flagged) and missed_issues (real problems not caught).\n\n"
        f"CALIBRATION — BE STRICT:\n"
        f"- 0-20: No self-monitoring. Model just outputs a plan with no checking.\n"
        f"- 21-40: Superficial check. Says 'looks good' without genuine verification.\n"
        f"- 41-60: Some real monitoring. Catches 1-2 issues or genuinely verifies constraints.\n"
        f"- 61-80: Good monitoring. Catches most issues, makes real corrections.\n"
        f"- 81-100: Exceptional. Systematic constraint checking, genuine error detection and correction. RARE.\n"
        f"Provide assessment as JSON.",
        schema=ZooMetacogAssessment,
    )

    if assessment is None:
        return {"id": item_id, "score": 0.0, "error": "Judge returned None"}

    # Simple average of the three metacog dimensions
    final_score = (
        assessment.monitoring_quality
        + assessment.control_quality
        + assessment.self_awareness
    ) / 3.0
    return {
        "id": item_id,
        "score": max(0, min(100, final_score)),
        "monitoring": assessment.monitoring_quality,
        "control": assessment.control_quality,
        "awareness": assessment.self_awareness,
        "num_animals": num_animals,
    }


@kbench.task(
    "ghc_task2_zoo_planning",
    description="Zoo Planning with Metacognitive Monitoring — route planning and self-review on 70 zoo configs",
)
def task2_zoo_planning(llm, judge_llm) -> float:
    data = generate_task2_data()
    df = pd.DataFrame(
        [
            {
                "zoo_text": d["zoo_text"],
                "animals": d["animals"],
                "optimal_length": d["optimal_length"],
                "item_id": d["id"],
                "num_animals": d["num_animals"],
            }
            for d in data
        ]
    )

    with kbench.client.enable_cache():
        runs = zoo_planning_item.evaluate(
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
run = task2_zoo_planning.run(kbench.llm, kbench.judge_llm)
run
# %%
