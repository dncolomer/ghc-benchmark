# %% [markdown]
# # Genuine Human Cognition Benchmark (GHC Benchmark) v1
# **Track: Metacognition**
#
# Probes metacognitive capabilities: self-monitoring of planning traces,
# error detection, effort calibration, and reasoning revision.
#
# 5 Tasks:
# 1. CoT Linearity Analysis
# 2. Zoo Planning with Metacognitive Monitoring
# 3. Verbal Traces Comparison (Game of 24)
# 4. Metacognitive Self-Interrogation Loop
# 5. Reasoning Effort Variability & Calibration

# %%
import kaggle_benchmarks as kbench
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import random
import json
import re
import itertools
from collections import deque

# %%
# =============================================================================
# DATA GENERATORS
# =============================================================================

random.seed(42)

# ---------------------------------------------------------------------------
# Task 1 Data: Logic puzzles (60 items)
# ---------------------------------------------------------------------------


def _river_crossing_puzzles():
    """Generate river crossing puzzle variants."""
    items = []
    # Classic fox-goose-beans
    items.append(
        {
            "id": "rc_001",
            "puzzle": (
                "A farmer needs to cross a river with a fox, a goose, and a bag of beans. "
                "The boat can carry only the farmer and one item at a time. "
                "If left alone, the fox will eat the goose, and the goose will eat the beans. "
                "How can the farmer get everything across safely?"
            ),
            "category": "river_crossing",
            "difficulty": "medium",
            "has_known_solution": True,
        }
    )
    # Variants with different entities
    triples = [
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
    for i, (a, b, c, r1, r2) in enumerate(triples):
        items.append(
            {
                "id": f"rc_{i + 2:03d}",
                "puzzle": (
                    f"A farmer needs to cross a river with a {a}, a {b}, and a bag of {c}. "
                    f"The boat can carry only the farmer and one item. "
                    f"If left alone, the {r1}, and the {r2}. "
                    f"How can the farmer get everything across safely?"
                ),
                "category": "river_crossing",
                "difficulty": "medium",
                "has_known_solution": True,
            }
        )
    return items


def _constraint_puzzles():
    """Generate constraint satisfaction / deduction puzzles."""
    items = []
    # Who-lives-where style puzzles
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
    pets_pool = ["dog", "cat", "bird", "fish", "hamster", "turtle", "rabbit", "snake"]

    for i in range(20):
        n = random.choice([3, 4])
        names = random.sample(names_pool, n)
        colors = random.sample(colors_pool, n)
        # Create a fixed assignment for ground truth
        assignment = list(zip(names, colors))
        random.shuffle(assignment)
        houses = {
            j + 1: {"name": assignment[j][0], "color": assignment[j][1]}
            for j in range(n)
        }

        # Generate clues from the assignment
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
            if random.random() > 0.5:
                clues.append(f"The {color} house is house number {pos}.")

        random.shuffle(clues)
        clue_text = "\n".join(f"{k + 1}. {c}" for k, c in enumerate(clues))

        items.append(
            {
                "id": f"cp_{i + 1:03d}",
                "puzzle": (
                    f"There are {n} houses in a row, numbered 1 to {n} from left to right. "
                    f"Each house has a unique color and is occupied by a different person.\n"
                    f"People: {', '.join(names)}\n"
                    f"Colors: {', '.join(colors)}\n\n"
                    f"Clues:\n{clue_text}\n\n"
                    f"Determine who lives in which colored house."
                ),
                "category": "constraint_satisfaction",
                "difficulty": random.choice(["easy", "medium", "hard"]),
                "has_known_solution": True,
            }
        )
    return items


def _logic_deduction_puzzles():
    """Generate various logic/deduction puzzles."""
    templates = [
        (
            "Three friends — {a}, {b}, and {c} — each ordered a different drink: coffee, tea, and juice. "
            "{a} didn't order coffee. {b} didn't order tea. {c} ordered juice. "
            "What did each person order?",
            "easy",
        ),
        (
            "Four students scored differently on a test. {a} scored higher than {b}. "
            "{c} scored lower than {b} but higher than {d}. {a} did not get the highest score. "
            "Rank all four students from highest to lowest.",
            "medium",
        ),
        (
            "In a row of five seats, {a} sits to the left of {b}. {c} sits at one end. "
            "{d} sits between {a} and {c}. {e} sits next to {b}. "
            "Determine the seating order from left to right.",
            "hard",
        ),
        (
            "{a} says '{b} is lying.' {b} says '{c} is lying.' {c} says 'Both {a} and {b} are lying.' "
            "If exactly one person is telling the truth, who is it? Explain your reasoning step by step.",
            "medium",
        ),
        (
            "A hat puzzle: {a}, {b}, and {c} each wear a hat that is either red or blue. "
            "Each can see the others' hats but not their own. {a} says 'I don't know my hat color.' "
            "{b} says 'I don't know either.' {c} then correctly guesses their own hat color. "
            "If there is at least one red hat, what color is {c}'s hat and why?",
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
        tmpl, diff = templates[i % len(templates)]
        names = random.sample(names_pool, 5)
        puzzle = tmpl.format(a=names[0], b=names[1], c=names[2], d=names[3], e=names[4])
        items.append(
            {
                "id": f"ld_{i + 1:03d}",
                "puzzle": puzzle,
                "category": "logic_deduction",
                "difficulty": diff,
                "has_known_solution": True,
            }
        )
    return items


def generate_task1_data():
    """Generate 60 logic puzzles for Task 1."""
    items = []
    items.extend(_river_crossing_puzzles())  # 5
    items.extend(_constraint_puzzles())  # 20
    items.extend(_logic_deduction_puzzles())  # 35
    random.shuffle(items)
    return items[:60]


# ---------------------------------------------------------------------------
# Task 2 Data: Zoo planning (70 items)
# ---------------------------------------------------------------------------


def generate_zoo_graph(num_animals, seed_val):
    """Generate a zoo graph with start, finish, animal cages, and paths.
    Returns adjacency list, animal positions, and optimal path."""
    rng = random.Random(seed_val)

    # Create a grid-based zoo: nodes arranged in a grid
    # Grid size depends on num_animals
    if num_animals <= 4:
        rows, cols = 3, 3
    elif num_animals <= 6:
        rows, cols = 3, 4
    else:
        rows, cols = 4, 4

    # Node naming: (r, c) -> label
    nodes = []
    for r in range(rows):
        for c in range(cols):
            nodes.append((r, c))

    # Start = top-left, Finish = bottom-right
    start = (0, 0)
    finish = (rows - 1, cols - 1)

    # Place animals at random interior nodes (not start/finish)
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
    animal_positions = {}
    cage_nodes = set()
    for i, animal in enumerate(animals_selected):
        pos = available[i]
        animal_positions[animal] = pos
        cage_nodes.add(pos)

    # Build adjacency (grid paths — up/down/left/right)
    adjacency = {n: [] for n in nodes}
    for r in range(rows):
        for c in range(cols):
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    adjacency[(r, c)].append((nr, nc))

    # Remove some random edges to make it more interesting (but keep connected)
    # Actually, let's keep the full grid but add the "no crossing cages" constraint:
    # You can visit a cage node (to feed the animal) but you can't pass THROUGH
    # a cage node without feeding. We'll encode this as: all cage nodes must be
    # in the path if they're traversed.

    # Find optimal path visiting all animals using BFS on state space
    # State: (current_node, frozenset of visited animals)
    target = frozenset(animals_selected)
    pos_to_animal = {v: k for k, v in animal_positions.items()}

    from collections import deque

    initial_visited = frozenset()
    if start in pos_to_animal:
        initial_visited = frozenset([pos_to_animal[start]])

    queue = deque()
    queue.append((start, initial_visited, [start]))
    seen = {(start, initial_visited)}
    optimal_path = None

    while queue:
        curr, visited_animals, path = queue.popleft()
        if curr == finish and visited_animals == target:
            optimal_path = path
            break
        for neighbor in adjacency[curr]:
            new_visited = visited_animals
            if neighbor in pos_to_animal:
                new_visited = visited_animals | frozenset([pos_to_animal[neighbor]])
            state = (neighbor, new_visited)
            if state not in seen:
                seen.add(state)
                queue.append((neighbor, new_visited, path + [neighbor]))

    # Build text representation
    node_labels = {}
    node_labels[start] = "Start"
    node_labels[finish] = "Finish"
    for animal, pos in animal_positions.items():
        node_labels[pos] = f"{animal} Cage"

    # Label remaining nodes
    junction_count = 0
    for n in nodes:
        if n not in node_labels:
            junction_count += 1
            node_labels[n] = f"Junction-{junction_count}"

    # Build text adjacency
    text_lines = []
    text_lines.append(f"Zoo Map ({rows}x{cols} grid):")
    text_lines.append(f"Start: {node_labels[start]} at position {start}")
    text_lines.append(f"Finish: {node_labels[finish]} at position {finish}")
    text_lines.append(f"\nAnimals to feed: {', '.join(animals_selected)}")
    text_lines.append(f"\nPaths (bidirectional connections):")
    seen_edges = set()
    for n in nodes:
        for nb in adjacency[n]:
            edge = tuple(sorted([n, nb]))
            if edge not in seen_edges:
                seen_edges.add(edge)
                text_lines.append(f"  {node_labels[n]} <-> {node_labels[nb]}")

    # ASCII map
    text_lines.append(f"\nGrid layout (row, col):")
    for r in range(rows):
        row_str = []
        for c in range(cols):
            label = node_labels[(r, c)]
            short = label[:8].ljust(8)
            row_str.append(f"[{short}]")
        text_lines.append("  " + " -- ".join(row_str))
        if r < rows - 1:
            spacers = []
            for c in range(cols):
                spacers.append("     |    ")
            text_lines.append("  " + "    ".join(spacers))

    zoo_text = "\n".join(text_lines)

    # Optimal path as text
    if optimal_path:
        opt_text = " -> ".join(node_labels[n] for n in optimal_path)
        opt_length = len(optimal_path)
    else:
        opt_text = "No valid path found"
        opt_length = -1

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
    """Generate 70 zoo planning configs."""
    items = []
    # 24 easy (4 animals), 24 medium (6 animals), 22 hard (8 animals)
    configs = [(4, 24), (6, 24), (8, 22)]
    idx = 0
    for num_animals, count in configs:
        for i in range(count):
            zoo = generate_zoo_graph(num_animals, seed_val=idx * 1000 + i)
            if zoo["optimal_length"] > 0:  # valid config
                items.append(
                    {
                        "id": f"zoo_{idx + 1:03d}",
                        "zoo_text": zoo["zoo_text"],
                        "animals": zoo["animals"],
                        "num_animals": num_animals,
                        "optimal_path": zoo["optimal_path"],
                        "optimal_length": zoo["optimal_length"],
                        "adjacency": zoo["adjacency"],
                        "animal_positions": zoo["animal_positions"],
                        "node_labels": zoo["node_labels"],
                        "difficulty": {4: "easy", 6: "medium", 8: "hard"}[num_animals],
                    }
                )
                idx += 1
    return items[:70]


# ---------------------------------------------------------------------------
# Task 3 Data: Game of 24 (80 items)
# ---------------------------------------------------------------------------


def _solve_24(nums):
    """Check if 4 numbers can make 24 using +,-,*,/. Return a solution or None."""
    if len(nums) == 1:
        if abs(nums[0] - 24) < 1e-9:
            return True
        return False

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
                elif op == "/":
                    if abs(b) < 1e-9:
                        continue
                    res = a / b
                if _solve_24(rest + [res]):
                    return True
    return False


def generate_task3_data():
    """Generate 80 Game-of-24 items with verified solvability."""
    items = []
    seen = set()
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
                {
                    "id": f"g24_{len(items) + 1:03d}",
                    "numbers": list(nums),
                    "target": 24,
                    "difficulty": "medium",
                }
            )
    return items[:80]


# ---------------------------------------------------------------------------
# Task 4 Data: Mixed problems for self-interrogation (60 items)
# ---------------------------------------------------------------------------


def generate_task4_data():
    """Generate 60 mixed logic/math/ambiguous problems."""
    items = []

    # Math problems with common pitfalls
    math_problems = [
        {
            "problem": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
            "answer": "$0.05",
            "trap": "Many say $0.10",
        },
        {
            "problem": "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
            "answer": "5 minutes",
            "trap": "Many say 100 minutes",
        },
        {
            "problem": "In a lake, there is a patch of lily pads. Every day, the patch doubles in size. If it takes 48 days for the patch to cover the entire lake, how long would it take for the patch to cover half of the lake?",
            "answer": "47 days",
            "trap": "Many say 24 days",
        },
        {
            "problem": "A farmer has 17 sheep. All but 9 die. How many sheep does the farmer have left?",
            "answer": "9",
            "trap": "Many say 8",
        },
        {
            "problem": "How many times can you subtract 5 from 25?",
            "answer": "Once (then it's 20, not 25)",
            "trap": "Many say 5 times",
        },
        {
            "problem": "If you have a bowl with six apples and you take away four, how many do you have?",
            "answer": "4 (you took them)",
            "trap": "Many say 2",
        },
        {
            "problem": "A doctor gives you 3 pills and tells you to take one every half hour. How long until all pills are taken?",
            "answer": "1 hour",
            "trap": "Many say 1.5 hours",
        },
        {
            "problem": "Some months have 30 days, some have 31. How many months have 28 days?",
            "answer": "All 12 months",
            "trap": "Many say 1 (February)",
        },
    ]

    # Logic puzzles requiring careful reasoning
    logic_problems = [
        {
            "problem": "There are three light switches outside a room. One controls a light bulb inside. You can flip switches as much as you want but can only enter the room once. How do you figure out which switch controls the bulb?",
            "answer": "Turn switch 1 on for a while, turn it off, turn switch 2 on, enter. If bulb is on = switch 2, if off and warm = switch 1, if off and cold = switch 3.",
            "trap": "Requires multi-step physical reasoning",
        },
        {
            "problem": "You have 12 identical-looking balls. One is a different weight (heavier or lighter). Using a balance scale exactly 3 times, find the odd ball and determine if it's heavier or lighter.",
            "answer": "Divide into groups of 4, weigh 4 vs 4. Based on result, narrow down with 2 more weighings.",
            "trap": "Requires systematic elimination",
        },
        {
            "problem": "Two fathers and two sons go fishing. They each catch one fish. They bring home exactly 3 fish. How?",
            "answer": "There are 3 people: grandfather, father, and son. The father is both a father and a son.",
            "trap": "Assumes 4 separate people",
        },
        {
            "problem": "I speak without a mouth and hear without ears. I have no body, but I come alive with wind. What am I?",
            "answer": "An echo",
            "trap": "Riddle requiring lateral thinking",
        },
        {
            "problem": "What disappears as soon as you say its name?",
            "answer": "Silence",
            "trap": "Simple riddle but easy to overthink",
        },
    ]

    # Ambiguous / tricky problems
    ambiguous_problems = [
        {
            "problem": "Is the following statement true or false: 'This statement is false.' Analyze carefully.",
            "answer": "This is a paradox (the Liar's Paradox) — it cannot be consistently assigned true or false.",
            "trap": "Models may try to assign a definitive truth value",
        },
        {
            "problem": "A plane crashes exactly on the US-Canada border. Where do you bury the survivors?",
            "answer": "You don't bury survivors.",
            "trap": "Focuses on the border detail instead of 'survivors'",
        },
        {
            "problem": "If there are 3 apples and you take away 2, how many apples do YOU have?",
            "answer": "2 (the ones you took)",
            "trap": "Many say 1",
        },
        {
            "problem": "What is 0.1 + 0.2? Discuss any nuances.",
            "answer": "Mathematically 0.3, but in floating-point it's 0.30000000000000004. The nuance depends on context.",
            "trap": "Context-dependent answer",
        },
        {
            "problem": "Is it possible for a man to marry his widow's sister? Explain.",
            "answer": "No — if he has a widow, he is dead.",
            "trap": "Models may say yes",
        },
    ]

    # Expand with parametric variations
    rng = random.Random(456)
    names = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Hank"]
    base_items = []

    for mp in math_problems:
        base_items.append(
            {
                "problem": mp["problem"],
                "answer": mp["answer"],
                "category": "math_trap",
                "difficulty": "medium",
            }
        )

    for lp in logic_problems:
        base_items.append(
            {
                "problem": lp["problem"],
                "answer": lp["answer"],
                "category": "logic",
                "difficulty": "hard",
            }
        )

    for ap in ambiguous_problems:
        base_items.append(
            {
                "problem": ap["problem"],
                "answer": ap["answer"],
                "category": "ambiguous",
                "difficulty": "hard",
            }
        )

    # Generate more math problems programmatically
    for i in range(42):
        a = rng.randint(2, 15)
        b = rng.randint(2, 15)
        c = rng.randint(2, 15)
        ptype = i % 6
        if ptype == 0:
            problem = f"If {a} workers can complete a job in {b} days, how many days would {c} workers take (assuming constant rate per worker)?"
            answer = f"{a * b / c:.2f} days"
            cat = "math"
            diff = "easy"
        elif ptype == 1:
            problem = f"A train travels {a * 10} km at {b * 10} km/h, then {c * 10} km at {(b + 3) * 10} km/h. What is the average speed for the entire journey?"
            total_dist = a * 10 + c * 10
            total_time = (a * 10) / (b * 10) + (c * 10) / ((b + 3) * 10)
            answer = (
                f"{total_dist / total_time:.2f} km/h (harmonic mean, not arithmetic)"
            )
            cat = "math_trap"
            diff = "medium"
        elif ptype == 2:
            n1, n2 = rng.sample(names, 2)
            problem = f"{n1} is twice as old as {n2}. In {a} years, {n1} will be {b} years older than {n2}. How old are they now?"
            # n1 = 2*n2, n1+a = (n2+a) + b => 2*n2 + a = n2 + a + b => n2 = b
            answer = f"{n2} is {b}, {n1} is {2 * b}"
            cat = "math"
            diff = "easy"
        elif ptype == 3:
            problem = (
                f"What is the remainder when {a * 100 + b * 10 + c} is divided by {a}?"
            )
            answer = f"{(a * 100 + b * 10 + c) % a}"
            cat = "math"
            diff = "easy"
        elif ptype == 4:
            problem = f"A snail climbs {a} feet during the day but slides back {a - 1} feet at night. How many days to reach the top of a {a * c} foot well?"
            # Each day net +1, except last day climbs a and is out
            target = a * c
            if a >= target:
                days = 1
            else:
                days = target - a + 1  # (target - a) nights of +1, then final day
            answer = f"{days} days"
            cat = "math_trap"
            diff = "medium"
        else:
            problem = f"You flip a fair coin {a} times. What is the probability of getting exactly {min(b, a)} heads? Express as a fraction or decimal."
            from math import comb, factorial

            k = min(b, a)
            prob = comb(a, k) / (2**a)
            answer = f"{comb(a, k)}/{2**a} = {prob:.6f}"
            cat = "math"
            diff = "medium"

        base_items.append(
            {"problem": problem, "answer": answer, "category": cat, "difficulty": diff}
        )

    for i, item in enumerate(base_items):
        item["id"] = f"si_{i + 1:03d}"

    return base_items[:60]


# ---------------------------------------------------------------------------
# Task 5 Data: Tiered problems for effort calibration (80 items)
# ---------------------------------------------------------------------------


def generate_task5_data():
    """Generate 80 tiered problems: easy/medium/hard."""
    items = []
    rng = random.Random(789)

    # Easy: simple arithmetic (27 items)
    for i in range(27):
        a = rng.randint(10, 999)
        b = rng.randint(10, 999)
        op = rng.choice(["+", "-", "*"])
        if op == "+":
            ans = a + b
        elif op == "-":
            ans = a - b
        else:
            a = rng.randint(10, 99)
            b = rng.randint(10, 99)
            ans = a * b
        items.append(
            {
                "id": f"ec_{i + 1:03d}",
                "problem": f"What is {a} {op} {b}?",
                "answer": str(ans),
                "true_difficulty": "easy",
                "category": "arithmetic",
            }
        )

    # Medium: Game of 24 (27 items)
    g24_data = generate_task3_data()
    for i in range(27):
        d = g24_data[i]
        items.append(
            {
                "id": f"ec_{28 + i:03d}",
                "problem": f"Using the numbers {d['numbers']}, make 24 using +, -, *, / (each number used exactly once).",
                "answer": "24",
                "true_difficulty": "medium",
                "category": "game_of_24",
            }
        )

    # Hard: mini-planning problems (26 items)
    planning_templates = [
        "You have {n} tasks with dependencies: {deps}. Find a valid execution order that respects all dependencies and minimizes total time.",
        "Schedule {n} meetings in {r} rooms. Each meeting has a time range. Minimize conflicts. Meetings: {meetings}",
        "You have {n} items with weights {weights} and values {values}. Your bag holds {cap} kg. Maximize value.",
    ]
    for i in range(26):
        tmpl_idx = i % 3
        if tmpl_idx == 0:
            n = rng.randint(4, 7)
            tasks_list = [chr(65 + j) for j in range(n)]
            deps = []
            for j in range(1, n):
                dep_from = rng.choice(tasks_list[:j])
                deps.append(f"{dep_from}->{tasks_list[j]}")
            problem = planning_templates[0].format(n=n, deps=", ".join(deps))
            answer = "topological_sort"
        elif tmpl_idx == 1:
            n = rng.randint(4, 6)
            r = rng.randint(2, 3)
            meetings = []
            for j in range(n):
                start = rng.randint(9, 15)
                end = start + rng.randint(1, 3)
                meetings.append(f"M{j + 1}({start}:00-{end}:00)")
            problem = planning_templates[1].format(
                n=n, r=r, meetings=", ".join(meetings)
            )
            answer = "scheduling"
        else:
            n = rng.randint(4, 6)
            weights = [rng.randint(1, 10) for _ in range(n)]
            values = [rng.randint(5, 50) for _ in range(n)]
            cap = sum(weights) // 2
            problem = planning_templates[2].format(
                n=n, weights=weights, values=values, cap=cap
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
    return items[:80]


# %%
# =============================================================================
# GENERATE ALL DATASETS
# =============================================================================

task1_data = generate_task1_data()
task2_data = generate_task2_data()
task3_data = generate_task3_data()
task4_data = generate_task4_data()
task5_data = generate_task5_data()

print(f"Task 1: {len(task1_data)} items")
print(f"Task 2: {len(task2_data)} items")
print(f"Task 3: {len(task3_data)} items")
print(f"Task 4: {len(task4_data)} items")
print(f"Task 5: {len(task5_data)} items")
print(
    f"Total:  {len(task1_data) + len(task2_data) + len(task3_data) + len(task4_data) + len(task5_data)} items"
)

# %%
# =============================================================================
# JUDGE SCHEMAS
# =============================================================================


@dataclass
class LinearityAssessment:
    """Assessment of CoT linearity for Task 1."""

    linearity_index: (
        int  # 0=highly non-linear (lots of revision/backtracking), 100=perfectly linear
    )
    back_references_count: (
        int  # Number of times the model refers back to previous steps
    )
    abandoned_branches_count: int  # Number of reasoning paths started then abandoned
    self_doubt_phrases_count: int  # e.g. "wait", "actually", "let me reconsider", "hmm"
    revision_count: int  # Number of explicit corrections/revisions
    monitoring_phrases: List[str]  # Extracted self-monitoring phrases
    reasoning: str  # Brief explanation of the assessment


@dataclass
class ZooRouteCheck:
    """Structured check of a zoo route."""

    all_animals_visited: bool
    starts_at_start: bool
    ends_at_finish: bool
    path_valid: bool  # All edges exist in adjacency
    path_length: int
    issues: List[str]


@dataclass
class ZooMetacogAssessment:
    """Judge assessment of zoo metacognitive monitoring."""

    monitoring_quality: int  # 0-100: How well did the model detect actual issues?
    control_quality: int  # 0-100: How well did the model revise/fix issues?
    self_awareness: int  # 0-100: Accuracy of confidence vs actual performance
    false_positives: int  # Issues reported that weren't real
    missed_issues: int  # Real issues not detected
    reasoning: str


@dataclass
class VerbalTraceAssessment:
    """Assessment of verbal reasoning trace for Task 3."""

    subgoals_identified: int  # Number of explicit subgoals stated
    revisions_count: int  # Times the model revised approach
    stuck_moments: int  # Times model expressed being stuck
    effort_adjustments: int  # Times model changed strategy
    metacog_richness: int  # 0-100 overall metacognitive richness
    correct_answer: bool
    reasoning: str


@dataclass
class SelfInterrogationAssessment:
    """Judge assessment of self-interrogation quality for Task 4."""

    interrogation_depth: (
        int  # 0-100: How thoroughly did model examine its own reasoning?
    )
    real_flaws_found: int  # Number of genuine flaws identified
    false_flaws: int  # Number of non-issues flagged as flaws
    missed_flaws: int  # Flaws present but not found
    correctness_improved: bool  # Did revision improve the answer?
    reasoning: str


@dataclass
class EffortPrediction:
    """Model's self-predicted effort."""

    predicted_difficulty: str  # easy, medium, hard
    estimated_steps: int
    reasoning: str


@dataclass
class EffortCalibrationAssessment:
    """Assessment of effort calibration for Task 5."""

    actual_steps_observed: int
    trace_length_tokens_approx: int
    strategy_adjustments: int  # Times model changed approach
    calibration_score: int  # 0-100: How well prediction matched reality
    reasoning: str


# %%
# =============================================================================
# TASK 1: CoT Linearity Analysis
# =============================================================================


@kbench.task("cot_linearity_item", store_task=False)
def cot_linearity_item(llm, judge_llm, puzzle: str, item_id: str) -> dict:
    """Evaluate metacognitive trace linearity on a single puzzle."""

    # Get the model's CoT
    response = llm.prompt(
        f"Solve the following puzzle. Think step by step, showing ALL your reasoning, "
        f"including any doubts, revisions, or changes in approach. Do NOT skip steps.\n\n"
        f"Puzzle: {puzzle}"
    )

    # Judge the linearity of the trace
    assessment = judge_llm.prompt(
        f"Analyze the following reasoning trace for metacognitive indicators. "
        f"Assess how LINEAR vs NON-LINEAR the reasoning is.\n\n"
        f"A LINEAR trace goes step-by-step without backtracking, revision, or self-doubt.\n"
        f"A NON-LINEAR trace includes back-references, abandoned branches, self-corrections, "
        f"monitoring phrases like 'wait', 'actually', 'let me reconsider', 'hmm', etc.\n\n"
        f"Reasoning trace:\n{response}\n\n"
        f"Provide your assessment as JSON.",
        schema=LinearityAssessment,
    )

    if assessment is None:
        return {"id": item_id, "score": 0.0, "error": "Judge returned None"}

    # Score: more non-linearity = more metacognitive = higher score
    score = 100 - assessment.linearity_index
    # Bonus for specific metacog indicators (capped at 100)
    bonus = min(
        20,
        (
            assessment.back_references_count * 2
            + assessment.abandoned_branches_count * 3
            + assessment.self_doubt_phrases_count * 2
            + assessment.revision_count * 3
        ),
    )
    final_score = min(100, score + bonus)

    return {
        "id": item_id,
        "score": final_score,
        "linearity_index": assessment.linearity_index,
        "back_refs": assessment.back_references_count,
        "abandoned": assessment.abandoned_branches_count,
        "self_doubt": assessment.self_doubt_phrases_count,
        "revisions": assessment.revision_count,
    }


@kbench.task(
    "task1_cot_linearity",
    description="CoT Linearity Analysis - measures metacognitive trace non-linearity",
)
def task1_cot_linearity(llm, judge_llm) -> float:
    """Task 1: Evaluate CoT linearity across 60 logic puzzles."""
    data = generate_task1_data()
    df = pd.DataFrame([{"puzzle": d["puzzle"], "item_id": d["id"]} for d in data])

    with kbench.client.enable_cache():
        runs = cot_linearity_item.evaluate(
            stop_condition=lambda runs: len(runs) == len(data),
            max_attempts=2,
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
# =============================================================================
# TASK 2: Zoo Planning with Metacognitive Monitoring
# =============================================================================


@kbench.task("zoo_planning_item", store_task=False)
def zoo_planning_item(
    llm,
    judge_llm,
    zoo_text: str,
    animals: list,
    optimal_length: int,
    adjacency: dict,
    animal_positions: dict,
    node_labels: dict,
    item_id: str,
    num_animals: int,
) -> dict:
    """Single zoo planning item with metacognitive monitoring."""

    # Turn 1: Plan the route
    plan_response = llm.prompt(
        f"You are helping the zookeeper feed the animals before the school bus arrives.\n\n"
        f"{zoo_text}\n\n"
        f"RULES:\n"
        f"1. Start at 'Start' and end at 'Finish'\n"
        f"2. Stay strictly on paths — you can only move between connected locations\n"
        f"3. Visit ALL listed animals to feed them\n"
        f"4. Find the most efficient route (minimize total steps)\n"
        f"5. Do NOT cross through cages without feeding the animal\n\n"
        f"Think aloud step-by-step and describe your planned route."
    )

    # Turn 2: Metacognitive monitoring — self-review
    review_response = llm.prompt(
        f"Now review your route carefully for ANY violations or inefficiencies:\n"
        f"- Did you miss any animals?\n"
        f"- Did you start at Start and end at Finish?\n"
        f"- Are all your path connections valid (exist on the map)?\n"
        f"- Could the route be shorter?\n"
        f"- Did you cross any cages unnecessarily?\n\n"
        f"List ALL issues you find. Then provide a revised plan if needed.\n\n"
        f"Output your response as JSON with these fields:\n"
        f"- detected_issues: list of strings describing each issue found\n"
        f"- revised_plan: string describing your revised route (or original if no issues)\n"
        f"- confidence_in_plan: integer 0-100 representing your confidence"
    )

    # Judge the metacognitive quality
    metacog_assessment = judge_llm.prompt(
        f"You are evaluating an AI's metacognitive monitoring on a zoo route planning task.\n\n"
        f"ZOO MAP:\n{zoo_text}\n\n"
        f"OPTIMAL PATH LENGTH: {optimal_length} steps\n\n"
        f"MODEL'S INITIAL PLAN:\n{plan_response}\n\n"
        f"MODEL'S SELF-REVIEW:\n{review_response}\n\n"
        f"Evaluate:\n"
        f"1. monitoring_quality (0-100): Did the model accurately detect real issues in its plan?\n"
        f"2. control_quality (0-100): Did the model effectively revise/fix issues?\n"
        f"3. self_awareness (0-100): Does confidence match actual plan quality?\n"
        f"4. Count false_positives (fake issues) and missed_issues (real issues not caught).\n\n"
        f"Provide assessment as JSON.",
        schema=ZooMetacogAssessment,
    )

    if metacog_assessment is None:
        return {"id": item_id, "score": 0.0, "error": "Judge returned None"}

    # Combined score
    route_score = max(0, 100 - max(0, (metacog_assessment.missed_issues * 15)))
    monitoring_score = metacog_assessment.monitoring_quality
    control_score = metacog_assessment.control_quality
    awareness_score = metacog_assessment.self_awareness

    final_score = (
        0.30 * route_score
        + 0.30 * monitoring_score
        + 0.20 * control_score
        + 0.20 * awareness_score
    )

    return {
        "id": item_id,
        "score": final_score,
        "route_score": route_score,
        "monitoring": monitoring_score,
        "control": control_score,
        "awareness": awareness_score,
        "num_animals": num_animals,
    }


@kbench.task(
    "task2_zoo_planning",
    description="Zoo Planning with Metacognitive Monitoring - route planning and self-review",
)
def task2_zoo_planning(llm, judge_llm) -> float:
    """Task 2: Zoo planning with metacognitive monitoring across 70 configs."""
    data = generate_task2_data()
    df = pd.DataFrame(
        [
            {
                "zoo_text": d["zoo_text"],
                "animals": d["animals"],
                "optimal_length": d["optimal_length"],
                "adjacency": d["adjacency"],
                "animal_positions": d["animal_positions"],
                "node_labels": d["node_labels"],
                "item_id": d["id"],
                "num_animals": d["num_animals"],
            }
            for d in data
        ]
    )

    with kbench.client.enable_cache():
        runs = zoo_planning_item.evaluate(
            stop_condition=lambda runs: len(runs) == len(data),
            max_attempts=2,
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
# =============================================================================
# TASK 3: Verbal Traces Comparison (Game of 24)
# =============================================================================


@kbench.task("verbal_trace_item", store_task=False)
def verbal_trace_item(llm, judge_llm, numbers: list, item_id: str) -> dict:
    """Single Game of 24 with verbal trace analysis."""

    response = llm.prompt(
        f"Solve this Game of 24 puzzle. Use the numbers {numbers} with +, -, *, / "
        f"(each number exactly once) to make 24.\n\n"
        f"IMPORTANT: Verbalize EVERY thought, subgoal, revision, or doubt. "
        f"If you get stuck, say so explicitly. If you change strategy, explain why. "
        f"Show your complete inner monologue as you work through this."
    )

    # Judge the verbal trace
    assessment = judge_llm.prompt(
        f"Analyze this think-aloud reasoning trace for a Game of 24 puzzle.\n"
        f"Numbers: {numbers}, Target: 24\n\n"
        f"Trace:\n{response}\n\n"
        f"Evaluate the metacognitive content of the trace:\n"
        f"1. subgoals_identified: How many explicit subgoals did the model state?\n"
        f"2. revisions_count: How many times did it revise its approach?\n"
        f"3. stuck_moments: How many times did it express being stuck/uncertain?\n"
        f"4. effort_adjustments: How many times did it change strategy?\n"
        f"5. metacog_richness: Overall 0-100 score for metacognitive richness\n"
        f"6. correct_answer: Did the model arrive at a valid solution making 24?\n\n"
        f"Provide assessment as JSON.",
        schema=VerbalTraceAssessment,
    )

    if assessment is None:
        return {"id": item_id, "score": 0.0, "error": "Judge returned None"}

    # Score: metacog richness + correctness bonus
    metacog_score = assessment.metacog_richness
    correctness_bonus = 20 if assessment.correct_answer else 0

    # Bonus for specific trace elements
    trace_bonus = min(
        15,
        (
            assessment.subgoals_identified * 2
            + assessment.revisions_count * 2
            + assessment.stuck_moments * 3
            + assessment.effort_adjustments * 3
        ),
    )

    final_score = min(100, metacog_score * 0.65 + correctness_bonus + trace_bonus)

    return {
        "id": item_id,
        "score": final_score,
        "metacog_richness": assessment.metacog_richness,
        "correct": assessment.correct_answer,
        "subgoals": assessment.subgoals_identified,
        "revisions": assessment.revisions_count,
    }


@kbench.task(
    "task3_verbal_traces",
    description="Verbal Traces Comparison - think-aloud metacognitive analysis on Game of 24",
)
def task3_verbal_traces(llm, judge_llm) -> float:
    """Task 3: Verbal trace analysis across 80 Game of 24 puzzles."""
    data = generate_task3_data()
    df = pd.DataFrame(
        [
            {
                "numbers": d["numbers"],
                "item_id": d["id"],
            }
            for d in data
        ]
    )

    with kbench.client.enable_cache():
        runs = verbal_trace_item.evaluate(
            stop_condition=lambda runs: len(runs) == len(data),
            max_attempts=2,
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
# =============================================================================
# TASK 4: Metacognitive Self-Interrogation Loop
# =============================================================================


@kbench.task("self_interrogation_item", store_task=False)
def self_interrogation_item(
    llm, judge_llm, problem: str, answer: str, item_id: str
) -> dict:
    """Single self-interrogation item: solve, interrogate, revise."""

    # Turn 1: Solve with full CoT
    initial_response = llm.prompt(
        f"Solve the following problem with full step-by-step reasoning. "
        f"Include your internal monologue — every thought, assumption, and deduction.\n\n"
        f"Problem: {problem}"
    )

    # Turn 2: Self-interrogation
    interrogation_response = llm.prompt(
        f"Now interrogate your own reasoning trace above. Carefully examine it for:\n"
        f"- Hidden assumptions you made without justification\n"
        f"- Logical leaps or gaps in reasoning\n"
        f"- Unverified claims or calculations\n"
        f"- Alternative interpretations you didn't consider\n"
        f"- Any errors in logic or math\n\n"
        f"Output your analysis as JSON with these fields:\n"
        f"- flaws_found: list of strings, each describing a specific flaw\n"
        f"- confidence_adjusted: integer 0-100 representing your confidence after self-examination"
    )

    # Turn 3: Revision
    revised_response = llm.prompt(
        f"Based on your self-interrogation, provide your final revised answer. "
        f"If you found real flaws, fix them. If your original answer was correct, confirm it.\n"
        f"State your final answer clearly."
    )

    # Judge the quality of self-interrogation
    assessment = judge_llm.prompt(
        f"Evaluate the quality of an AI's self-interrogation on this problem.\n\n"
        f"PROBLEM: {problem}\n"
        f"CORRECT ANSWER: {answer}\n\n"
        f"INITIAL RESPONSE:\n{initial_response}\n\n"
        f"SELF-INTERROGATION:\n{interrogation_response}\n\n"
        f"REVISED RESPONSE:\n{revised_response}\n\n"
        f"Evaluate:\n"
        f"1. interrogation_depth (0-100): How thoroughly did the model examine its reasoning?\n"
        f"2. real_flaws_found: Count of genuine flaws correctly identified\n"
        f"3. false_flaws: Count of non-issues incorrectly flagged as flaws\n"
        f"4. missed_flaws: Count of real flaws not detected\n"
        f"5. correctness_improved: Did the revised answer improve over the initial?\n\n"
        f"Provide assessment as JSON.",
        schema=SelfInterrogationAssessment,
    )

    if assessment is None:
        return {"id": item_id, "score": 0.0, "error": "Judge returned None"}

    # Score components
    depth_score = assessment.interrogation_depth
    accuracy_bonus = 15 if assessment.correctness_improved else 0
    flaw_penalty = min(30, assessment.false_flaws * 5 + assessment.missed_flaws * 10)

    final_score = max(
        0, min(100, depth_score * 0.7 + accuracy_bonus + 15 - flaw_penalty)
    )

    return {
        "id": item_id,
        "score": final_score,
        "depth": assessment.interrogation_depth,
        "real_flaws": assessment.real_flaws_found,
        "false_flaws": assessment.false_flaws,
        "missed_flaws": assessment.missed_flaws,
        "improved": assessment.correctness_improved,
    }


@kbench.task(
    "task4_self_interrogation",
    description="Metacognitive Self-Interrogation Loop - self-examination and revision",
)
def task4_self_interrogation(llm, judge_llm) -> float:
    """Task 4: Self-interrogation across 60 mixed problems."""
    data = generate_task4_data()
    df = pd.DataFrame(
        [
            {
                "problem": d["problem"],
                "answer": d["answer"],
                "item_id": d["id"],
            }
            for d in data
        ]
    )

    with kbench.client.enable_cache():
        runs = self_interrogation_item.evaluate(
            stop_condition=lambda runs: len(runs) == len(data),
            max_attempts=2,
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
# =============================================================================
# TASK 5: Reasoning Effort Variability & Calibration
# =============================================================================


@kbench.task("effort_calibration_item", store_task=False)
def effort_calibration_item(
    llm, judge_llm, problem: str, true_difficulty: str, answer: str, item_id: str
) -> dict:
    """Single effort calibration item: predict effort then solve."""

    # Turn 1: Predict effort
    prediction_response = llm.prompt(
        f"Before solving the following problem, predict the effort required.\n\n"
        f"Problem: {problem}\n\n"
        f"Output JSON with:\n"
        f"- predicted_difficulty: 'easy', 'medium', or 'hard'\n"
        f"- estimated_steps: integer estimate of how many reasoning steps you'll need\n"
        f"- reasoning: brief explanation of your prediction"
    )

    # Turn 2: Solve with full think-aloud
    solution_response = llm.prompt(
        f"Now solve the problem. Think aloud, showing every step of your reasoning. "
        f"If you change strategy, explain why."
    )

    # Judge the calibration
    assessment = judge_llm.prompt(
        f"Evaluate the effort calibration of an AI model.\n\n"
        f"PROBLEM: {problem}\n"
        f"TRUE DIFFICULTY: {true_difficulty}\n"
        f"CORRECT ANSWER: {answer}\n\n"
        f"MODEL'S EFFORT PREDICTION:\n{prediction_response}\n\n"
        f"MODEL'S SOLUTION TRACE:\n{solution_response}\n\n"
        f"Evaluate:\n"
        f"1. actual_steps_observed: Count the actual reasoning steps in the trace\n"
        f"2. trace_length_tokens_approx: Rough estimate of token count in the solution\n"
        f"3. strategy_adjustments: How many times did the model change approach?\n"
        f"4. calibration_score (0-100): How well did the prediction match reality?\n"
        f"   - Difficulty prediction matches true difficulty\n"
        f"   - Estimated steps roughly matches actual steps\n"
        f"   - Trace length is proportional to actual difficulty (short for easy, long for hard)\n\n"
        f"Provide assessment as JSON.",
        schema=EffortCalibrationAssessment,
    )

    if assessment is None:
        return {"id": item_id, "score": 0.0, "error": "Judge returned None"}

    # Score: calibration accuracy + effort variability
    calibration = assessment.calibration_score

    # Bonus for appropriate trace length variation
    trace_len = assessment.trace_length_tokens_approx
    if true_difficulty == "easy" and trace_len < 200:
        variability_bonus = 10
    elif true_difficulty == "hard" and trace_len > 500:
        variability_bonus = 10
    elif true_difficulty == "medium" and 200 <= trace_len <= 500:
        variability_bonus = 10
    else:
        variability_bonus = 0

    # Strategy adjustment bonus for hard problems
    strategy_bonus = (
        min(10, assessment.strategy_adjustments * 5) if true_difficulty == "hard" else 0
    )

    final_score = min(100, calibration * 0.8 + variability_bonus + strategy_bonus)

    return {
        "id": item_id,
        "score": final_score,
        "calibration": assessment.calibration_score,
        "actual_steps": assessment.actual_steps_observed,
        "trace_tokens": assessment.trace_length_tokens_approx,
        "strategy_adj": assessment.strategy_adjustments,
        "true_difficulty": true_difficulty,
    }


@kbench.task(
    "task5_effort_calibration",
    description="Reasoning Effort Variability & Calibration - effort prediction accuracy",
)
def task5_effort_calibration(llm, judge_llm) -> float:
    """Task 5: Effort calibration across 80 tiered problems."""
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
            max_attempts=2,
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
# =============================================================================
# ROOT BENCHMARK TASK
# =============================================================================


@kbench.task(
    "GenuineHumanCognitionBenchmark-v1",
    description="Genuine Human Cognition Benchmark: Metacognitive Traces, Zoo Planning Monitoring & Thinking Effort in Frontier LLMs",
)
def ghc_benchmark(llm, judge_llm) -> float:
    """
    GHC Benchmark v1 — Metacognition Track

    5 tasks probing metacognitive capabilities:
    1. CoT Linearity Analysis
    2. Zoo Planning with Metacognitive Monitoring
    3. Verbal Traces Comparison (Game of 24)
    4. Metacognitive Self-Interrogation Loop
    5. Reasoning Effort Variability & Calibration

    Returns: Average score across all 5 tasks (0-100).
    """
    task_runs = [
        task1_cot_linearity.run(llm, judge_llm),
        task2_zoo_planning.run(llm, judge_llm),
        task3_verbal_traces.run(llm, judge_llm),
        task4_self_interrogation.run(llm, judge_llm),
        task5_effort_calibration.run(llm, judge_llm),
    ]

    scores = []
    for run in task_runs:
        if run.result is not None:
            scores.append(float(run.result))
        else:
            scores.append(0.0)

    avg = sum(scores) / len(scores) if scores else 0.0

    print(f"\n=== GHC Benchmark Results ===")
    task_names = [
        "CoT Linearity Analysis",
        "Zoo Planning + Metacog",
        "Verbal Traces (Game of 24)",
        "Self-Interrogation Loop",
        "Effort Calibration",
    ]
    for name, score in zip(task_names, scores):
        print(f"  {name}: {score:.1f}/100")
    print(f"  OVERALL: {avg:.1f}/100")

    return avg


# %%
# =============================================================================
# RUN THE BENCHMARK
# =============================================================================

run = ghc_benchmark.run(kbench.llm, kbench.judge_llm)
run

# %%
