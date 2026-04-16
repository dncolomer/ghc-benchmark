# =============================================================================
# GHC Benchmark — Shared Data Generators & Schemas
# =============================================================================
# This file contains all data generators and judge schemas shared across
# the 5 GHC Benchmark tasks. Paste this as the FIRST cell in each task notebook,
# or upload as a Kaggle dataset and import it.

import random
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from math import comb

# =============================================================================
# JUDGE SCHEMAS
# =============================================================================


@dataclass
class LinearityAssessment:
    """Assessment of CoT linearity for Task 1."""

    linearity_index: int  # 0=highly non-linear, 100=perfectly linear
    back_references_count: int
    abandoned_branches_count: int
    self_doubt_phrases_count: int
    revision_count: int
    monitoring_phrases: List[str]
    reasoning: str


@dataclass
class ZooMetacogAssessment:
    """Judge assessment of zoo metacognitive monitoring."""

    monitoring_quality: int  # 0-100
    control_quality: int  # 0-100
    self_awareness: int  # 0-100
    false_positives: int
    missed_issues: int
    reasoning: str


@dataclass
class VerbalTraceAssessment:
    """Assessment of verbal reasoning trace for Task 3."""

    subgoals_identified: int
    revisions_count: int
    stuck_moments: int
    effort_adjustments: int
    metacog_richness: int  # 0-100
    correct_answer: bool
    reasoning: str


@dataclass
class SelfInterrogationAssessment:
    """Judge assessment of self-interrogation quality for Task 4."""

    interrogation_depth: int  # 0-100
    real_flaws_found: int
    false_flaws: int
    missed_flaws: int
    correctness_improved: bool
    reasoning: str


@dataclass
class EffortCalibrationAssessment:
    """Assessment of effort calibration for Task 5."""

    actual_steps_observed: int
    trace_length_tokens_approx: int
    strategy_adjustments: int
    calibration_score: int  # 0-100
    reasoning: str


# =============================================================================
# TASK 1 DATA: Logic Puzzles (60 items)
# =============================================================================


def _river_crossing_puzzles():
    items = []
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
        }
    )
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
                "puzzle": (
                    f"There are {n} houses in a row, numbered 1 to {n} from left to right. "
                    f"Each house has a unique color and is occupied by a different person.\n"
                    f"People: {', '.join(names)}\nColors: {', '.join(colors)}\n\n"
                    f"Clues:\n{clue_text}\n\nDetermine who lives in which colored house."
                ),
                "category": "constraint_satisfaction",
                "difficulty": rng.choice(["easy", "medium", "hard"]),
            }
        )
    return items


def _logic_deduction_puzzles():
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
        rng = random.Random(100 + i)
        tmpl, diff = templates[i % len(templates)]
        names = rng.sample(names_pool, 5)
        puzzle = tmpl.format(a=names[0], b=names[1], c=names[2], d=names[3], e=names[4])
        items.append(
            {
                "id": f"ld_{i + 1:03d}",
                "puzzle": puzzle,
                "category": "logic_deduction",
                "difficulty": diff,
            }
        )
    return items


def generate_task1_data():
    """Generate 60 logic puzzles for Task 1."""
    items = (
        _river_crossing_puzzles() + _constraint_puzzles() + _logic_deduction_puzzles()
    )
    random.Random(42).shuffle(items)
    return items[:60]


# =============================================================================
# TASK 2 DATA: Zoo Planning (70 items)
# =============================================================================


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

    # BFS over state space to find optimal path visiting all animals
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

    # Build labels
    node_labels = {start: "Start", finish: "Finish"}
    for animal, pos in animal_positions.items():
        node_labels[pos] = f"{animal} Cage"
    jc = 0
    for n in nodes:
        if n not in node_labels:
            jc += 1
            node_labels[n] = f"Junction-{jc}"

    # Build text representation
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
    if optimal_path:
        opt_text = " -> ".join(node_labels[n] for n in optimal_path)
        opt_length = len(optimal_path)
    else:
        opt_text, opt_length = "No valid path found", -1

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
    configs = [(4, 24), (6, 24), (8, 22)]
    idx = 0
    for num_animals, count in configs:
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
    return items[:70]


# =============================================================================
# TASK 3 DATA: Game of 24 (80 items)
# =============================================================================


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
    return items[:80]


# =============================================================================
# TASK 4 DATA: Mixed problems (60 items)
# =============================================================================


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

    # Programmatic math problems
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
    return base[:60]


# =============================================================================
# TASK 5 DATA: Tiered problems (80 items)
# =============================================================================


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
    g24 = generate_task3_data()
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
    return items[:80]
