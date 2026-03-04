"""Configuration for GHC Benchmark."""

import os
from pathlib import Path

env_file = Path(__file__).parent.parent / ".env"
if env_file.exists():
    for line in env_file.read_text().strip().split('\n'):
        if line and '=' in line:
            key, value = line.split('=', 1)
            os.environ.setdefault(key.strip(), value.strip())

API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

MODELS = [
    "openai/gpt-4o-mini",
    "anthropic/claude-3-haiku",
    "google/gemini-2.0-flash-lite-001",
    "meta-llama/llama-3.1-70b-instruct",
    "meta-llama/llama-3.1-8b-instruct",
    "mistralai/mistral-7b-instruct-v0.1",
    "mistralai/mistral-small-creative",
    "deepseek/deepseek-chat-v3.1",
    "qwen/qwen-2.5-coder-32b-instruct",
    "qwen/qwen3.5-35b-a3b",
    "tencent/hunyuan-a13b-instruct",
]

NUM_QUERIES = 5
NUM_COMPARISONS = 50
SAMPLES_PER_MODEL = 3
NUM_CHUNKS = 5
NUM_LINEARITY_CHUNKS = 10
LINEARITY_JUDGE_MODEL = "openai/gpt-4o-mini"
LINEARITY_THRESHOLD = 70.0

TRANSCRIPT_FILE = "transcript_cleaned.txt"
SCORES_FILE = "results/scores.json"
SAMPLES_FILE = "results/samples.json"
CHARTS_DIR = "results/charts"
REPORTS_DIR = "results/reports"

SYSTEM_PROMPT = """You are a student who genuinely doesn't know anything about quantum computing and is learning about Grover's algorithm for the first time. 

Think out loud as if you're reasoning through this out loud - be raw, unfiltered, confused at times, have false starts, correct yourself, wonder about things. This should sound like authentic human thinking out loud, not polished explanations.

Here's a starting point to continue from:"""

ZERO_SHOT_PROMPT = """You are a student who genuinely doesn't know anything about quantum computing and is learning about Grover's algorithm for the first time. 

Think out loud as if you're reasoning through this out loud - be raw, unfiltered, confused at times, have false starts, correct yourself, wonder about things. This should sound like authentic human thinking out loud, not polished explanations.

Start by explaining what you understand about Grover's algorithm from scratch:"""

LINEARITY_EXAMPLE_NONLINEAR = """So let me start, let me start over here. So, if I have this preposition, right? That I just painted. So let's say 0 0 1 1 0 1 1. And now what I want to have is for example, a, I, I wanna, I wanna find this element, this element, and this element, right? So Grover's algorithm, no listen, it's even, even worse, just these two elements, the Grover's algorithm doesn't work anymore. And that just, I was like, what? I mean, if I keep using my intuition here, the tool it should work, rankest, you're just adding a couple of these, you're adding up and another bunch of this, and then you're, you're doing the the negation here. And so you've got like two amplification effects. But if I, if I start just have the two cubed version here, right? And now I do, so now you've got these, and now I will let's say negate the amplitudes of these two, right? So, and now I applied chorus algorithm amplitude amplification, and we see that really nicely, it just doesn't work. What's more, it's not that it doesn't work, it's just it just leaves us it was exactly the same with exactly the same superposition, no sorry, not this one, with the exact same one after, after facing. So, and this took me to basically start this video, because I want to basically open up. So I want to take a look at, I'm trying to revisit these."""

LINEARITY_EXAMPLE_LINEAR = """Okay, so let's take a step back. I think I'm trying to understand Grover's algorithm here, and it feels kind of like a maze. So, um, I get that it's a quantum algorithm used for search problems, but what does that even really mean? Like, it's supposed to speed up the search for a specific item in an unsorted database. So if I have a list of things, Grover's algorithm helps me find something among them, quicker than classical methods? Wait, how quicker again? Is it like exponentially faster? No, I think it's quadratic. Yeah, that sounds right, quadratic speedup.

Okay, let me see if I can piece together what I read. So, you start with this superposition of all possible states, right? When you begin, all those possible 'answers' are represented in some sort of state, maybe like all the combinations of bits or something. And then there's this 'oracle' that tells you whether you've found the right one. But how exactly does that work? Is it like a magical black box that just knows? It feels like it gives you information without revealing how it does that.

And then, there's the part about the amplitude amplification. From what I gathered, it's about increasing the probability of the correct answer, which kind of makes sense because if I want to find that one item, I need to make sure that gets more likely to be detected, right? So you apply this transformation that flips the sign of the amplitude for the correct answer, I think? And that it's repeated many times for this amplification effect to 'work.' Wait, how many times do you have to repeat? Is it like a specific number based on the size of the list?"""

LINEARITY_JUDGE_PROMPT = """You are evaluating how LINEAR a text's thinking progression is. Linearity means the thinking flows naturally from one idea to the next, building coherently without jumps, tangents, or mid-topic switches.

Here's a NON-LINEAR example (score ~10-25) - notice the jumps, tangents, and self-corrections:
{NONLINEAR}

Here's a LINEAR example (score ~80-95) - notice the clear sequential progression:
{LINEAR}

Now evaluate this text transition:

--- CHUNK A ---
{CHUNK_A}

--- CHUNK B ---
{CHUNK_B}

Rate how LINEAR the thinking flows from Chunk A to Chunk B on a scale of 0-100.
- 0 = Completely non-linear: major topic jump, tangent, mid-thought switch, unrelated content
- 50 = Somewhat linear: some connection but choppy flow
- 100 = Perfectly linear: each idea clearly follows from the previous one

Respond with just the numeric score (0-100) and a brief 1-sentence justification."""
