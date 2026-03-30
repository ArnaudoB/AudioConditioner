from typing import List, Union
import random
import os

characters = [
    "Roman senator",
    "soldier",
    "astronaut",
    "queen",
    "father",
    "general",
    "monk",
    "princess",
    "detective",
    "revolutionary",
    "king",
    "child",
    "explorer",
    "prophet",
    "assassin",
    "thief",
    "artist",
    "sailor",
    "scientist",
    "warrior",
]


characteristics = [
    "lonely",
    "joyful",
    "grieving",
    "desperate",
    "plotting",
    "fearful",
    "hopeful",
    "hostile",
    "confident",
    "nervous",
    "determined",
    "betrayed",
    "vengeful",
    "calm",
    "melancholic",
    "triumphant",
    "conflicted",
    "resigned",
    "driven",
    "haunted",
]


settings = [
    "in a silent marble hall",
    "under a burning sunset",
    "in the middle of a battlefield",
    "on a distant frozen planet",
    "inside a candlelit cathedral",
    "at the edge of a stormy sea",
    "in a deserted city street at night",
    "beneath ancient stone ruins",
    "on a windswept mountain peak",
    "inside a dimly lit throne room",
    "in a crowded marketplace at dawn",
    "under a sky filled with falling stars",
    "inside a crumbling fortress",
    "in a quiet forest clearing",
    "at the gates of a besieged city",
    "inside a small wooden cabin during winter",
    "on the deck of a drifting ship",
    "in a laboratory filled with flickering lights",
    "inside an abandoned theater",
    "at the entrance of a dark cave",
]


emotional_events = [
    "wins the final battle",
    "is crowned before a cheering crowd",
    "returns home in triumph",
    "reunites with a long-lost companion",
    "embraces a newfound purpose",
    "discovers an unexpected opportunity",
    "achieves a lifelong dream",
    "receives long-awaited forgiveness",
    "overcomes a great challenge",
    "begins a promising new journey",
    "saves a kingdom from ruin",
    "stands before an uncertain future",
    "watches a kingdom fall",
    "protects someone at great personal cost",
    "is crowned before a silent crowd",
    "abandons everything to start anew",
    "uncovers a hidden truth",
    "begins a journey with no return",
    "confronts a long-feared enemy",
    "accepts a painful truth",
    "experiences a moment of unexpected hope",
]

def generate_scene(N: int, 
                   seed: Union[int, None] = None, 
                   characters: List[str] = characters,
                   characteristics: List[str] = characteristics,
                   settings: List[str] = settings,
                   emotional_events: List[str] = emotional_events,
                   output_file: Union[str, None] = None) -> List[str]:
    """
    Generates N unique scenes by randomly combining characters, characteristics, settings, and emotional events.

    Parameters:
    - N: The number of unique scenes to generate.
    - seed: An optional random seed for reproducibility.
    - characters: A list of character archetypes to choose from.
    - characteristics: A list of emotional characteristics to choose from.
    - settings: A list of settings to choose from.
    - emotional_events: A list of emotional events to choose from.
    - output_file: An optional file path to save the generated scenes. If None, scenes will not be saved to a file.
    """

    max_combinations = (
        len(characters) *
        len(characteristics) *
        len(settings) *
        len(emotional_events)
    )

    if N > max_combinations:
        raise ValueError("N exceeds total possible unique combinations.")

    if seed is not None:
        random.seed(seed)

    scenes = set()

    while len(scenes) < N:
        character = random.choice(characters)
        characteristic = random.choice(characteristics)
        setting = random.choice(settings)
        event = random.choice(emotional_events)
        scene = f"A {characteristic} {character} {setting} {event}."
        if scene not in scenes:
            scenes.add(scene)

    if output_file:
        print("Saving to:", os.path.abspath(output_file))
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            for scene in scenes:
                f.write(scene + "\n")

    return list(scenes)