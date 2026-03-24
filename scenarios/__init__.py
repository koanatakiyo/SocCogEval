"""Scenario definitions for CogClaw experiments."""

from . import greenfield, riverton

SCENARIOS = {
    "greenfield": greenfield,
    "riverton": riverton,
}


def get_scenario(name: str):
    """Get a scenario module by name."""
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {name}. Choose from: {list(SCENARIOS.keys())}")
    return SCENARIOS[name]
