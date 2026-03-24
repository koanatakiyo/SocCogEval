"""Experimental condition factory.

From working_plan_v3.md §4.2:
| # | Name                | Connect.    | Seed      | Interact.  | Reps |
|---|---------------------|-------------|-----------|------------|------|
| 1 | Open+Seed           | All-to-all  | Same      | 30 rounds  | 7    |
| 2 | Segreg+AsymSeed     | Two groups  | Different | 30 rounds  | 8    |
| 3 | Segreg+NoSeed       | Two groups  | None      | 30 rounds  | 5    |
| 4 | SeedOnly+NoInteract | N/A         | Different | None       | 5    |
| 5 | NoSeed+NoInteract   | N/A         | None      | None       | 3    |
"""

from .config import ExperimentConfig, NetworkConfig, SeedConfig, InteractionConfig


def make_condition(
    condition_id: int,
    scenario: str,
    model: str = "",
    use_s2: bool = False,
) -> ExperimentConfig:
    """Create an ExperimentConfig for a given condition number and scenario."""

    if condition_id == 1:
        return ExperimentConfig(
            condition_id=1,
            condition_name="open_seed",
            scenario=scenario,
            n_agents=8,
            network=NetworkConfig(topology="all_to_all", n_groups=1, group_size=8),
            seed=SeedConfig(use_seed=True, asymmetric=False),  # same seed for all
            interaction=InteractionConfig(n_rounds=30, conversations_per_round=4),
            model=model or "",
            n_repeats=7,
            use_s2=use_s2,
        )

    elif condition_id == 2:
        return ExperimentConfig(
            condition_id=2,
            condition_name="segreg_asym_seed",
            scenario=scenario,
            n_agents=8,
            network=NetworkConfig(topology="segregated", n_groups=2, group_size=4),
            seed=SeedConfig(use_seed=True, asymmetric=True),
            interaction=InteractionConfig(n_rounds=30, conversations_per_round=4),
            model=model or "",
            n_repeats=8,
            use_s2=use_s2,
        )

    elif condition_id == 3:
        return ExperimentConfig(
            condition_id=3,
            condition_name="segreg_no_seed",
            scenario=scenario,
            n_agents=8,
            network=NetworkConfig(topology="segregated", n_groups=2, group_size=4),
            seed=SeedConfig(use_seed=False, asymmetric=False),
            interaction=InteractionConfig(n_rounds=30, conversations_per_round=4),
            model=model or "",
            n_repeats=5,
            use_s2=use_s2,
        )

    elif condition_id == 4:
        return ExperimentConfig(
            condition_id=4,
            condition_name="seed_only_no_interact",
            scenario=scenario,
            n_agents=8,
            network=NetworkConfig(topology="none", n_groups=2, group_size=4),
            seed=SeedConfig(use_seed=True, asymmetric=True),
            interaction=InteractionConfig(n_rounds=0, conversations_per_round=0),
            model=model or "",
            n_repeats=5,
            use_s2=False,
        )

    elif condition_id == 5:
        return ExperimentConfig(
            condition_id=5,
            condition_name="no_seed_no_interact",
            scenario=scenario,
            n_agents=8,
            network=NetworkConfig(topology="none", n_groups=1, group_size=8),
            seed=SeedConfig(use_seed=False, asymmetric=False),
            interaction=InteractionConfig(n_rounds=0, conversations_per_round=0),
            model=model or "",
            n_repeats=3,
            use_s2=False,
        )

    else:
        raise ValueError(f"Unknown condition_id: {condition_id}. Must be 1-5.")


def generate_all_configs(model: str = "", output_dir=None):
    """Generate config files for all conditions × both scenarios."""
    from pathlib import Path
    from .config import CONFIGS_DIR

    out = Path(output_dir) if output_dir else CONFIGS_DIR
    out.mkdir(parents=True, exist_ok=True)

    configs = []
    for scenario in ("greenfield", "riverton"):
        for cid in range(1, 6):
            cfg = make_condition(cid, scenario, model=model)
            fname = f"condition_{cid}_{cfg.condition_name}_{scenario}.json"
            cfg.save(out / fname)
            configs.append(cfg)

    return configs
