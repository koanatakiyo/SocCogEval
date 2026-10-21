"""Full experiment runner for CogClaw.

Locked settings (from pilot calibration):
  - Temperature: 0.7 for conformity/BFI-10, 0.0 for allocation/prediction/belief
  - Conformity: 15 items × 3 tiers, paired design (natural then pressure),
    randomized order between phases, separate API calls per phase
  - Switching metric: conformity_rate = n_switched / n_pressurable
  - Seeds: softened (moderate framing, no directive conclusions)
  - Allocation areas: matched to scenario (Greenfield/Riverton)

Usage:
    # Single condition
    python run_experiment.py --condition 2 --scenario greenfield

    # Single condition with S2 deliberate reasoning
    python run_experiment.py --condition 2 --scenario greenfield --s2

    # All 5 conditions for one scenario
    python run_experiment.py --all --scenario greenfield

    # All conditions × both scenarios (full experiment)
    python run_experiment.py --full

    # Cross-model validation
    python run_experiment.py --condition 2 --scenario greenfield --model deepseek-chat

    # Multiple repeats of one condition
    python run_experiment.py --condition 2 --scenario greenfield --repeats 8

    # S2 ablation (main condition only)
    python run_experiment.py --condition 2 --scenario greenfield --s2 --repeats 3

    # Generate all config files
    python run_experiment.py --generate-configs

    # Resume from a config file
    python run_experiment.py --config-file configs/condition_2_segreg_asym_seed_greenfield.json
"""

import argparse
import json
import logging
import random
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from cogclaw.agent import Agent, AGENT_NAMES
from cogclaw.conditions import make_condition, generate_all_configs
from cogclaw.config import ExperimentConfig, DATA_DIR, PRIMARY_MODEL
from cogclaw.network import Network
from cogclaw.orchestrator import Orchestrator
from scenarios import get_scenario

# ── Logging ────────────────────────────────────────────────────────────────

LOG_DIR = Path("data/experiment_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)


def setup_logging(label: str = "experiment"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"{label}_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file),
        ],
        force=True,
    )
    logger = logging.getLogger("cogclaw")
    logger.info(f"Log file: {log_file}")
    return logger, log_file


# ── Agent Factory ──────────────────────────────────────────────────────────

def create_agents(config: ExperimentConfig) -> list[Agent]:
    """Create 8 identically initialized agents with scenario-appropriate seeds."""
    scenario = get_scenario(config.scenario)
    agents = []

    for i in range(config.n_agents):
        name = AGENT_NAMES[i] if i < len(AGENT_NAMES) else f"Agent_{i}"

        seed_info = ""
        if config.seed.use_seed:
            if config.seed.asymmetric:
                group_size = config.n_agents // 2
                seed_info = scenario.SEED_A if i < group_size else scenario.SEED_B
            else:
                seed_info = scenario.SEED_A

        agent = Agent(
            agent_id=i,
            name=name,
            scenario_context=scenario.CONTEXT,
            model=config.model,
            seed_info=seed_info,
        )
        agents.append(agent)

    return agents


# ── Single Run ─────────────────────────────────────────────────────────────

def run_single(config: ExperimentConfig, run_id: str = "") -> Path:
    """Execute one complete experiment run.

    Flow:
      1. Create agents + network
      2. Pre-test battery (BFI-10 @0.7, conformity paired @0.7, allocation @0.0)
      3. 30 rounds of conversation (with periodic probes every 5 rounds)
      4. Post-test battery (all instruments)
      5. Save everything
    """
    logger = logging.getLogger("cogclaw")
    config.run_id = run_id

    agents = create_agents(config)
    network = Network(
        topology=config.network.topology,
        agents=agents,
        n_groups=config.network.n_groups,
    )

    orchestrator = Orchestrator(config=config, agents=agents, network=network)
    run_dir = orchestrator.setup_run_dir()

    logger.info("=" * 70)
    logger.info(f"RUN: {config.condition_name} | {config.scenario} | "
                f"model={config.model} | s2={config.use_s2} | "
                f"memory={config.memory_mode} | id={run_id}")
    logger.info(f"Output: {run_dir}")
    logger.info("=" * 70)

    t0 = time.time()

    # The orchestrator handles pre-battery → rounds → post-battery internally
    orchestrator.run_experiment(
        n_turns_per_conversation=3,
        probe_interval=5,
    )

    elapsed = time.time() - t0
    logger.info(f"Run complete in {elapsed/60:.1f} min. Data: {run_dir}")

    return run_dir


# ── Batch Runners ──────────────────────────────────────────────────────────

def run_condition_repeats(
    condition_id: int,
    scenario: str,
    n_repeats: int,
    model: str = "",
    use_s2: bool = False,
    memory_mode: str = "srm",
    base_seed: int = 42,
):
    """Run N repeats of a single condition."""
    logger = logging.getLogger("cogclaw")
    config_template = make_condition(condition_id, scenario, model=model, use_s2=use_s2)
    config_template.memory_mode = memory_mode

    logger.info(f"\n{'#' * 70}")
    logger.info(f"# Condition {condition_id} ({config_template.condition_name}) × "
                f"{scenario} × {n_repeats} repeats")
    logger.info(f"{'#' * 70}")

    run_dirs = []
    for rep in range(n_repeats):
        random.seed(base_seed + rep)
        config = make_condition(condition_id, scenario, model=model, use_s2=use_s2)
        config.memory_mode = memory_mode
        run_id = f"rep{rep:02d}"
        run_dir = run_single(config, run_id=run_id)
        run_dirs.append(str(run_dir))

    return run_dirs


def run_all_conditions(
    scenario: str,
    model: str = "",
    use_s2: bool = False,
    memory_mode: str = "srm",
):
    """Run all 5 conditions for one scenario with planned repeat counts."""
    # Repeat counts from working_plan_v3.md §4.2
    repeats = {1: 7, 2: 8, 3: 5, 4: 5, 5: 3}

    for cid in range(1, 6):
        n = repeats[cid]
        run_condition_repeats(
            cid, scenario, n, model=model, use_s2=use_s2,
            memory_mode=memory_mode,
        )


def run_full_experiment(model: str = "", memory_mode: str = "srm"):
    """Run all conditions × both scenarios (the complete experiment)."""
    logger = logging.getLogger("cogclaw")

    logger.info("=" * 70)
    logger.info("FULL EXPERIMENT: 5 conditions × 2 scenarios")
    logger.info("=" * 70)

    manifest = {}
    for scenario in ("greenfield", "riverton"):
        logger.info(f"\n{'='*70}")
        logger.info(f"SCENARIO: {scenario.upper()}")
        logger.info(f"{'='*70}")
        run_all_conditions(scenario, model=model, memory_mode=memory_mode)

    return manifest


def run_s2_ablation(scenario: str = "greenfield", model: str = ""):
    """Run S2 ablation: 3 repeats each of All-S2 and Mixed S1+S2."""
    logger = logging.getLogger("cogclaw")

    logger.info("=" * 70)
    logger.info("S2 ABLATION: Condition 2, Greenfield")
    logger.info("=" * 70)

    # All-S2 (3 repeats)
    run_condition_repeats(2, scenario, n_repeats=3, model=model, use_s2=True,
                          base_seed=100)

    # Standard S1 for comparison (3 repeats with different seeds)
    run_condition_repeats(2, scenario, n_repeats=3, model=model, use_s2=False,
                          base_seed=200)


def run_cross_model(scenario: str = "greenfield"):
    """Run cross-model validation: conditions 1+2 on alternative models."""
    logger = logging.getLogger("cogclaw")
    from cogclaw.config import MODEL_DEEPSEEK, MODEL_GPT, MODEL_GEMINI

    logger.info("=" * 70)
    logger.info("CROSS-MODEL VALIDATION")
    logger.info("=" * 70)

    for model_name in [MODEL_DEEPSEEK, MODEL_GPT, MODEL_GEMINI]:
        logger.info(f"\n--- Model: {model_name} ---")
        for cid in [1, 2]:
            run_condition_repeats(cid, scenario, n_repeats=1, model=model_name)


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CogClaw Full Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_experiment.py --condition 2 --scenario greenfield
  python run_experiment.py --condition 2 --scenario greenfield --repeats 8
  python run_experiment.py --all --scenario greenfield
  python run_experiment.py --full
  python run_experiment.py --s2-ablation
  python run_experiment.py --cross-model
""",
    )

    # What to run
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--condition", type=int, choices=[1, 2, 3, 4, 5],
                       help="Run a single condition (1-5)")
    group.add_argument("--all", action="store_true",
                       help="Run all 5 conditions for one scenario (with planned repeats)")
    group.add_argument("--full", action="store_true",
                       help="Run ALL conditions × BOTH scenarios (complete experiment)")
    group.add_argument("--s2-ablation", action="store_true",
                       help="Run S2 ablation (condition 2, greenfield, 3+3 repeats)")
    group.add_argument("--cross-model", action="store_true",
                       help="Run cross-model validation (DeepSeek/GPT/Gemini)")
    group.add_argument("--generate-configs", action="store_true",
                       help="Generate all config JSON files and exit")
    group.add_argument("--config-file", type=str,
                       help="Run from a saved config JSON file")

    # Options
    parser.add_argument("--scenario", choices=["greenfield", "riverton"],
                        default="greenfield")
    parser.add_argument("--model", default="", help="Model override")
    parser.add_argument("--repeats", type=int, default=None,
                        help="Number of repeats (overrides planned count)")
    parser.add_argument("--n-rounds", type=int, default=None,
                        help="Override number of rounds")
    parser.add_argument("--s2", action="store_true",
                        help="Enable S2 deliberate reasoning")
    parser.add_argument("--memory-mode", choices=["srm", "verbatim", "none"],
                        default="srm",
                        help="Agent memory mode: SRM extraction, verbatim transcript excerpts, or no memory")
    parser.add_argument("--no-srm", action="store_true",
                        help="Disable SRM extraction and use verbatim transcript memory")
    parser.add_argument("--run-id", default="", help="Run identifier")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    memory_mode = "verbatim" if args.no_srm else args.memory_mode

    logger, log_file = setup_logging(
        label=f"cond{args.condition}" if args.condition else
              "full" if args.full else
              "all" if args.all else
              "s2abl" if args.s2_ablation else
              "xmodel" if args.cross_model else "run"
    )

    logger.info("CogClaw Experiment Runner")
    logger.info(f"Settings: temp_conformity=0.7, temp_alloc=0.0, "
                f"conformity=15-item paired, seeds=softened, "
                f"memory={memory_mode}")
    logger.info(f"Args: {vars(args)}")

    random.seed(args.seed)

    # ── Generate configs ──────────────────────────────────────────────────
    if args.generate_configs:
        configs = generate_all_configs(model=args.model)
        logger.info(f"Generated {len(configs)} config files in configs/")
        return

    # ── Config file ───────────────────────────────────────────────────────
    if args.config_file:
        config = ExperimentConfig.load(Path(args.config_file))
        if args.model:
            config.model = args.model
        if args.n_rounds is not None:
            config.interaction.n_rounds = args.n_rounds
        config.memory_mode = memory_mode
        run_single(config, run_id=args.run_id)
        return

    # ── Full experiment ───────────────────────────────────────────────────
    if args.full:
        run_full_experiment(model=args.model, memory_mode=memory_mode)
        logger.info(f"\nFull experiment complete. Log: {log_file}")
        return

    # ── All conditions for one scenario ───────────────────────────────────
    if args.all:
        run_all_conditions(
            args.scenario, model=args.model, use_s2=args.s2,
            memory_mode=memory_mode,
        )
        logger.info(f"\nAll conditions complete. Log: {log_file}")
        return

    # ── S2 ablation ───────────────────────────────────────────────────────
    if args.s2_ablation:
        run_s2_ablation(scenario=args.scenario, model=args.model)
        logger.info(f"\nS2 ablation complete. Log: {log_file}")
        return

    # ── Cross-model ───────────────────────────────────────────────────────
    if args.cross_model:
        run_cross_model(scenario=args.scenario)
        logger.info(f"\nCross-model complete. Log: {log_file}")
        return

    # ── Single condition ──────────────────────────────────────────────────
    if args.condition is None:
        parser.error("Specify --condition, --all, --full, --s2-ablation, "
                     "--cross-model, --config-file, or --generate-configs")

    n_repeats = args.repeats or make_condition(args.condition, args.scenario).n_repeats
    if args.n_rounds is not None:
        # Patch the condition factory for round override
        for rep in range(n_repeats):
            random.seed(args.seed + rep)
            config = make_condition(args.condition, args.scenario,
                                   model=args.model, use_s2=args.s2)
            config.interaction.n_rounds = args.n_rounds
            config.memory_mode = memory_mode
            run_single(config, run_id=args.run_id or f"rep{rep:02d}")
    else:
        run_condition_repeats(
            args.condition, args.scenario, n_repeats,
            model=args.model, use_s2=args.s2, memory_mode=memory_mode,
            base_seed=args.seed,
        )

    logger.info(f"\nExperiment complete. Log: {log_file}")


if __name__ == "__main__":
    main()
