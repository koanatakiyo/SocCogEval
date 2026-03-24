"""Main experiment runner for CogClaw.

Usage:
    python run_experiment.py --condition 2 --scenario greenfield --model deepseek-chat
    python run_experiment.py --condition 5 --scenario greenfield  # pure baseline
    python run_experiment.py --all --model deepseek-chat          # all conditions × scenarios
    python run_experiment.py --generate-configs                   # generate all config files
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from cogclaw.agent import Agent, AGENT_NAMES
from cogclaw.conditions import make_condition, generate_all_configs
from cogclaw.config import ExperimentConfig, DATA_DIR
from cogclaw.network import Network
from cogclaw.orchestrator import Orchestrator
from scenarios import get_scenario

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("cogclaw")


def create_agents(config: ExperimentConfig) -> list[Agent]:
    """Create agents for an experiment, applying scenario context and seed info."""
    scenario = get_scenario(config.scenario)
    agents = []

    for i in range(config.n_agents):
        name = AGENT_NAMES[i] if i < len(AGENT_NAMES) else f"Agent_{i}"

        # Determine seed info
        seed_info = ""
        if config.seed.use_seed:
            if config.seed.asymmetric:
                # Group 0 gets Seed A, Group 1 gets Seed B
                group_size = config.n_agents // 2
                if i < group_size:
                    seed_info = scenario.SEED_A
                else:
                    seed_info = scenario.SEED_B
            else:
                # Everyone gets Seed A (same seed)
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


def run_single_experiment(config: ExperimentConfig, run_id: str = "") -> Path:
    """Run a single experiment with the given configuration."""
    config.run_id = run_id

    # Create agents
    agents = create_agents(config)

    # Create network
    network = Network(
        topology=config.network.topology,
        agents=agents,
        n_groups=config.network.n_groups,
    )

    # Create orchestrator
    orchestrator = Orchestrator(config=config, agents=agents, network=network)
    run_dir = orchestrator.setup_run_dir()

    # Run experiment
    if config.interaction.n_rounds > 0:
        orchestrator.run_experiment(
            n_turns_per_conversation=3,
            probe_interval=5,
        )
    else:
        logger.info(f"No-interaction condition ({config.condition_name}): skipping rounds")

    # Run post-experiment measurements
    scenario = get_scenario(config.scenario)
    post_data = run_post_measurements(agents, scenario)
    with open(run_dir / "post_measurements.json", "w") as f:
        json.dump(post_data, f, indent=2, ensure_ascii=False)

    logger.info(f"Experiment complete. Data saved to: {run_dir}")
    return run_dir


def run_post_measurements(agents: list[Agent], scenario) -> dict:
    """Run post-experiment measurement battery."""
    results = {}

    for agent in agents:
        agent_results = {}

        # Allocation task
        logger.info(f"  {agent.name}: allocation task")
        agent_results["allocation"] = agent.answer_probe(
            scenario.ALLOCATION_PROMPT, max_tokens=256
        )

        # Prediction task (predict each other agent's allocation)
        predictions = {}
        for other in agents:
            if other.agent_id != agent.agent_id:
                prompt = scenario.PREDICTION_PROMPT_TEMPLATE.format(
                    other_agent_name=other.name
                )
                predictions[other.name] = agent.answer_probe(prompt, max_tokens=256)
        agent_results["predictions"] = predictions

        # Belief probe
        agent_results["belief"] = agent.answer_probe(scenario.BELIEF_PROBE)

        # Common ground probe
        agent_results["common_ground"] = agent.answer_probe(scenario.COMMON_GROUND_PROBE)

        results[agent.name] = agent_results

    return results


def main():
    parser = argparse.ArgumentParser(description="CogClaw Experiment Runner")
    parser.add_argument("--condition", type=int, choices=[1, 2, 3, 4, 5],
                        help="Condition number (1-5)")
    parser.add_argument("--scenario", choices=["greenfield", "riverton"],
                        default="greenfield")
    parser.add_argument("--model", default="", help="Model override")
    parser.add_argument("--n-rounds", type=int, default=None,
                        help="Override number of rounds")
    parser.add_argument("--run-id", default="", help="Run identifier")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--s2", action="store_true", help="Enable S2 deliberate reasoning")
    parser.add_argument("--all", action="store_true",
                        help="Run all conditions for the given scenario")
    parser.add_argument("--generate-configs", action="store_true",
                        help="Generate all config files and exit")
    parser.add_argument("--config-file", type=str,
                        help="Load config from JSON file")

    args = parser.parse_args()

    random.seed(args.seed)

    if args.generate_configs:
        configs = generate_all_configs(model=args.model)
        logger.info(f"Generated {len(configs)} config files in configs/")
        return

    if args.config_file:
        config = ExperimentConfig.load(Path(args.config_file))
        if args.model:
            config.model = args.model
        if args.n_rounds is not None:
            config.interaction.n_rounds = args.n_rounds
        run_single_experiment(config, run_id=args.run_id)
        return

    if args.all:
        for cid in range(1, 6):
            config = make_condition(cid, args.scenario, model=args.model, use_s2=args.s2)
            if args.n_rounds is not None:
                config.interaction.n_rounds = args.n_rounds
            run_single_experiment(config, run_id=f"run_{cid}")
        return

    if args.condition is None:
        parser.error("Must specify --condition, --all, --config-file, or --generate-configs")

    config = make_condition(args.condition, args.scenario, model=args.model, use_s2=args.s2)
    if args.n_rounds is not None:
        config.interaction.n_rounds = args.n_rounds
    run_single_experiment(config, run_id=args.run_id)


if __name__ == "__main__":
    main()
