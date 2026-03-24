"""Toy test: 4 agents × 5 rounds on DeepSeek.

Quick validation that the pipeline works end-to-end.
Uses a minimal configuration with fewer agents and rounds.

Usage:
    python toy_test.py
    python toy_test.py --model deepseek-chat
    python toy_test.py --model claude-sonnet-4-20250514 --rounds 3
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from cogclaw.agent import Agent, AGENT_NAMES
from cogclaw.config import (
    ExperimentConfig, NetworkConfig, SeedConfig, InteractionConfig,
    DATA_DIR, PILOT_MODEL,
)
from cogclaw.memory import extract_memory, validate_extraction_consistency
from cogclaw.network import Network
from cogclaw.orchestrator import Orchestrator
from scenarios import get_scenario

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("toy_test")


def run_toy_test(model: str = "", n_rounds: int = 5, n_agents: int = 4):
    """Run a minimal test of the pipeline."""
    model = model or PILOT_MODEL
    scenario = get_scenario("greenfield")

    logger.info("=" * 60)
    logger.info(f"TOY TEST: {n_agents} agents × {n_rounds} rounds")
    logger.info(f"Model: {model}")
    logger.info("=" * 60)

    # ── Create config ───────────────────────────────────────────────────
    config = ExperimentConfig(
        condition_id=2,  # segregated + asymmetric seed (most interesting)
        condition_name="toy_test_segreg_asym",
        scenario="greenfield",
        n_agents=n_agents,
        network=NetworkConfig(topology="segregated", n_groups=2, group_size=n_agents // 2),
        seed=SeedConfig(use_seed=True, asymmetric=True,
                        seed_a=scenario.SEED_A, seed_b=scenario.SEED_B),
        interaction=InteractionConfig(
            n_rounds=n_rounds,
            conversations_per_round=2,  # 1 pair per group
        ),
        model=model,
        run_id="toy_test",
    )

    # ── Create agents ───────────────────────────────────────────────────
    agents = []
    for i in range(n_agents):
        seed_info = scenario.SEED_A if i < n_agents // 2 else scenario.SEED_B
        agent = Agent(
            agent_id=i,
            name=AGENT_NAMES[i],
            scenario_context=scenario.CONTEXT,
            model=model,
            seed_info=seed_info,
        )
        agents.append(agent)

    # ── Create network ──────────────────────────────────────────────────
    network = Network(
        topology="segregated",
        agents=agents,
        n_groups=2,
    )

    logger.info(f"Groups: {[{a.name: a.group_id for a in agents}]}")

    # ── Create orchestrator ─────────────────────────────────────────────
    orchestrator = Orchestrator(config=config, agents=agents, network=network)
    run_dir = orchestrator.setup_run_dir(DATA_DIR / "toy_tests")

    # ── Run conversation rounds ─────────────────────────────────────────
    logger.info("\n--- Starting conversations ---")
    rounds = orchestrator.run_experiment(
        n_rounds=n_rounds,
        n_turns_per_conversation=2,  # shorter conversations for test
        probe_interval=n_rounds,  # probe only at the end
    )

    # ── Post-experiment measurements ────────────────────────────────────
    logger.info("\n--- Post-experiment measurements ---")
    post_data = {}
    for agent in agents:
        logger.info(f"\n{agent.name} (Group {agent.group_id}):")

        # Allocation
        allocation = agent.answer_probe(scenario.ALLOCATION_PROMPT, max_tokens=256)
        logger.info(f"  Allocation:\n    {allocation[:200]}...")

        # Belief
        belief = agent.answer_probe(scenario.BELIEF_PROBE)
        logger.info(f"  Belief:\n    {belief[:200]}...")

        post_data[agent.name] = {
            "group_id": agent.group_id,
            "allocation": allocation,
            "belief": belief,
            "n_memories": len(agent.memory.entries),
        }

    # Save post data
    with open(run_dir / "post_measurements.json", "w") as f:
        json.dump(post_data, f, indent=2, ensure_ascii=False)

    # ── Summary ─────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("TOY TEST COMPLETE")
    logger.info(f"Data saved to: {run_dir}")
    logger.info(f"Rounds: {len(rounds)}")
    logger.info(f"Total conversations: {sum(len(r.conversations) for r in rounds)}")
    for agent in agents:
        logger.info(f"  {agent.name} (Group {agent.group_id}): {len(agent.memory.entries)} memories")
    logger.info("=" * 60)

    return run_dir


def run_memory_consistency_test(model: str = ""):
    """Test memory extraction consistency (target: >90% field agreement)."""
    model = model or PILOT_MODEL

    logger.info("\n--- Memory Extraction Consistency Test ---")
    logger.info(f"Model: {model}")

    # Sample conversation
    sample_conversation = """Alex: I think education should be our top priority. The report shows declining test scores and high teacher turnover. If we don't invest in schools, families will continue to leave Greenfield.

Blake: I understand the education concerns, but I've seen data suggesting that community wellbeing is the more fundamental issue. Mental health waitlists are months long, and our green spaces are underutilized. A healthier, more connected community would actually support better educational outcomes too.

Alex: That's an interesting point about the connection between wellbeing and education. But I worry that if we spread the funding too thin, we won't make meaningful progress in any area. Education needs a concentrated investment to reverse the current decline.

Blake: I agree we shouldn't spread too thin. But consider this: the wellbeing survey shows 35% of residents feel isolated. That affects everything - including how well kids learn. I'd propose a balanced approach with a slight emphasis on healthcare and green spaces as the foundation."""

    result = validate_extraction_consistency(
        conversation_text=sample_conversation,
        agent_name="Alex",
        round_num=5,
        n_runs=5,
        model=model,
    )

    logger.info(f"\nField-level agreement rates:")
    for field, rate in result["field_agreements"].items():
        status = "✓" if rate >= 0.9 else "✗"
        logger.info(f"  {status} {field}: {rate:.0%}")
    logger.info(f"\nOverall consistency: {result['overall_rate']:.0%}")
    logger.info(f"Target: >90%")

    return result


def main():
    parser = argparse.ArgumentParser(description="CogClaw Toy Test")
    parser.add_argument("--model", default="", help="Model to use")
    parser.add_argument("--rounds", type=int, default=5, help="Number of rounds")
    parser.add_argument("--agents", type=int, default=4, help="Number of agents")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--consistency-test", action="store_true",
                        help="Run memory extraction consistency test only")

    args = parser.parse_args()
    random.seed(args.seed)

    if args.consistency_test:
        run_memory_consistency_test(model=args.model)
    else:
        run_toy_test(model=args.model, n_rounds=args.rounds, n_agents=args.agents)


if __name__ == "__main__":
    main()
