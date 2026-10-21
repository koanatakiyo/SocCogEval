#!/usr/bin/env python3
"""Run one frozen C6 matched solo-generation replica."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from datetime import datetime
from pathlib import Path

from cogclaw.agent import AGENT_NAMES, Agent
from cogclaw.config import DATA_DIR, PRIMARY_MODEL
from cogclaw.solo import (
    ExposureRecorder,
    SoloOrchestrator,
    TokenCounter,
    build_serving_metadata,
    make_c6_config,
    validate_c6_serving_metadata,
)
from scenarios import get_scenario


def setup_logging(label: str) -> tuple[logging.Logger, Path]:
    log_dir = DATA_DIR / "experiment_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{label}_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
        force=True,
    )
    return logging.getLogger("cogclaw.c6"), log_file


def create_c6_agents(config) -> list[Agent]:
    """Create controlled clones with C2/C4 asymmetric A/B briefs."""
    scenario = get_scenario(config.scenario)
    agents = []
    group_size = config.n_agents // 2
    for index in range(config.n_agents):
        agents.append(
            Agent(
                agent_id=index,
                name=AGENT_NAMES[index],
                scenario_context=scenario.CONTEXT,
                model=config.model,
                seed_info=(
                    scenario.SEED_A if index < group_size else scenario.SEED_B
                ),
            )
        )
    return agents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen C6 matched solo-generation control"
    )
    parser.add_argument(
        "--scenario",
        required=True,
        choices=["greenfield", "riverton"],
    )
    parser.add_argument("--replica", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--model", default=PRIMARY_MODEL)
    parser.add_argument(
        "--tokenizer-path",
        required=True,
        type=Path,
        help="Exact local checkpoint/tokenizer directory used by vLLM",
    )
    parser.add_argument(
        "--output-dir",
        default=DATA_DIR,
        type=Path,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Barriered concurrent requests per stage (amended protocol: 8)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Allow a shortened non-production run",
    )
    parser.add_argument(
        "--n-rounds",
        type=int,
        default=None,
        help="Only accepted with --smoke; production is frozen at 30",
    )
    args = parser.parse_args()

    if args.replica < 0:
        parser.error("--replica must be non-negative")
    if args.batch_size != 8:
        parser.error("amended C6 protocol requires --batch-size 8")
    if args.n_rounds is not None and not args.smoke:
        parser.error("--n-rounds is only allowed with --smoke")
    if args.smoke and args.n_rounds is None:
        args.n_rounds = 1
    if not args.smoke:
        args.n_rounds = 30
    if args.n_rounds < 1:
        parser.error("--n-rounds must be positive")
    if not args.tokenizer_path.is_dir():
        parser.error(f"tokenizer directory not found: {args.tokenizer_path}")
    return args


def main() -> None:
    args = parse_args()
    run_id = args.run_id or f"c6rep{args.replica:02d}"
    if args.smoke and not run_id.startswith("smoke_"):
        run_id = f"smoke_{run_id}"

    logger, log_file = setup_logging(
        f"c6_{args.scenario}_rep{args.replica:02d}"
    )
    random.seed(args.seed)
    config = make_c6_config(
        scenario=args.scenario,
        model=args.model,
        n_rounds=args.n_rounds,
    )
    config.run_id = run_id

    logger.info(
        "C6 start: scenario=%s replica=%02d rounds=%d model=%s seed=%d "
        "smoke=%s batch_size=%d",
        args.scenario,
        args.replica,
        args.n_rounds,
        args.model,
        args.seed,
        args.smoke,
        args.batch_size,
    )
    token_counter = TokenCounter(args.tokenizer_path)
    # setup_run_dir redirects this placeholder into the timestamped directory.
    recorder = ExposureRecorder(
        args.output_dir / ".c6_exposure_placeholder.jsonl",
        token_counter,
        run_id=run_id,
        scenario=args.scenario,
        replica=args.replica,
        model=args.model,
        python_seed=args.seed,
    )
    agents = create_c6_agents(config)
    orchestrator = SoloOrchestrator(
        config,
        agents,
        recorder,
        replica=args.replica,
        python_seed=args.seed,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )
    run_dir = orchestrator.setup_run_dir()

    metadata = build_serving_metadata(
        tokenizer_path=args.tokenizer_path,
        model=args.model,
        scenario=args.scenario,
        replica=args.replica,
        python_seed=args.seed,
        batch_size=args.batch_size,
        argv=sys.argv,
    )
    metadata["smoke"] = args.smoke
    serving_errors = validate_c6_serving_metadata(metadata)
    metadata["serving_validation"] = {
        "status": "failed" if serving_errors else "passed",
        "errors": serving_errors,
    }
    metadata["n_rounds"] = args.n_rounds
    recorder.set_serving_context(metadata)
    with (run_dir / "serving_metadata.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    if serving_errors:
        joined_errors = "; ".join(serving_errors)
        logger.error("C6 serving validation failed: %s", joined_errors)
        raise RuntimeError(
            f"Refusing to start C6 with non-frozen serving config: {joined_errors}"
        )

    t0 = time.time()
    orchestrator.run_experiment(probe_interval=5)
    logger.info(
        "C6 complete in %.1f minutes: %s (log: %s)",
        (time.time() - t0) / 60,
        run_dir,
        log_file,
    )


if __name__ == "__main__":
    main()
