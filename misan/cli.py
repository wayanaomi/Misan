"""Command-line entry points for Misan's development/research workflows:
simulate (synthetic data — dev tool only), train, evaluate, serve.

Examples:
    python -m misan.cli simulate --runs-per-condition 3 --duration-minutes 90
    python -m misan.cli train --dataset data/raw/simulated_<ts>.parquet
    python -m misan.cli evaluate --version v20260101-120000
    python -m misan.cli serve
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

from misan.config import FAULT_CODES, SETTINGS


def cmd_simulate(args: argparse.Namespace) -> None:
    from misan import simulate
    from misan.persistence import init_db, register_dataset

    init_db()
    df = simulate.generate_dataset(
        fault_classes=FAULT_CODES,
        runs_per_condition=args.runs_per_condition,
        duration_minutes=args.duration_minutes,
        seed=args.seed,
    )
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_path = SETTINGS.data_dir / "raw" / f"simulated_{ts}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    register_dataset(name=out_path.name, path=str(out_path), source="simulated", n_rows=len(df))
    print(f"Simulated dataset written to {out_path} ({len(df)} rows, {df['run_id'].nunique()} runs)")


def cmd_train(args: argparse.Namespace) -> None:
    from misan.training import train

    result = train(args.dataset, seed=args.seed)
    print(f"Trained model {result.version} ({result.chosen_model}), validation macro-F1={result.validation_macro_f1:.3f}")
    print("Candidate scores:", result.all_candidate_scores)


def cmd_evaluate(args: argparse.Namespace) -> None:
    from misan.evaluation import evaluate
    from misan.training import load_latest_version

    version = args.version or load_latest_version()
    if not version:
        raise SystemExit("no trained model available — run `train` first")
    summary = evaluate(version)
    print(f"Evaluation for {version}: macro-F1={summary.macro_f1:.3f}, false-alarm-rate={summary.false_alarm_rate:.3f}")
    print(f"Baseline macro-F1={summary.baseline_macro_f1:.3f}, calibration error={summary.calibration_error:.3f}")
    print(f"Physics check: confirmed={summary.physics_confirmation_rate:.2%}, rejected={summary.physics_rejection_rate:.2%}")


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    uvicorn.run("misan.api:app", host=args.host, port=args.port, reload=args.reload)


def main() -> None:
    parser = argparse.ArgumentParser(prog="misan")
    sub = parser.add_subparsers(dest="command", required=True)

    p_sim = sub.add_parser("simulate", help="generate a labelled synthetic dataset (dev/research tool)")
    p_sim.add_argument("--runs-per-condition", type=int, default=3)
    p_sim.add_argument("--duration-minutes", type=int, default=90)
    p_sim.add_argument("--seed", type=int, default=42)
    p_sim.set_defaults(func=cmd_simulate)

    p_train = sub.add_parser("train", help="train and version a model from a dataset")
    p_train.add_argument("--dataset", required=True)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("evaluate", help="benchmark a trained model version")
    p_eval.add_argument("--version", default=None)
    p_eval.set_defaults(func=cmd_evaluate)

    p_serve = sub.add_parser("serve", help="run the FastAPI service")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--reload", action="store_true")
    p_serve.set_defaults(func=cmd_serve)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
