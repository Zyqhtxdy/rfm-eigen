from __future__ import annotations

import argparse
import json
from pathlib import Path

from paper_assets.paths import EXPERIMENT4_CONFIGS

from .config import RunConfig
from .paper_assets import generate_experiment4_assets
from .plotting import plot_baseline_curves
from .reference import save_reference, solve_reference
from .rfm import audit_rfm_results, run_configured_rfm, search_configured_rfm
from .trainer import RESULTS, train_baseline


def command_reference(args: argparse.Namespace) -> None:
    solution = solve_reference(args.subdiv)
    output = RESULTS / f"reference_subdiv{args.subdiv}.npz"
    save_reference(solution, output)
    print(json.dumps({"path": str(output), "subdiv": args.subdiv, "keff": solution.keff}, indent=2))


def command_baseline(args: argparse.Namespace) -> None:
    config = RunConfig.from_toml(args.config)
    summary = train_baseline(
        config,
        epoch_override=args.epochs,
        output_directory=args.output_directory,
        resume=args.resume,
    )
    print(json.dumps(summary, indent=2))


def command_baseline_suite(args: argparse.Namespace) -> None:
    summaries = []
    config_paths = args.config or [
        str(EXPERIMENT4_CONFIGS / "drm.toml"),
        str(EXPERIMENT4_CONFIGS / "gipmnn.toml"),
        str(EXPERIMENT4_CONFIGS / "pc_gipmnn.toml"),
    ]
    for config_path in config_paths:
        config = RunConfig.from_toml(config_path)
        output_directory = Path(args.output_root) / config.result_name
        summary = train_baseline(config, output_directory=output_directory)
        summaries.append(summary)
        print(json.dumps(summary, indent=2), flush=True)
    output = Path(args.output_root) / "baseline_suite.json"
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(summaries, handle, indent=2)
    print(output)


def command_audit(_: argparse.Namespace) -> None:
    directories = (
        "drm_seed2026071101",
        "gipmnn_seed2026071102",
        "pc_gipmnn_seed2026071103",
    )
    summaries = []
    for directory in directories:
        path = RESULTS / directory / "summary.json"
        if not path.exists():
            raise RuntimeError(f"Missing completed baseline: {directory}")
        summary = json.loads(path.read_text(encoding="utf-8"))
        if not summary["accepted"]:
            raise RuntimeError(f"Baseline did not reach the published accuracy: {directory}")
        summaries.append(summary)
    output = RESULTS / "baseline_acceptance.json"
    output.write_text(json.dumps(summaries, indent=2), encoding="utf-8", newline="\n")
    print(output)


def command_plot(args: argparse.Namespace) -> None:
    output = Path(args.output)
    plot_baseline_curves(RESULTS, output)
    print(output)


def command_rfm(args: argparse.Namespace) -> None:
    print(json.dumps(run_configured_rfm(args.config), indent=2))


def command_search_rfm(args: argparse.Namespace) -> None:
    print(json.dumps(search_configured_rfm(args.config), indent=2))


def command_audit_rfm(_: argparse.Namespace) -> None:
    print(json.dumps(audit_rfm_results(), indent=2))


def command_paper_assets(args: argparse.Namespace) -> None:
    generate_experiment4_assets(args.output_root)
    print(Path(args.output_root).resolve())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproduce the 2D IAEA K-eigenvalue benchmark.")
    subparsers = parser.add_subparsers(required=True)
    reference = subparsers.add_parser("reference")
    reference.add_argument("--subdiv", type=int, default=16)
    reference.set_defaults(function=command_reference)
    baseline = subparsers.add_parser("baseline")
    baseline.add_argument("--config", required=True)
    baseline.add_argument("--epochs", type=int)
    baseline.add_argument("--output-directory")
    baseline.add_argument("--resume", action="store_true")
    baseline.set_defaults(function=command_baseline)
    baseline_suite = subparsers.add_parser("baseline-suite")
    baseline_suite.add_argument(
        "--config",
        action="append",
        help="TOML configuration; repeat to replace the default three-run suite.",
    )
    baseline_suite.add_argument(
        "--output-root",
        required=True,
        help="new empty suite directory; stored completed results are never read",
    )
    baseline_suite.set_defaults(function=command_baseline_suite)
    audit = subparsers.add_parser("audit-baselines")
    audit.set_defaults(function=command_audit)
    plot = subparsers.add_parser("plot-baselines")
    plot.add_argument("--output", default=str(RESULTS / "baseline_convergence.pdf"))
    plot.set_defaults(function=command_plot)
    rfm = subparsers.add_parser("rfm")
    rfm.add_argument("--config", default=str(EXPERIMENT4_CONFIGS / "rfm.toml"))
    rfm.set_defaults(function=command_rfm)
    rfm_search = subparsers.add_parser("rfm-search")
    rfm_search.add_argument("--config", default=str(EXPERIMENT4_CONFIGS / "rfm.toml"))
    rfm_search.set_defaults(function=command_search_rfm)
    audit_rfm = subparsers.add_parser("audit-rfm")
    audit_rfm.set_defaults(function=command_audit_rfm)
    paper_assets = subparsers.add_parser("paper-assets")
    paper_assets.add_argument("--output-root", required=True)
    paper_assets.set_defaults(function=command_paper_assets)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
