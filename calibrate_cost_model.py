"""Build lookup-backed Phase C cost tables and held-out validation reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root", type=Path, default=Path("measurements/phase_c_v2")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("measurements/phase_c_v2/calibration")
    )
    return parser.parse_args()


def interpolate(train_x: np.ndarray, train_y: np.ndarray, query: float) -> float:
    order = np.argsort(train_x)
    x, y = train_x[order], train_y[order]
    return float(np.interp(np.log2(query), np.log2(x), y))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    validations: list[dict[str, object]] = []
    model: dict[str, object] = {
        "collectives": {},
        "matmul": {},
        "overlap": {},
        "sequences": {},
    }

    for csv_path in sorted((args.input_root / "collectives").glob("*/collectives.csv")):
        data = pd.read_csv(csv_path)
        data = data[data.timing_mode == "amortized"]
        topology = csv_path.parent.name
        model["collectives"][topology] = data.to_dict(orient="records")
        for collective, group in data.groupby("collective"):
            group = group.sort_values("actual_bytes_per_rank")
            validation_indices = {2, 4, 6}
            train = group.iloc[
                [i for i in range(len(group)) if i not in validation_indices]
            ]
            for index in validation_indices:
                observed = group.iloc[index]
                predicted = interpolate(
                    train.actual_bytes_per_rank.to_numpy(),
                    train.median_us.to_numpy(),
                    observed.actual_bytes_per_rank,
                )
                validations.append(
                    {
                        "component": "collective",
                        "configuration": topology,
                        "case": collective,
                        "size": int(observed.actual_bytes_per_rank),
                        "observed_us": observed.median_us,
                        "predicted_us": predicted,
                        "relative_error": abs(predicted - observed.median_us)
                        / observed.median_us,
                    }
                )

    for csv_path in sorted((args.input_root / "matmul").glob("*/matmul.csv")):
        data = pd.read_csv(csv_path)
        data = data[data.timing_mode == "amortized"].reset_index(drop=True)
        configuration = csv_path.parent.name
        validation_indices = {1, 4}
        train = data.iloc[[i for i in range(len(data)) if i not in validation_indices]]
        coefficients = np.polyfit(np.log(train.flops), np.log(train.median_us), 1)
        model["matmul"][configuration] = {
            "log_latency_fit_slope": float(coefficients[0]),
            "log_latency_fit_intercept": float(coefficients[1]),
            "lookup": data.to_dict(orient="records"),
        }
        for index in validation_indices:
            observed = data.iloc[index]
            predicted = float(np.exp(np.polyval(coefficients, np.log(observed.flops))))
            validations.append(
                {
                    "component": "matmul",
                    "configuration": configuration,
                    "case": f"{observed.m}x{observed.k}x{observed.n}",
                    "size": int(observed.flops),
                    "observed_us": observed.median_us,
                    "predicted_us": predicted,
                    "relative_error": abs(predicted - observed.median_us)
                    / observed.median_us,
                }
            )

    for csv_path in sorted((args.input_root / "overlap").glob("*/overlap.csv")):
        data = pd.read_csv(csv_path)
        configuration = csv_path.parent.name
        predictions = (
            data.baseline_median_us
            + data.corrected_compute_us
            + data.corrected_communication_us
        )
        model["overlap"][configuration] = {
            "realization": "serialized",
            "reason": "optimized HLO marks all-reduce is_sync=true and is_pipelined=false",
            "lookup": data.to_dict(orient="records"),
        }
        for (_, observed), predicted in zip(data.iterrows(), predictions, strict=True):
            validations.append(
                {
                    "component": "overlap",
                    "configuration": configuration,
                    "case": "serialized_matmul_all_reduce",
                    "size": int(observed.bytes_per_rank),
                    "observed_us": observed.combined_median_us,
                    "predicted_us": predicted,
                    "relative_error": abs(predicted - observed.combined_median_us)
                    / observed.combined_median_us,
                }
            )

    for csv_path in sorted((args.input_root / "sequences").glob("*/sequences.csv")):
        model["sequences"][csv_path.parent.name] = pd.read_csv(csv_path).to_dict(
            orient="records"
        )

    validation = pd.DataFrame(validations)
    validation.to_csv(args.output_dir / "validation.csv", index=False)
    (args.output_dir / "cost_model.json").write_text(json.dumps(model, indent=2) + "\n")
    aggregate = validation.groupby("component").relative_error.agg(["median", "max"])
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for csv_path in sorted((args.input_root / "collectives").glob("*/collectives.csv")):
        data = pd.read_csv(csv_path)
        data = data[
            (data.timing_mode == "amortized") & (data.collective == "all_reduce")
        ]
        axes[0].plot(
            data.actual_bytes_per_rank / 2**20,
            data.median_us,
            marker="o",
            label=csv_path.parent.name,
        )
    axes[0].set(
        xscale="log",
        yscale="log",
        xlabel="Input MiB per rank",
        ylabel="Median us",
        title="All-reduce cost",
    )
    axes[0].legend()
    axes[1].scatter(
        validation.observed_us,
        validation.predicted_us,
        c=pd.Categorical(validation.component).codes,
    )
    maximum = max(validation.observed_us.max(), validation.predicted_us.max())
    axes[1].plot([0, maximum], [0, maximum], linestyle="--", color="black")
    axes[1].set(
        xscale="log",
        yscale="log",
        xlabel="Observed us",
        ylabel="Predicted us",
        title="Validation",
    )
    figure.tight_layout()
    figure.savefig(args.output_dir / "calibration.png", dpi=160)
    plt.close(figure)
    lines = [
        "# Phase C calibration report",
        "",
        "Primary cost representation: measured lookup tables with interpolation. ",
        "Analytical fits are diagnostics; they do not replace the raw tables.",
        "",
        "## Held-out and structural validation",
        "",
        "| Component | Median relative error | Maximum relative error |",
        "|---|---:|---:|",
    ]
    for component, row in aggregate.iterrows():
        lines.append(f"| {component} | {row['median']:.1%} | {row['max']:.1%} |")
    lines += [
        "",
        "Collective validation holds out 64 KiB, 1 MiB, and 16 MiB points and ",
        "interpolates between measured neighbors. Matmul validation holds out two ",
        "shapes and uses a log-FLOP/log-latency diagnostic fit.",
        "",
        "The tested combined HLO uses synchronous, non-pipelined all-reduce. Its ",
        "model is therefore serialized after removing one shared launch baseline.",
        "",
        "## Interpretation and limits",
        "",
        "- Collective interpolation is reliable in the small-message regime but ",
        "  fails around backend algorithm transitions; the worst held-out error is ",
        "  intentionally retained. Phase D should use exact lookup points.",
        "- FLOP count alone does not model matmul efficiency. The lookup key is ",
        "  `(M, K, N, dtype)`; unseen shapes require measurement or a richer model.",
        "- The 1 MiB overlap correction is noise-sensitive because useful device ",
        "  work is small relative to the launch floor. For 16 and 64 MiB, the ",
        "  serialized model is the supported realization.",
        "- Memory values are XLA static executable analysis, not sampled allocator ",
        "  high-water marks. They are suitable as candidate estimates but require ",
        "  trace validation for final performance claims.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n")
    print(aggregate.to_string(float_format=lambda value: f"{value:.1%}"))
    print(f"Calibration written to: {args.output_dir}")


if __name__ == "__main__":
    main()
