#!/usr/bin/env python3
"""Plot fold-wise yield and within-county anomaly metrics as composites accumulate.

Conventional yield performance rewards recovering a county's long-run mean,
which is 84.6% of the variance in this cohort. The anomaly curve removes each
county's mean from both observed and predicted yield. Plotting them together
shows that early performance of the learned representations largely reflects
stable between-county differences: mean outer-fold R^2 is already near 0.66 in
April while anomaly R^2 is near zero.

    python scripts/plot_inseason_anomaly.py \\
        --predictions outputs/inseason_covered/inseason_results_predictions.csv \\
        --out figures/inseason_anomaly.png
"""
import argparse, pathlib
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STYLE = {"s2_indices": ("Sentinel-2 indices", "blue"),
         "clay": ("Clay", "green"),
         "prithvi": ("Prithvi-EO-2.0", "red")}
LABELS = ["Apr 15", "May 13", "Jun 10", "Jul 8", "Aug 5", "Sep 2", "Sep 30"]


def rmse(observed, predicted):
    observed = np.asarray(observed, float)
    predicted = np.asarray(predicted, float)
    return float(np.sqrt(np.mean((observed - predicted) ** 2)))


def r2(observed, predicted):
    observed = np.asarray(observed, float)
    predicted = np.asarray(predicted, float)
    return 1 - ((observed - predicted) ** 2).sum() / (
        (observed - observed.mean()) ** 2).sum()


def curves(frame):
    """Mean and population SD of each metric across outer test folds."""
    rows = []
    for (rep, k, fold), group in frame.groupby(
            ["representation", "composites", "fold"]):
        # Counties observed once contribute no within-county variance.
        repeated = group.groupby("county_id").filter(lambda g: len(g) > 1)
        observed = repeated.observed_yield - repeated.groupby(
            "county_id").observed_yield.transform("mean")
        predicted = repeated.predicted_yield - repeated.groupby(
            "county_id").predicted_yield.transform("mean")
        rows.append({"representation": rep, "composites": k, "fold": fold,
                     "yield_r2": r2(group.observed_yield, group.predicted_yield),
                     "anomaly_r2": r2(observed, predicted),
                     "yield_rmse": rmse(group.observed_yield, group.predicted_yield),
                     "anomaly_rmse": rmse(observed, predicted),
                     # Zero anomaly is the reference prediction in the
                     # county-demeaned evaluation space.
                     "zero_anomaly_rmse": rmse(observed, np.zeros(len(observed)))})
    per_fold = pd.DataFrame(rows)
    metrics = ("yield_r2", "anomaly_r2", "yield_rmse", "anomaly_rmse",
               "zero_anomaly_rmse")
    summary = per_fold.groupby(["representation", "composites"], as_index=False)[
        list(metrics)
    ].agg(["mean", lambda values: np.std(values, ddof=0)])
    summary.columns = [
        "representation", "composites",
        *[f"{metric}_{stat}" for metric in metrics for stat in ("mean", "std")],
    ]
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--xlabel", default="Composite date")
    ap.add_argument("--metric", choices=("r2", "rmse"), default="r2",
                    help="r2 plots mean outer-fold yield and anomaly R^2; rmse "
                         "plots the corresponding errors in bushels per acre, "
                         "with the zero-anomaly reference RMSE")
    args = ap.parse_args()

    data = curves(pd.read_csv(args.predictions, dtype={"county_id": str}))

    if args.metric == "r2":
        panels = ((0, "yield_r2", "(a) Mean outer-fold $R^2$", "-"),
                  (1, "anomaly_r2", "(b) Within-county anomaly $R^2$", "--"))
        ylabel = "$R^2$"
    else:
        panels = ((0, "yield_rmse", "(c) Mean outer-fold RMSE", "-"),
                  (1, "anomaly_rmse", "(d) Within-county anomaly RMSE", "--"))
        ylabel = "RMSE (bu/acre)"

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), dpi=600, sharex=True)
    for index, metric, title, dash in panels:
        ax = axes[index]
        for rep, (name, colour) in STYLE.items():
            block = data[data.representation == rep].sort_values("composites")
            if block.empty:
                continue
            positions = np.arange(len(block))
            mean = block[f"{metric}_mean"].to_numpy()
            std = block[f"{metric}_std"].to_numpy()
            ax.fill_between(positions, mean - std, mean + std, color=colour,
                            alpha=0.12, linewidth=0)
            ax.plot(positions, mean, marker="s",
                    linestyle=dash, linewidth=2.0, markersize=4,
                    label=name, color=colour)
        if args.metric == "r2":
            ax.axhline(0.0, color="0.6", linewidth=0.8, zorder=0)
        elif metric == "anomaly_rmse":
            reference = float(data.zero_anomaly_rmse_mean.mean())
            ax.axhline(reference, color="0.45", linewidth=1.0, linestyle=":",
                       zorder=0)
            ax.text(0.02, reference, "zero-anomaly baseline",
                    transform=ax.get_yaxis_transform(),
                    va="bottom", fontsize=9, color="0.35")
        ax.set_xticks(np.arange(len(LABELS)), LABELS)
        ax.set_xlabel(args.xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.tick_params(axis="x", labelsize=9)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right",
                 rotation_mode="anchor")
        ax.tick_params(axis="y", labelsize=10)
    if args.metric == "r2":
        lower = (data.anomaly_r2_mean - data.anomaly_r2_std).min()
        axes[1].set_ylim(bottom=min(-0.9, lower - 0.05))
    axes[0].legend(fontsize=10,
                   loc="lower right" if args.metric == "r2" else "upper right",
                   frameon=True, fancybox=False)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=600, bbox_inches="tight")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
