import argparse, pathlib, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--results", required=True)
ap.add_argument("--out-prefix", required=True)
ap.add_argument("--labels", nargs="+",
                default=["Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct"])
ap.add_argument("--xlabel", default="Month")
ap.add_argument("--xtick-rotation", type=float, default=None,
                help="defaults to 30 degrees when any label exceeds four "
                     "characters, which is where dated ticks start to collide")
a = ap.parse_args()

d = pd.read_csv(a.results)
style = {"s2_indices": ("S2", "blue"), "clay": ("Clay", "green"),
         "prithvi": ("Prithvi", "red")}

# Solid for RMSE, dashed for R^2, and the panel letter parked in whichever
# corner the curves leave empty: RMSE falls through the season so the label
# goes bottom-left, R^2 rises so it goes top-left.
PANELS = (
    ("rmse", "RMSE (bu/acre)", "(a)", "-", "bottom", 0.05),
    ("r2", "$R^2$", "(b)", "--", "top", 0.95),
)

for metric, ylabel, panel, dash, valign, ypos in PANELS:
    fig, ax = plt.subplots(figsize=(5.2, 3.8), dpi=600)
    for rep, (name, colour) in style.items():
        block = d[d.representation == rep].sort_values("composites")
        if block.empty:
            continue
        ax.plot(a.labels[:len(block)], block[f"{metric}_mean"],
                marker="s", linestyle=dash, linewidth=2.0, markersize=4,
                label=name, color=colour)
    ax.set_xlabel(a.xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    long_labels = max(len(x) for x in a.labels[:7])
    rotation = a.xtick_rotation if a.xtick_rotation is not None else (
        30.0 if long_labels > 4 else 0.0)
    ax.tick_params(axis="x", labelsize=11 if long_labels > 4 else 15)
    if rotation:
        plt.setp(ax.get_xticklabels(), rotation=rotation, ha="right",
                 rotation_mode="anchor")
    ax.tick_params(axis="y", labelsize=13)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.text(0.02, ypos, panel, transform=ax.transAxes,
            fontsize=14, fontweight="bold", va=valign)
    # Rotated ticks plus the axis label need more clearance than upright ones,
    # otherwise the legend lands on top of the xlabel.
    ax.legend(fontsize=10, loc="upper center",
              bbox_to_anchor=(0.5, -0.42 if rotation else -0.18),
              ncol=3, frameon=True, fancybox=False)
    plt.tight_layout()
    out = f"{a.out_prefix}{metric}_xgboost.png"
    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=600, bbox_inches="tight")
    print("wrote", out)
