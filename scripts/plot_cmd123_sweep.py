import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt


def plot_facet_scatter(df, xcol, ycol, zcol, out_path):
    modes = ["cmd1", "cmd2", "cmd3"]
    ncols = len(modes)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4), sharey=True)
    if ncols == 1:
        axes = [axes]

    vmin = df[zcol].min()
    vmax = df[zcol].max()

    for ax, mode in zip(axes, modes):
        sub = df[df["mode"] == mode]
        sc = ax.scatter(sub[xcol], sub[ycol], c=sub[zcol], s=30, vmin=vmin, vmax=vmax)
        ax.set_title(mode)
        ax.set_xlabel(xcol)
        ax.set_ylabel(ycol)

    cbar = fig.colorbar(sc, ax=axes, shrink=0.9)
    cbar.set_label(zcol)
    fig.suptitle(f"{zcol} faceted by mode: {xcol} vs {ycol}", y=1.02)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Plot 3D scatter for cmd123 sweep results")
    p.add_argument("--csv", type=str, default="scripts/bench_cmd123_sweep.csv")
    p.add_argument("--out_dir", type=str, default="scripts/bench_cmd123_sweep_plots")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.csv)

    pairs = [
        ("num_envs", "num_motions"),
        ("num_envs", "terminated_prob"),
        ("num_motions", "terminated_prob"),
    ]

    for xcol, ycol in pairs:
        out_name = f"{xcol}_vs_{ycol}.png"
        out_path = os.path.join(args.out_dir, out_name)
        plot_facet_scatter(df, xcol, ycol, "total_ms", out_path)

    print(f"Wrote plots to: {args.out_dir}")


if __name__ == "__main__":
    main()
