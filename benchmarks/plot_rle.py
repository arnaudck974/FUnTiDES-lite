#!/usr/bin/env python3
"""Plot the RLE compression ratio of each snapshot over the simulation.

Usage: python3 benchmarks/plot_rle.py [rle/rle_snapshots.csv] [-o rle_ratio.png]
"""
import argparse
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", default="rle/rle_snapshots.csv")
    parser.add_argument("-o", "--output", default="rle_ratio.png")
    args = parser.parse_args()

    with open(args.csv, newline="") as source:
        rows = list(csv.DictReader(source))
    time = [float(r["time_s"]) for r in rows]
    ratio = [float(r["ratio"]) for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(time, ratio, marker="o")
    ax.axhline(1.0, color="grey", linestyle="--", label="no gain (ratio = 1)")
    ax.set_yscale("log")
    ax.set(xlabel="Simulated time (s)", ylabel="Compression ratio (raw / RLE)",
           title="RLE compression ratio per snapshot")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()