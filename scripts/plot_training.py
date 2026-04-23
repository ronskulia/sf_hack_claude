"""Plot a training CSV to a PNG.

Usage:
    python scripts/plot_training.py outputs/plots/tactical_training.csv
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from visualization.render import plot_training_curve


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv")
    p.add_argument("--out", default=None)
    p.add_argument("--title", default=None)
    args = p.parse_args()
    out = args.out or args.csv.replace(".csv", ".png")
    title = args.title or os.path.basename(args.csv)
    fig = plot_training_curve(args.csv, save_path=out, title=title)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
