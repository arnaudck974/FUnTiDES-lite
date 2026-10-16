#!/usr/bin/env python3
"""Statistiques ad-hoc sur les snapshots (TP2).

Usage :
  python3 snapshot_stats.py build/snapshots_4_threads --dt 0.001
"""
import argparse
import glob
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("folder")
parser.add_argument("--dt", type=float, default=0.001)
parser.add_argument("--src", type=float, nargs=3, default=[1010, 1010, 1010])
parser.add_argument("--bins", type=int, default=50)
parser.add_argument("--t1", type=float, default=0.3)
parser.add_argument("--t2", type=float, default=0.5)
args = parser.parse_args()

# Liste des snapshots triés par numéro de pas
files = glob.glob(f"{args.folder}/snapshot_*.csv")
files = sorted(files, key=lambda f: int(re.search(r"snapshot_(\d+)", f).group(1)))

rows = []
for f in files:
    step = int(re.search(r"snapshot_(\d+)", f).group(1))
    data = pd.read_csv(f)
    p = data["pressure"].to_numpy(dtype=np.float64)

    # Rayon du front : distance max à la source des noeuds où |p| > 10% du max
    dist = np.sqrt((data["x"] - args.src[0]) ** 2 +
                   (data["y"] - args.src[1]) ** 2 +
                   (data["z"] - args.src[2]) ** 2).to_numpy()
    seuil = 0.1 * np.abs(p).max()
    radius = dist[np.abs(p) > seuil].max() if seuil > 0 else 0.0

    rows.append({
        "step": step,
        "time": (step + 1) * args.dt,
        "min": p.min(),
        "max": p.max(),
        "mean": p.mean(),
        "var": p.var(),
        "energy": np.sum(p ** 2),
        "front_radius": radius,
    })
    print(f"{f} : max={p.max():.3e}  energy={np.sum(p**2):.3e}")

    # Histogramme du dernier snapshot lu (pour la figure)
    last_p, last_step = p, step

stats = pd.DataFrame(rows)
stats.to_csv("adhoc_snapshot_stats.csv", index=False)
print("-> adhoc_snapshot_stats.csv")

# Évolution du max et de l'énergie dans le temps
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(stats["time"], stats["max"])
ax1.set_ylabel("max(p)")
ax2.plot(stats["time"], stats["energy"])
ax2.set_ylabel("énergie")
ax2.set_xlabel("temps (s)")
fig.savefig("adhoc_max_energy.png")

# Histogramme
plt.figure()
plt.hist(last_p, bins=args.bins, log=True)
plt.title(f"Histogramme de la pression, step {last_step}")
plt.xlabel("pression")
plt.savefig("adhoc_histogram.png")

# Vitesse de l'onde : rayon du front à deux instants (snapshots les plus proches
# de --t1 et --t2 ; éviter le tout début et le moment où le front touche les bords)
i1 = (stats["time"] - args.t1).abs().idxmin()
i2 = (stats["time"] - args.t2).abs().idxmin()
r1, r2 = stats["front_radius"][i1], stats["front_radius"][i2]
t1, t2 = stats["time"][i1], stats["time"][i2]
v = (r2 - r1) / (t2 - t1)
print(f"Vitesse : ({r2:.0f} - {r1:.0f}) / ({t2:.3f} - {t1:.3f}) = {v:.0f} m/s")
