#!/usr/bin/env python3
"""Show three pressure snapshots with a shared plane and color scale."""

import argparse
import math
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.snapshot_io import load_mesh, load_snapshot


def prepare_planes(directory, steps, requested_z=None):
    """Read the mesh once and extract the same plane without interpolation."""
    coordinates = load_mesh(directory / "mesh.bin")
    levels = np.unique(coordinates[:, 2])
    target = (float(levels[0]) + float(levels[-1])) / 2 if requested_z is None else requested_z
    z = float(levels[np.argmin(abs(levels.astype(np.float64) - target))])
    selected = coordinates[:, 2] == z
    points = coordinates[selected]
    x, columns = np.unique(points[:, 0], return_inverse=True)
    y, rows = np.unique(points[:, 1], return_inverse=True)
    if len(x) < 2 or len(y) < 2:
        raise ValueError("Le plan doit contenir plusieurs coordonnées x et y.")

    # Reject holes or duplicate positions instead of silently altering the field.
    cells = rows * len(x) + columns
    if len(points) != len(x) * len(y) or len(np.unique(cells)) != len(points):
        raise ValueError("Le plan ne forme pas une grille cartésienne complète.")

    planes = []
    for step in steps:
        pressures = load_snapshot(directory / f"snapshot_{step}.bin", len(coordinates))
        plane = np.empty((len(y), len(x)), dtype=np.float32)
        plane[rows, columns] = pressures[selected]
        planes.append(plane)
    return x, y, z, planes


def edges(centers):
    """Bound the node-centered cells by midpoints and the outermost nodes."""
    midpoints = (centers[:-1].astype(np.float64) + centers[1:]) / 2
    return np.r_[centers[0], midpoints, centers[-1]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Dossier contenant mesh.bin et les snapshots")
    parser.add_argument("--steps", nargs=3, type=int, required=True,
                        help="Trois indices de snapshots existants, dans l'ordre croissant")
    parser.add_argument("--dt", type=float, required=True,
                        help="Pas de temps réellement utilisé, en secondes")
    parser.add_argument("--z", type=float,
                        help="Plan disponible le plus proche ; milieu du domaine par défaut")
    parser.add_argument("--output", type=Path, default=Path("figures/propagation.svg"))
    args = parser.parse_args()
    if args.steps[0] < 0 or not args.steps[0] < args.steps[1] < args.steps[2]:
        parser.error("--steps : trois indices distincts, croissants et positifs ou nuls")
    if not math.isfinite(args.dt) or args.dt <= 0:
        parser.error("--dt doit être fini et strictement positif")
    if args.z is not None and not math.isfinite(args.z):
        parser.error("--z doit être fini")
    if args.output.suffix.lower() not in (".svg", ".png", ".pdf"):
        parser.error("--output doit finir par .svg, .png ou .pdf")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        x, y, z, planes = prepare_planes(args.directory, args.steps, args.z)
    except (ImportError, OSError, ValueError) as error:
        parser.exit(1, f"Erreur : {error}\n")

    amplitude = max(float(np.max(np.abs(plane))) for plane in planes)
    limit = amplitude if amplitude else 1.0
    fig, axes = plt.subplots(
        1, 3, figsize=(13, 4.6), sharex=True, sharey=True, constrained_layout=True,
    )
    try:
        for ax, step, plane in zip(axes, args.steps, planes):
            # Snapshot index 0 follows the first computeOneStep call.
            time = (step + 1) * args.dt
            image = ax.pcolormesh(
                edges(x), edges(y), plane, shading="flat",
                cmap="RdBu_r", vmin=-limit, vmax=limit, rasterized=True,
            )
            ax.set(title=f"t = {time:.6g} s — indice {step}", xlabel="Coordonnée x")
            ax.set_aspect("equal")
        axes[0].set_ylabel("Coordonnée y")
        fig.colorbar(image, ax=list(axes), label="Pression (unités du modèle)", shrink=0.8)
        title = f"Propagation de l'onde — coupes à z = {z:g}"
        if amplitude == 0:
            title += " — pression nulle dans les trois coupes"
        fig.suptitle(title)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=180)
    finally:
        plt.close(fig)
    print(f"Plan retenu : z = {z:g}")
    print(f"Échelle commune : [{-limit:g}, {limit:g}]")
    print(f"Figure : {args.output}")


if __name__ == "__main__":
    main()
