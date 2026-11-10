#!/usr/bin/env python3
"""Plot SEM proxy binary sismos and horizontal binary snapshot slices."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze import load_metrics
from snapshot_io import load_mesh, load_snapshot


def save_metrics(section, values):
    # Simulation et visualisations partagent le rapport du dossier courant.
    path = Path("metrics.json")
    metrics = load_metrics()
    metrics["ad_hoc"]["visualization"][section] = values
    path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


def plot_sismos(bin_path, output):
    start_read = perf_counter()
    # Native format: int32 receiver/sample counts, then float32 time + pressures.
    with bin_path.open("rb") as source:
        dimensions = np.fromfile(source, dtype="=i4", count=2)
        if dimensions.size != 2 or np.any(dimensions <= 0):
            raise ValueError("Expected positive receiver and sample counts")
        receivers, samples = map(int, dimensions)
        if bin_path.stat().st_size != 8 + 4 * samples * (receivers + 1):
            raise ValueError("Sismogram file size does not match its dimensions")
        values = np.fromfile(source, dtype="=f4")
    read_seconds = perf_counter() - start_read

    start_analysis = perf_counter()
    values = values.reshape(samples, receivers + 1)
    times = values[:, 0]
    pressures = values[:, 1:]
    preparation_seconds = perf_counter() - start_analysis

    start_visualization = perf_counter()
    fig, ax = plt.subplots()
    for receiver in range(receivers):
        ax.plot(times, pressures[:, receiver], label=f"receiver_{receiver}")
    ax.set(xlabel="Temps (s)", ylabel="Pression", title="Sismogrammes")
    ax.grid(True)
    ax.legend()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    render_seconds = perf_counter() - start_visualization
    save_metrics("sismograms", {
        "read_seconds": read_seconds,
        "preparation_seconds": preparation_seconds,
        "render_seconds": render_seconds,
    })
    print(output)


def edges(centers):
    midpoints = (centers[:-1] + centers[1:]) / 2
    return np.r_[centers[0], midpoints, centers[-1]]


def plot_snapshots(paths, requested_z, out_dir):
    start_read = perf_counter()
    coordinates = load_mesh(paths[0].parent / "mesh.bin")
    node_count = len(coordinates)
    first = load_snapshot(paths[0], node_count)
    read_seconds = perf_counter() - start_read
    start_analysis = perf_counter()
    levels = np.unique(coordinates[:, 2])
    middle_z = (levels[0] + levels[-1]) / 2
    z = levels[np.argmin(abs(levels - (requested_z if requested_z is not None else middle_z)))]
    selected = coordinates[:, 2] == z

    planes = []
    preparation_seconds = perf_counter() - start_analysis
    for index, path in enumerate(paths):
        if index == 0:
            data = first
        else:
            start_read = perf_counter()
            data = load_snapshot(path, node_count)
            read_seconds += perf_counter() - start_read
        start_analysis = perf_counter()
        points = coordinates[selected]
        if not len(points):
            raise ValueError(f"No nodes at z={z:g} in {path}")
        x, x_index = np.unique(points[:, 0], return_inverse=True)
        y, y_index = np.unique(points[:, 1], return_inverse=True)
        if len(x) < 2 or len(y) < 2:
            raise ValueError(f"At least two x and y coordinates are needed in {path}")
        pressure = np.full((len(y), len(x)), np.nan)
        pressure[y_index, x_index] = data[selected]
        planes.append((path, x, y, pressure))
        preparation_seconds += perf_counter() - start_analysis

    start_analysis = perf_counter()
    scale = max(np.nanmax(np.abs(pressure)) for _, _, _, pressure in planes)
    if scale == 0:
        scale = 1
    preparation_seconds += perf_counter() - start_analysis
    out_dir.mkdir(parents=True, exist_ok=True)
    render_seconds = 0.0
    for path, x, y, pressure in planes:
        start_visualization = perf_counter()
        fig, ax = plt.subplots()
        image = ax.pcolormesh(edges(x), edges(y), pressure,
                              cmap="RdBu_r", vmin=-scale, vmax=scale)
        ax.set(xlabel="x", ylabel="y", title=f"{path.name} — z = {z:g}")
        ax.set_aspect("equal")
        fig.colorbar(image, ax=ax, label="Pression")
        output = out_dir / f"{path.stem}.svg"
        fig.savefig(output, bbox_inches="tight")
        plt.close(fig)
        render_seconds += perf_counter() - start_visualization
        print(output)
    save_metrics("snapshots", {
        "read_seconds": read_seconds,
        "preparation_seconds": preparation_seconds,
        "render_seconds": render_seconds,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    sismos = commands.add_parser("sismos")
    sismos.add_argument("bin", type=Path)
    sismos.add_argument("--output", type=Path, default=Path("sismos.svg"))
    snapshots = commands.add_parser("snapshots")
    snapshots.add_argument(
        "directory", type=Path, help="directory containing snapshot_<step>.bin files"
    )
    snapshots.add_argument("--z", type=float, help="nearest available z plane")
    snapshots.add_argument("--out-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()
    if args.command == "sismos":
        plot_sismos(args.bin, args.output)
    else:
        if not args.directory.is_dir():
            parser.error(f"Snapshot directory does not exist: {args.directory}")
        paths = sorted(
            (path for path in args.directory.glob("snapshot_*.bin")
             if path.is_file() and path.stem.split("_")[-1].isdigit()),
            key=lambda path: int(path.stem.split("_")[-1]),
        )
        if not paths:
            parser.error(f"No snapshot_<step>.bin files found in {args.directory}")
        plot_snapshots(paths, args.z, args.out_dir)


if __name__ == "__main__":
    main()
