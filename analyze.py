#!/usr/bin/env python3
"""Compute ad-hoc pressure statistics from SEM binary outputs."""

import argparse
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np

from snapshot_io import load_mesh, load_snapshot


def empty_statistics_metrics():
    return {
        "snapshots": {"read_seconds": None, "analysis_seconds": None},
        "sismograms": {"read_seconds": None, "analysis_seconds": None},
        "output_file": {"write_seconds": None, "size_mb": None},
    }


def load_metrics():
    """Read the shared report, retaining measurements from the other scripts."""
    path = Path("metrics.json")
    metrics = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    metrics.setdefault("simulation", dict.fromkeys((
        "total_seconds", "compute_seconds", "snapshot_write_seconds",
        "sismogram_write_seconds", "snapshot_mb", "sismogram_mb",
    )))
    ad_hoc = metrics.setdefault("ad_hoc", {})
    ad_hoc.setdefault("statistics", empty_statistics_metrics())
    visualization = ad_hoc.setdefault("visualization", {})
    # Accept TP1 reports produced before recompilation, without duplicating them.
    previous = metrics.pop("postprocessing", {})
    for name, previous_name in (("snapshots", "snapshots"), ("sismograms", "sismos")):
        values = previous.get(previous_name, {})
        visualization.setdefault(name, {
            "read_seconds": values.get("read_seconds"),
            "preparation_seconds": values.get("analysis_seconds"),
            "render_seconds": values.get("visualization_seconds"),
        })
    metrics.setdefault("in_situ", None)
    return metrics


def pressure_statistics(pressures):
    if pressures.size == 0 or not np.all(np.isfinite(pressures)):
        raise ValueError("Expected nonempty, finite pressure values")
    minimum = float(np.min(pressures))
    maximum = float(np.max(pressures))
    return {
        "minimum_pressure": minimum,
        "maximum_pressure": maximum,
        "maximum_absolute_pressure": max(abs(minimum), abs(maximum)),
        "mean_pressure": float(np.mean(pressures, dtype=np.float64)),
        "pressure_variance": float(np.var(pressures, dtype=np.float64, ddof=0)),
    }


def analyze_snapshots(directory, dt):
    start_read = perf_counter()
    if not directory.is_dir():
        raise ValueError(f"Snapshot directory does not exist: {directory}")
    paths = sorted(
        (path for path in directory.glob("snapshot_*.bin")
         if path.is_file() and path.stem[len("snapshot_"):].isdigit()),
        key=lambda path: int(path.stem[len("snapshot_"):]),
    )
    if not paths:
        raise ValueError(f"No snapshot_<step>.bin files found in {directory}")
    node_count = len(load_mesh(directory / "mesh.bin"))
    read_seconds = perf_counter() - start_read
    analysis_seconds = 0.0
    results = []
    extrema = None
    for path in paths:
        start_read = perf_counter()
        pressures = load_snapshot(path, node_count)
        read_seconds += perf_counter() - start_read

        start_analysis = perf_counter()
        step = int(path.stem[len("snapshot_"):])
        time = (step + 1) * dt if dt is not None else None
        if time is not None and not math.isfinite(time):
            raise ValueError(f"Nonfinite snapshot time: {path}")
        statistics = pressure_statistics(pressures)
        results.append({
            "timestep_index": step,
            "time_seconds": time,
            "node_count": node_count,
            **statistics,
        })
        if extrema is None:
            extrema = {name: statistics[name] for name in (
                "minimum_pressure", "maximum_pressure", "maximum_absolute_pressure",
            )}
        else:
            extrema["minimum_pressure"] = min(
                extrema["minimum_pressure"], statistics["minimum_pressure"])
            for name in ("maximum_pressure", "maximum_absolute_pressure"):
                extrema[name] = max(extrema[name], statistics[name])
        analysis_seconds += perf_counter() - start_analysis
        del pressures  # Release this field before loading the next snapshot.
    return {"per_snapshot": results, "global_extrema": extrema}, {
        "read_seconds": read_seconds, "analysis_seconds": analysis_seconds,
    }


def analyze_sismograms(path):
    start_read = perf_counter()
    # Native format: int32 receiver/sample counts, then float32 time + pressures.
    with path.open("rb") as source:
        dimensions = np.fromfile(source, dtype="=i4", count=2)
        if dimensions.size != 2 or np.any(dimensions <= 0):
            raise ValueError("Expected positive receiver and sample counts")
        receivers, samples = map(int, dimensions)
        if path.stat().st_size != 8 + 4 * samples * (receivers + 1):
            raise ValueError("Sismogram file size does not match its dimensions")
        values = np.fromfile(source, dtype="=f4")
    read_seconds = perf_counter() - start_read

    start_analysis = perf_counter()
    values = values.reshape(samples, receivers + 1)
    if not np.all(np.isfinite(values[:, 0])):
        raise ValueError("Expected finite sismogram times")
    results = [
        {"receiver_index": receiver, "sample_count": samples,
         **pressure_statistics(values[:, receiver + 1])}
        for receiver in range(receivers)
    ]
    return {"per_receiver": results}, {
        "read_seconds": read_seconds,
        "analysis_seconds": perf_counter() - start_analysis,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=Path, help="Directory of snapshot_<step>.bin files")
    parser.add_argument("--dt", type=float, help="Actual simulation time step in seconds (optional)")
    parser.add_argument("--sismos", type=Path, help="Binary sismogram file")
    args = parser.parse_args()
    if args.snapshots is None and args.sismos is None:
        parser.error("Provide --snapshots and/or --sismos")
    if args.dt is not None:
        if args.snapshots is None:
            parser.error("--dt is only used with --snapshots")
        if not math.isfinite(args.dt) or args.dt <= 0:
            parser.error("--dt must be finite and strictly positive")

    try:
        metrics = load_metrics()
        statistics = {"snapshots": None, "sismograms": None}
        timings = empty_statistics_metrics()
        if args.snapshots is not None:
            statistics["snapshots"], timings["snapshots"] = analyze_snapshots(args.snapshots, args.dt)
        if args.sismos is not None:
            statistics["sismograms"], timings["sismograms"] = analyze_sismograms(args.sismos)

        start_write = perf_counter()
        output = Path("statistics_ad-hoc.json")
        output.write_text(json.dumps(statistics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        size_mb = output.stat().st_size / 1e6
        timings["output_file"] = {
            "write_seconds": perf_counter() - start_write, "size_mb": size_mb,
        }
        metrics["ad_hoc"]["statistics"] = timings
        Path("metrics.json").write_text(
            json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(output)
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as error:
        parser.exit(1, f"Erreur : {error}\n")


if __name__ == "__main__":
    main()
