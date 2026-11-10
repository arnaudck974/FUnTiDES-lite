"""Estimate raw binary storage for compare.py, without running simulations."""

import argparse
import json
import math
from pathlib import Path


def positive_integer(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} doit être un entier positif")
    return value


def count_receivers(path):
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        values = [float(value) for value in line.split()]
        if len(values) != 3 or not all(map(math.isfinite, values)):
            raise ValueError(f"Position de récepteur invalide : {path}")
        count += 1
    return positive_integer(count, "Nombre de récepteurs")


def snapshot_storage_bytes(nodes, snapshots):
    """One float32 XYZ mesh per run, then one float32 pressure per node/output."""
    return nodes * (12 + 4 * snapshots) if snapshots else 0


def estimate(config_path, cases, data):
    # Do not import compare.py: estimation must also work before compilation.
    config = json.loads(config_path.read_text(encoding="utf-8"))
    params = config["parameters"]
    order = positive_integer(params["order"], "order")
    repetitions = positive_integer(config["repetitions"], "repetitions")
    cadence = positive_integer(params["snap-time-interval"], "snap-time-interval")
    intervals = config["precision_intervals"]
    if not isinstance(intervals, list) or not intervals:
        raise ValueError("precision_intervals doit être une liste non vide")
    for interval in intervals:
        positive_integer(interval, "Intervalle de précision")
    if len(set(intervals)) != len(intervals) or 1 not in intervals or cadence not in intervals:
        raise ValueError("Intervalles distincts requis, avec 1 et snap-time-interval")
    for name in ("dt", "timemax"):
        value = params[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} doit être positif et fini")
    steps = int(params["timemax"] / params["dt"])
    if steps < 1:
        raise ValueError("La simulation doit contenir au moins un pas")

    # Steps 0, k, 2k, ... are saved: ceil(T/k) snapshots, not floor(T/k).
    def count_snapshots(interval):
        return (steps + interval - 1) // interval

    snapshots_per_size = 0
    meshes_per_size = 0
    if "snapshots" in data:
        # Ad-hoc timing trials + one precision run per other k. The first
        # timing trial is reused at cadence. In-situ writes no raw fields.
        snapshots_per_size = repetitions * count_snapshots(cadence) + sum(
            count_snapshots(interval) for interval in intervals if interval != cadence
        )
        meshes_per_size = repetitions + sum(interval != cadence for interval in intervals)
    receivers = 0
    if "sismograms" in data:
        receivers = count_receivers((config_path.parent / config["receivers"]).resolve())

    rows = []
    for case in ("S", "M", "L"):
        if case not in cases:
            continue
        sizes = config["sizes"][case]
        if not isinstance(sizes, list) or not sizes:
            raise ValueError(f"sizes.{case} doit être une liste non vide")
        for size in sizes:
            positive_integer(size, f"sizes.{case}")
            nodes = (order * size + 1) ** 3
            snapshot_bytes = 4 * nodes
            mesh_bytes = 12 * nodes * meshes_per_size
            # Signal file: two int32 counts, then T rows of time + R pressures.
            signal_bytes = repetitions * (8 + 4 * steps * (receivers + 1)) if receivers else 0
            rows.append({
                "case": case, "size": size, "nodes": nodes,
                "snapshot_bytes": snapshot_bytes, "snapshot_count": snapshots_per_size,
                "mesh_bytes": mesh_bytes,
                "snapshots_bytes": mesh_bytes + snapshot_bytes * snapshots_per_size,
                "sismograms_bytes": signal_bytes,
            })
    return steps, receivers, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--cases", nargs="+", choices=("S", "M", "L"), default=["S", "M", "L"])
    parser.add_argument("--data", nargs="+", choices=("snapshots", "sismograms"),
                        default=["snapshots", "sismograms"])
    args = parser.parse_args()
    try:
        if len(set(args.cases)) != len(args.cases) or len(set(args.data)) != len(args.data):
            raise ValueError("Cas ou types répétés")
        steps, receivers, rows = estimate(args.config.resolve(), args.cases, args.data)
        print(f"Pas physiques estimés : {steps} ; récepteurs : {receivers}")
        print("Cas     N       Nœuds    MB/snapshot  Fichiers snapshot  Champs GB  Maillages GB  Sismos GB    Total GB")
        total = 0
        for row in rows:
            size_bytes = row["snapshots_bytes"] + row["sismograms_bytes"]
            total += size_bytes
            print(f"{row['case']:3} {row['size']:5} {row['nodes']:11} "
                  f"{row['snapshot_bytes']/1e6:14.5f} {row['snapshot_count']:18} "
                  f"{row['snapshot_bytes']*row['snapshot_count']/1e9:10.5f} "
                  f"{row['mesh_bytes']/1e9:12.5f} {row['sismograms_bytes']/1e9:10.6f} "
                  f"{size_bytes/1e9:11.5f}")
        print(f"TOTAL BRUT : {total / 1e9:.5f} GB = {total / 2**30:.5f} GiB")
        print("GB = 1 000 000 000 octets ; GiB = 1 073 741 824 octets.")
        print("Un mesh.bin par essai ad-hoc avec snapshots ; les références in-situ n'en écrivent pas.")
        print("T estimé par int(timemax/dt), comme l'estimation de compare.py ; "
              "les arrondis float32 du C++ peuvent modifier le nombre réel de pas.")
        print("Hors JSON de statistiques, métriques, logs, figures, rapports, "
              "espace de gestion du système de fichiers et résultats déjà présents.")
    except (OSError, ValueError, TypeError, KeyError, OverflowError) as error:
        parser.exit(1, f"Erreur : {error}\n")


if __name__ == "__main__":
    main()
