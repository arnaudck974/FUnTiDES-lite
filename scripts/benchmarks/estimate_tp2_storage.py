#!/usr/bin/env python3
"""Estimate cumulative writes and peak temporary storage for the TP2 campaigns."""

import json
import math
from pathlib import Path
import shutil
import sys

# This estimator only reads files, including when importing the shared helpers.
if __name__ == "__main__":
    sys.dont_write_bytecode = True
from estimate_storage import count_receivers, estimate, positive_integer, snapshot_storage_bytes


STORAGE_BUDGET = 20_000_000_000  # Decimal GB, as in the measurement reports.
STORAGE_RESERVE = 1_000_000_000  # New statistics, logs and filesystem overhead.
CAMPAIGNS = ("strong-scaling", "weak-scaling", "ad-hoc-vs-in-situ-cuda",
             "extrema-cpu-vs-gpu")


def raw_run_bytes(config, size, kind, interval, receivers, extra_step=False):
    params = config["parameters"]
    steps = int(params["timemax"] / params["dt"]) + int(extra_step)
    if kind == "snapshots":
        dimensions = (size, size, size) if isinstance(size, int) else size
        nodes = math.prod(params["order"] * n + 1 for n in dimensions)
        return snapshot_storage_bytes(nodes, (steps + interval - 1) // interval)
    return 8 + 4 * steps * (count_receivers(Path(receivers)) + 1)


def check_storage_budget(config, size, kind, mode, interval, directory, receivers):
    """Check before timing; include existing build/reports and allow one extra step."""
    raw = (raw_run_bytes(config, size, kind, interval, receivers, extra_step=True)
           if mode == "ad-hoc" else 0)
    results = directory.parent.parent
    paths = {Path(config["executable"]).resolve().parents[1], directory.parent}
    paths.update(results / name for name in CAMPAIGNS)
    occupied = 0
    for path in paths:
        if path.is_symlink():
            raise ValueError(f"Contrôle du stockage : lien symbolique refusé : {path}")
        if path.exists():
            occupied += sum(p.stat().st_size for p in path.rglob("*")
                            if p.is_file() and not p.is_symlink())
    required = raw + STORAGE_RESERVE
    if occupied + required > STORAGE_BUDGET:
        raise ValueError(f"Budget de 20 GB dépassé avant {directory.name} : "
                         f"{occupied/1e9:.3f} GB présents + {raw/1e9:.3f} GB bruts "
                         "+ 1 GB de réserve. Aucun nouvel essai lancé.")
    if shutil.disk_usage(directory.parent).free < required:
        raise ValueError(f"Espace libre insuffisant avant {directory.name} : "
                         f"{required/1e9:.3f} GB nécessaires, réserve comprise")


def read_config(path):
    config = json.loads(path.read_text(encoding="utf-8"))
    params = config["parameters"]
    order = positive_integer(params["order"], "order")
    repetitions = positive_integer(config["repetitions"], "repetitions")
    cadence = positive_integer(params["snap-time-interval"], "snap-time-interval")
    if order not in (1, 2, 3) or repetitions < 2:
        raise ValueError(f"{path.name} : order doit être 1, 2 ou 3 et repetitions >= 2")
    for key in ("dt", "timemax"):
        if type(params[key]) not in (int, float) or not math.isfinite(params[key]) or params[key] <= 0:
            raise ValueError(f"{path.name} : {key} doit être positif et fini")
    steps = int(params["timemax"] / params["dt"])
    if steps < 1:
        raise ValueError(f"{path.name} : au moins un pas de temps requis")
    return config, order, repetitions, cadence, steps


def estimate_all(folder):
    rows = []
    path = folder / "compare_config.json"
    config = read_config(path)[0]
    _, _, sizes = estimate(path, ["S", "M", "L"], ["snapshots", "sismograms"])
    rows.append({"config": path.name,
                 "snapshots_bytes": sum(row["snapshots_bytes"] for row in sizes),
                 "sismograms_bytes": sum(row["sismograms_bytes"] for row in sizes),
                 "peak_bytes": max(raw_run_bytes(config, row["size"], kind,
                                                 min(config["precision_intervals"]),
                                                 folder / config["receivers"])
                                   for row in sizes for kind in ("snapshots", "sismograms"))})

    for name in ("strong_scaling_config.json", "weak_scaling_config.json"):
        path = folder / name
        config, order, repetitions, cadence, steps = read_config(path)
        if name == "strong_scaling_config.json":
            size = positive_integer(config["size"], "size")
            threads = config["threads"]
            if not isinstance(threads, list) or len(threads) < 2:
                raise ValueError("threads doit contenir au moins deux valeurs")
            for p in threads:
                positive_integer(p, "threads")
            if threads != sorted(set(threads)) or threads[0] != 1:
                raise ValueError("Threads distincts croissants, avec référence à 1")
            node_sum = (order * size + 1) ** 3 * len(threads)
            largest_nodes = (order * size + 1) ** 3
            cases = len(threads)
        else:
            cases_list = config["cases"]
            if not isinstance(cases_list, list) or len(cases_list) < 2:
                raise ValueError("cases doit contenir au moins deux configurations")
            node_sum = 0
            largest_nodes = 0
            base_elements = None
            previous = 0
            for case in cases_list:
                p = positive_integer(case["threads"], "threads")
                if p <= previous:
                    raise ValueError("Threads weak scaling distincts et croissants requis")
                previous = p
                dimensions = [positive_integer(case[axis], axis) for axis in ("ex", "ey", "ez")]
                elements = math.prod(dimensions)
                if base_elements is None:
                    if p != 1:
                        raise ValueError("La référence weak scaling doit utiliser un thread")
                    base_elements = elements
                if elements != p * base_elements:
                    raise ValueError("Weak scaling : éléments par thread non constants")
                nodes = math.prod(order * n + 1 for n in dimensions)
                node_sum += nodes
                largest_nodes = max(largest_nodes, nodes)
            cases = len(cases_list)
        receivers = count_receivers((folder / config["receivers"]).resolve())
        # One ad-hoc snapshot campaign and one ad-hoc signal campaign per case.
        # Their in-situ counterparts produce no raw binary output.
        fields = (steps + cadence - 1) // cadence
        rows.append({"config": name,
                     "snapshots_bytes": snapshot_storage_bytes(node_sum, fields) * repetitions,
                     "sismograms_bytes": cases * repetitions * (8 + 4 * steps * (receivers + 1)),
                     "peak_bytes": max(snapshot_storage_bytes(largest_nodes, fields),
                                       8 + 4 * steps * (receivers + 1))})

    path = folder / "cpu_gpu_config.json"
    read_config(path)
    # Both extrema paths run in-situ: JSON/CSV/figures, but no raw snapshots/signals.
    rows.append({"config": path.name, "snapshots_bytes": 0, "sismograms_bytes": 0,
                 "peak_bytes": 0})
    return rows


def main():
    try:
        if len(sys.argv) != 1:
            raise ValueError("Ce script s'utilise sans argument")
        rows = estimate_all(Path(__file__).resolve().parent)
        print("Volumes écrits cumulés et pic brut temporaire avec nettoyage après chaque essai")
        print(f"{'Configuration':31} {'Snapshots (GB)':>15} {'Sismos (GB)':>14} "
              f"{'Écrit total (GB)':>17} {'Pic brut (GB)':>15}")
        snapshots = signals = 0
        for row in rows:
            snapshots += row["snapshots_bytes"]
            signals += row["sismograms_bytes"]
            total = row["snapshots_bytes"] + row["sismograms_bytes"]
            print(f"{row['config']:31} {row['snapshots_bytes']/1e9:15.5f} "
                  f"{row['sismograms_bytes']/1e9:14.5f} {total/1e9:17.5f} "
                  f"{row['peak_bytes']/1e9:15.5f}")
        print(f"{'SOMME DES QUATRE CONFIGS':31} {snapshots/1e9:15.5f} "
              f"{signals/1e9:14.5f} {(snapshots+signals)/1e9:17.5f}")
        print("GB = 1 000 000 000 octets. Répétitions et sorties de précision incluses.")
        print("Volumes snapshots : mesh.bin écrit une fois par essai, puis pressions float32 seules.")
        peak = max(row["peak_bytes"] for row in rows)
        print(f"PIC BRUT GLOBAL : {peak/1e9:.5f} GB (maximum, pas somme des pics).")
        print("Après chaque essai validé et ses mesures sauvegardées : suppression des binaires uniquement.")
        print("À la fin : CSV, JSON, logs, configurations et figures conservés ; données brutes supprimées.")
        print("Avant chaque essai : contrôle de build/ + dossiers des campagnes, du brut prévu et de 1 GB de réserve.")
        print("Arrêt si le budget de 20 GB ou l'espace libre ne suffisent pas ; aucune ancienne donnée hors campagne supprimée.")
        print("CPU/GPU : 0 donnée brute ; ses statistiques et figures occupent néanmoins du stockage.")
        print("La colonne Pic brut exclut JSON, CSV, logs, figures et build/ : le contrôle à l'exécution les prend en compte.")
        print("Nombre de pas estimé par int(timemax/dt) ; les arrondis float32 du C++ peuvent le modifier.")
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as error:
        sys.exit(f"Erreur : {error}")


if __name__ == "__main__":
    main()
