#!/usr/bin/env python3
"""Measure OpenMP weak scaling with a constant number of elements per thread."""

import argparse
import json
import math
import os
from pathlib import Path
import shutil
from statistics import median
import subprocess
import sys

from benchmark import RESULTS_DIR, environment
from compare import remove_raw_output, write_csv
from estimate_storage import snapshot_storage_bytes
from strong_scaling import (
    KINDS, MODES, LIBRARY_THREADS, check_runtime, cpu_resources, execute,
    process_environment,
)


AXES = ("ex", "ey", "ez")
LENGTHS = ("lx", "ly", "lz")


def load_config(path):
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or set(config) != {"executable", "receivers", "cases", "repetitions", "parameters"}:
        raise ValueError("Champs requis : executable, receivers, cases, repetitions, parameters")
    cases = config["cases"]
    if not isinstance(cases, list) or len(cases) < 2:
        raise ValueError("cases doit contenir au moins deux configurations")
    previous = 0
    base_elements = None
    for case in cases:
        if (not isinstance(case, dict) or set(case) != {"threads", *AXES} or
                any(type(v) is not int or v <= 0 for v in case.values())):
            raise ValueError("Chaque cas exige threads, ex, ey, ez : entiers positifs")
        p = case["threads"]
        if p <= previous or (previous == 0 and p != 1):
            raise ValueError("Threads distincts croissants, avec référence à 1")
        elements = math.prod(case[axis] for axis in AXES)
        if base_elements is None:
            base_elements = elements
        if elements != p * base_elements:
            raise ValueError("Le nombre d'éléments par thread doit rester exactement constant")
        previous = p
    if type(config["repetitions"]) is not int or config["repetitions"] < 2:
        raise ValueError("repetitions doit être un entier >= 2")
    params = config["parameters"]
    if not isinstance(params, dict) or set(params) != {"order", *LENGTHS, "dt", "timemax", "snap-time-interval"}:
        raise ValueError("Paramètres requis : order, lx, ly, lz, dt, timemax, snap-time-interval")
    if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in params.values()):
        raise ValueError("Les paramètres doivent être positifs et finis")
    if type(params["order"]) is not int or params["order"] not in (1, 2, 3):
        raise ValueError("order doit être 1, 2 ou 3")
    if type(params["snap-time-interval"]) is not int or params["timemax"] < params["dt"]:
        raise ValueError("Intervalle entier et au moins un pas de temps requis")
    for name in ("executable", "receivers"):
        target = (path.parent / config[name]).resolve()
        if not target.is_file():
            raise ValueError(f"Fichier introuvable : {target}")
        config[name] = str(target)
    if not os.access(config["executable"], os.X_OK):
        raise ValueError("semproxy n'est pas exécutable")
    return config


def case_config(config, case):
    """Grow physical lengths with the mesh to preserve each element's dimensions."""
    base = config["cases"][0]
    lengths = {length: config["parameters"][length] * case[axis] / base[axis]
               for length, axis in zip(LENGTHS, AXES)}
    return {**config, "size": tuple(case[axis] for axis in AXES),
            "parameters": {**config["parameters"], **lengths}}


def node_count(config, case):
    return math.prod(config["parameters"]["order"] * case[axis] + 1 for axis in AXES)


def summarize(records):
    groups = {}
    for row in records:
        key = row["data"], row["mode"], row["threads"]
        groups.setdefault(key, []).append(row["workflow_seconds"])
    times = {key: median(values) for key, values in groups.items()}
    # Weak efficiency uses a fixed amount of work per thread: no extra division by p.
    return [{"data": kind, "mode": mode, "threads": p, "seconds": seconds,
             "efficiency": 100 * times[kind, mode, 1] / seconds}
            for (kind, mode, p), seconds in sorted(times.items())]


def plot_results(summary, output, config, plt):
    threads = [case["threads"] for case in config["cases"]]
    ticks = [f"{case['threads']}\n{case['ex']}×{case['ey']}×{case['ez']}" for case in config["cases"]]
    elements = math.prod(config["cases"][0][axis] for axis in AXES)
    for kind, title in (("snapshots", "Snapshots"), ("sismograms", "Sismogrammes")):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
        for mode, color, style, marker in (("ad-hoc", "#2878b5", "-", "o"),
                                          ("in-situ", "#e58525", "--", "^")):
            rows = sorted((r for r in summary if r["data"] == kind and r["mode"] == mode),
                          key=lambda r: r["threads"])
            for ax, metric in zip(axes, ("seconds", "efficiency")):
                ax.plot([r["threads"] for r in rows], [r[metric] for r in rows],
                        color=color, linestyle=style, marker=marker, label=mode)
        axes[1].axhline(100, linestyle=":", color="gray", label="Idéal : 100 %")
        for ax, label in zip(axes, ("Temps médian du workflow (s)", "Efficacité faible : 100 × T(1)/T(p) (%)")):
            ax.set(xlabel="Threads CPU / maillage (ex×ey×ez)", ylabel=label)
            ax.set_xticks(threads)
            ax.set_xticklabels(ticks)
            ax.set_ylim(bottom=0)
            ax.grid(alpha=.25)
            ax.legend(handlelength=3)
        cadence = (f"mêmes instants entre approches, k={config['parameters']['snap-time-interval']}"
                   if kind == "snapshots" else "récepteurs fixes, chaque pas")
        fig.suptitle(f"{title} — weak scaling Kokkos/OpenMP, {elements} éléments/thread\n"
                     f"{config['repetitions']} répétitions, médianes ; {cadence}")
        fig.tight_layout()
        fig.savefig(output / f"weak_scaling_{kind}.svg")
        plt.close(fig)


def run_campaign(config_path, config, info, plt):
    output = RESULTS_DIR / "weak-scaling"
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(config_path, output / "weak_scaling_config.json")
    shutil.copyfile(config["receivers"], output / "receivers.txt")
    (output / "environment.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    analyzer = Path(__file__).resolve().parents[1] / "analyze.py"
    repetitions = config["repetitions"]
    count = len(config["cases"]) * repetitions * len(KINDS) * len(MODES)
    print(f"Weak scaling : {count} simulations ; récepteurs fixes.", flush=True)
    samples = int(config["parameters"]["timemax"] / config["parameters"]["dt"])
    fields = len(range(0, samples, config["parameters"]["snap-time-interval"]))
    nodes = sum(node_count(config, case) for case in config["cases"])
    volume = snapshot_storage_bytes(nodes, fields) * repetitions
    print(f"Snapshots et maillages bruts écrits puis supprimés : environ {volume / 1e9:.2f} GB.", flush=True)
    records = []
    signatures = {}
    for kind in KINDS:
        for repeat in range(1, repetitions + 1):
            cases = config["cases"] if repeat % 2 else list(reversed(config["cases"]))
            for case in cases:
                current = case_config(config, case)
                p = case["threads"]
                modes = MODES if repeat % 2 else tuple(reversed(MODES))
                for mode in modes:
                    directory = output / f"{kind}-p{p}-run{repeat:02d}-{mode}"
                    print(f"{kind} ; {p} threads ; maillage={current['size']} ; essai {repeat}/{repetitions} ; {mode}", flush=True)
                    values, stats, actual_samples = execute(current, kind, mode, p, directory,
                                                           output / "receivers.txt", analyzer)
                    if values["workflow_seconds"] <= 0:
                        raise ValueError("Temps du workflow nul")
                    rows = stats[kind]["per_snapshot" if kind == "snapshots" else "per_receiver"]
                    # Mesh sizes change; the temporal sampling and receivers must not.
                    signature = actual_samples, tuple(row["timestep_index" if kind == "snapshots" else "receiver_index"] for row in rows)
                    if signatures.setdefault(kind, signature) != signature:
                        raise ValueError("Les essais n'analysent pas les mêmes pas ou récepteurs")
                    elements = math.prod(current["size"])
                    records.append({"data": kind, **case,
                                    **{key: current["parameters"][key] for key in LENGTHS},
                                    "element_count": elements, "elements_per_thread": elements // p,
                                    "node_count": node_count(config, case),
                                    "nodes_per_thread": node_count(config, case) / p,
                                    "mode": mode, "repeat": repeat, "samples": actual_samples, **values})
                    write_csv(output / "runs.csv", records)
                    if mode == "ad-hoc":
                        remove_raw_output(directory, kind)
    figures = output / "figures"
    figures.mkdir()
    plot_results(summarize(records), figures, config, plt)
    print(f"Campagne terminée : {output}\nFigures : {figures}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    try:
        config_path = args.config.resolve()
        config = load_config(config_path)
        threads = [case["threads"] for case in config["cases"]]
        resources = cpu_resources()
        limit = resources["physical_cores"] or len(resources["affinity_cpus"])
        if max(threads) > limit:
            raise ValueError(f"threads dépasse les {limit} cœurs disponibles détectés ; adapter la configuration")
        if resources["physical_cores"] is None:
            print("Topologie physique inconnue : vérifier les cœurs disponibles avant d'interpréter les résultats.")
        info = environment(config["executable"])
        if info["build"].get("CMAKE_BUILD_TYPE") != "Release":
            raise ValueError("Compilation Release non confirmée dans CMakeCache.txt ; configurer votre build en Release")
        for p in threads:
            probe = subprocess.run([config["executable"], f"--kokkos-num-threads={p}", "--help"],
                                   env=process_environment(p), capture_output=True, text=True, check=True)
            check_runtime(probe.stdout, p)
            if "--outputs" not in probe.stdout:
                raise ValueError("Recompiler semproxy : --outputs est absent")
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        def recorded_settings(p):
            return {key: value for key, value in process_environment(p).items()
                    if key.startswith(("OMP_", "KOKKOS_")) or key in LIBRARY_THREADS}
        info.update(resources)
        info.update(executable=config["executable"], python=sys.version, numpy=np.__version__,
                    matplotlib=matplotlib.__version__, OMP_NUM_THREADS=threads,
                    simulation_environment={str(p): recorded_settings(p) for p in threads},
                    analysis_environment=recorded_settings(1))
        run_campaign(config_path, config, info, plt)
    except (OSError, ValueError, KeyError, TypeError, IndexError, RuntimeError, ImportError,
            subprocess.SubprocessError) as error:
        parser.exit(1, f"Erreur : {error}\n")


if __name__ == "__main__":
    main()
