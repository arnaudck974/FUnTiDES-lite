#!/usr/bin/env python3
"""Measure fixed-problem strong scaling of the ad-hoc and in-situ workflows."""

import argparse
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
from statistics import median
import subprocess
import sys
from time import perf_counter

from benchmark import RESULTS_DIR, environment
from compare import command, read_run, remove_raw_output, write_csv
from estimate_tp2_storage import check_storage_budget
from estimate_storage import snapshot_storage_bytes


KINDS = ("snapshots", "sismograms")
MODES = ("ad-hoc", "in-situ")
LIBRARY_THREADS = dict.fromkeys((
    "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
), "1")


def load_config(path):
    config = json.loads(path.read_text(encoding="utf-8"))
    if set(config) != {"executable", "receivers", "size", "threads", "repetitions", "parameters"}:
        raise ValueError("Champs requis : executable, receivers, size, threads, repetitions, parameters")
    if type(config["size"]) is not int or config["size"] <= 0:
        raise ValueError("size doit être un entier positif : ex = ey = ez = size")
    threads = config["threads"]
    if (not isinstance(threads, list) or len(threads) < 2 or
            any(type(p) is not int or p <= 0 for p in threads) or
            threads != sorted(set(threads)) or threads[0] != 1):
        raise ValueError("threads : au moins deux entiers distincts croissants, avec référence à 1")
    if type(config["repetitions"]) is not int or config["repetitions"] < 2:
        raise ValueError("repetitions doit être un entier >= 2")
    params = config["parameters"]
    if set(params) != {"order", "lx", "ly", "lz", "dt", "timemax", "snap-time-interval"}:
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


def cpu_resources():
    """Count physical cores within the process affinity mask when Linux exposes them."""
    cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(os.cpu_count() or 1))
    cores = set()
    try:
        for cpu in cpus:
            topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
            cores.add(((topology / "physical_package_id").read_text().strip(),
                       (topology / "core_id").read_text().strip()))
    except OSError:
        cores = set()
    model = None
    try:
        model = next((line.split(":", 1)[1].strip()
                      for line in Path("/proc/cpuinfo").read_text().splitlines()
                      if line.startswith("model name")), None)
    except OSError:
        pass
    return {"affinity_cpus": cpus, "physical_cores": len(cores) or None, "cpu_model": model}


def process_environment(threads):
    settings = os.environ.copy()
    settings.update(LIBRARY_THREADS)
    settings.update(OMP_NUM_THREADS=str(threads), KOKKOS_NUM_THREADS=str(threads),
                    OMP_THREAD_LIMIT=str(threads), OMP_DYNAMIC="FALSE",
                    OMP_PROC_BIND="spread", OMP_PLACES="cores")
    return settings


def check_runtime(text, threads):
    backend = re.findall(r"^Kokkos execution space: (\S+)\s*$", text, re.MULTILINE)
    actual = re.findall(r"^Kokkos host threads: (\d+)\s*$", text, re.MULTILINE)
    if backend != ["OpenMP"]:
        raise ValueError("Cette expérience exige Kokkos/OpenMP ; recompiler si le backend est absent")
    if actual != [str(threads)]:
        raise ValueError(f"Pool Kokkos différent des {threads} threads demandés ; recompiler ou vérifier les ressources")


def run_logged(args, directory, name, settings):
    log_path = directory / f"{name}.log"
    with log_path.open("w", encoding="utf-8") as log:
        log.write(shlex.join(args) + "\n")
        log.flush()
        try:
            subprocess.run(args, cwd=directory, env=settings, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"Échec de {name} ; consulter {log_path}") from error


def execute(config, kind, mode, threads, directory, receivers, analyzer):
    size = config["size"]
    interval = config["parameters"]["snap-time-interval"]
    check_storage_budget(config, size, kind, mode, interval, directory, receivers)
    directory.mkdir()
    args = command(config, size, kind, mode, interval, receivers)
    args.append(f"--kokkos-num-threads={threads}")
    simulation_env = process_environment(threads)
    analysis_env = process_environment(1)
    # The same wall clock covers every child process of the workflow.
    start = perf_counter()
    run_logged(args, directory, "simulation", simulation_env)
    if mode == "ad-hoc":
        args = [sys.executable, str(analyzer)]
        args.extend(["--snapshots", "snapshots", "--dt", str(config["parameters"]["dt"])]
                    if kind == "snapshots" else ["--sismos", "sismos.bin"])
        run_logged(args, directory, "analysis", analysis_env)
    elapsed = perf_counter() - start
    # Reporting and metadata checks are outside the measured workflow.
    check_runtime((directory / "simulation.log").read_text(encoding="utf-8"), threads)
    return read_run(directory, kind, mode, elapsed, size, config, interval)


def summarize(records):
    groups = {}
    for row in records:
        key = row["data"], row["mode"], row["threads"]
        groups.setdefault(key, []).append(row["workflow_seconds"])
    times = {key: median(values) for key, values in groups.items()}
    return [{"data": kind, "mode": mode, "threads": p, "seconds": seconds,
             "speedup": times[kind, mode, 1] / seconds,
             "efficiency": 100 * times[kind, mode, 1] / (p * seconds)}
            for (kind, mode, p), seconds in sorted(times.items())]


def plot_results(summary, output, config, plt):
    for kind, title in (("snapshots", "Snapshots"), ("sismograms", "Sismogrammes")):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
        for mode, color, style, marker in (("ad-hoc", "#2878b5", "-", "o"),
                                            ("in-situ", "#e58525", "--", "^")):
            rows = sorted((r for r in summary if r["data"] == kind and r["mode"] == mode),
                          key=lambda r: r["threads"])
            for ax, metric in zip(axes, ("seconds", "speedup", "efficiency")):
                ax.plot([r["threads"] for r in rows], [r[metric] for r in rows],
                        color=color, linestyle=style, marker=marker, label=mode)
        threads = config["threads"]
        axes[1].plot(threads, threads, ":", color="gray", label="Idéal S(p) = p")
        axes[2].axhline(100, linestyle=":", color="gray", label="Idéal : 100 %")
        for ax, label in zip(axes, ("Temps médian du workflow (s)", "Accélération T(1) / T(p)",
                                   "Efficacité (%)")):
            ax.set(xlabel="Nombre de threads CPU", ylabel=label)
            ax.set_xticks(threads)
            ax.set_ylim(bottom=0)
            ax.grid(alpha=.25)
            ax.legend(handlelength=3)
        cadence = (f"mêmes champs, k={config['parameters']['snap-time-interval']}"
                   if kind == "snapshots" else "mêmes récepteurs, chaque pas")
        fig.suptitle(f"{title} — strong scaling Kokkos/OpenMP, N={config['size']} fixe\n"
                     f"{config['repetitions']} répétitions, médianes ; {cadence}")
        fig.tight_layout()
        fig.savefig(output / f"strong_scaling_{kind}.svg")
        plt.close(fig)


def run_campaign(config_path, config, info, plt):
    output = RESULTS_DIR / "strong-scaling"
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(config_path, output / "strong_scaling_config.json")
    shutil.copyfile(config["receivers"], output / "receivers.txt")
    (output / "environment.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    analyzer = Path(__file__).resolve().parents[1] / "analyze.py"
    records = []
    signatures = {}
    repetitions = config["repetitions"]
    count = len(config["threads"]) * repetitions * len(KINDS) * len(MODES)
    print(f"N={config['size']} fixe ; threads={config['threads']} ; {count} simulations.", flush=True)
    samples = int(config["parameters"]["timemax"] / config["parameters"]["dt"])
    fields = len(range(0, samples, config["parameters"]["snap-time-interval"]))
    nodes = (config["size"] * config["parameters"]["order"] + 1) ** 3
    volume = snapshot_storage_bytes(nodes, fields) * repetitions * len(config["threads"])
    print(f"Snapshots et maillages bruts écrits puis supprimés : environ {volume / 1e9:.2f} GB.", flush=True)
    # Reverse the thread sweep on alternate repetitions to limit order bias.
    for kind in KINDS:
        for repeat in range(1, repetitions + 1):
            threads = config["threads"] if repeat % 2 else list(reversed(config["threads"]))
            for p in threads:
                modes = MODES if repeat % 2 else tuple(reversed(MODES))
                for mode in modes:
                    directory = output / f"{kind}-p{p}-run{repeat:02d}-{mode}"
                    print(f"{kind} ; {p} threads ; essai {repeat}/{repetitions} ; {mode}", flush=True)
                    values, stats, actual_samples = execute(config, kind, mode, p, directory,
                                                           output / "receivers.txt", analyzer)
                    if values["workflow_seconds"] <= 0:
                        raise ValueError("Temps du workflow nul")
                    keys = ("timestep_index", "node_count") if kind == "snapshots" else ("receiver_index", "sample_count")
                    rows = stats[kind]["per_snapshot" if kind == "snapshots" else "per_receiver"]
                    signature = actual_samples, tuple(tuple(row[key] for key in keys) for row in rows)
                    if signatures.setdefault(kind, signature) != signature:
                        raise ValueError("Les essais n'analysent pas les mêmes pas, nœuds ou récepteurs")
                    records.append({"data": kind, "elements": config["size"], "threads": p,
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
        resources = cpu_resources()
        limit = resources["physical_cores"] or len(resources["affinity_cpus"])
        if max(config["threads"]) > limit:
            raise ValueError(f"threads dépasse les {limit} cœurs disponibles détectés ; adapter la configuration")
        if resources["physical_cores"] is None:
            print("Topologie physique inconnue : vérifier les cœurs disponibles avant d'interpréter les résultats.")
        info = environment(config["executable"])
        if info["build"].get("CMAKE_BUILD_TYPE") != "Release":
            raise ValueError("Compilation Release non confirmée dans CMakeCache.txt ; configurer votre build en Release")
        # Probe initialization only; no physics or timed workflow is run here.
        for p in config["threads"]:
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
                    matplotlib=matplotlib.__version__, OMP_NUM_THREADS=config["threads"],
                    simulation_environment={str(p): recorded_settings(p) for p in config["threads"]},
                    analysis_environment=recorded_settings(1))
        run_campaign(config_path, config, info, plt)
    except (OSError, ValueError, KeyError, TypeError, IndexError, RuntimeError, ImportError,
            subprocess.SubprocessError) as error:
        parser.exit(1, f"Erreur : {error}\n")


if __name__ == "__main__":
    main()
