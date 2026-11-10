import argparse
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import platform
import shlex
import shutil
from statistics import median
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results"


METRICS = {
    "total_seconds": ("simulation", "total_seconds"),
    "compute_seconds": ("simulation", "compute_seconds"),
    "snapshot_write_seconds": ("simulation", "snapshot_write_seconds"),
    "sismogram_write_seconds": ("simulation", "sismogram_write_seconds"),
    "snapshot_mb": ("simulation", "snapshot_mb"),
    "sismogram_mb": ("simulation", "sismogram_mb"),
    **{
        f"{kind}_{phase}_seconds": ("ad_hoc", "visualization", report_kind, f"{report_phase}_seconds")
        for kind, report_kind in (("snapshots", "snapshots"), ("sismos", "sismograms"))
        for phase, report_phase in (("read", "read"), ("analysis", "preparation"),
                                    ("visualization", "render"))
    },
}


def load_config(path):
    config = json.loads(path.read_text(encoding="utf-8"))
    if set(config) != {"executable", "receivers", "repetitions", "parameters", "sizes"}:
        raise ValueError("Champs attendus : executable, receivers, repetitions, parameters, sizes")
    if type(config["repetitions"]) is not int or config["repetitions"] < 2:
        raise ValueError("repetitions doit être un entier >= 2 pour mesurer la dispersion")
    sizes = config["sizes"]
    if set(sizes) != {"S", "M", "L"} or any(
        type(size) is not int or size <= 0 for size in sizes.values()
    ):
        raise ValueError("S, M et L doivent être des entiers strictement positifs")
    if not sizes["S"] < sizes["M"] < sizes["L"]:
        raise ValueError("Les tailles doivent respecter S < M < L")
    parameters = config["parameters"]
    if set(parameters) != {"order", "lx", "ly", "lz", "dt", "timemax", "snap-time-interval"}:
        raise ValueError("Paramètres attendus : order, lx, ly, lz, dt, timemax, snap-time-interval")
    if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
           for value in parameters.values()):
        raise ValueError("Les paramètres doivent être des nombres finis strictement positifs")
    if type(parameters["order"]) is not int or parameters["order"] not in (1, 2, 3):
        raise ValueError("order doit être 1, 2 ou 3")
    if type(parameters["snap-time-interval"]) is not int:
        raise ValueError("snap-time-interval doit être un entier")
    if parameters["timemax"] < parameters["dt"]:
        raise ValueError("timemax doit permettre au moins un pas de temps")
    for key in ("executable", "receivers"):
        target = (path.parent / config[key]).resolve()
        if not target.is_file():
            raise ValueError(f"Fichier introuvable : {target}")
        config[key] = str(target)
    if not os.access(config["executable"], os.X_OK):
        raise ValueError("Le programme indiqué dans executable n'est pas exécutable")
    return config


def environment(executable):
    """Conserver la machine et les réglages de compilation, sans les modifier."""
    info = {
        "machine": platform.node(),
        "system": platform.platform(),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "build": {},
        "gpu": None,
    }
    cache = Path(executable).parent.parent / "CMakeCache.txt"
    if cache.is_file():
        keys = {"CMAKE_BUILD_TYPE", "USE_KOKKOS", "USE_VECTOR",
                "ENABLE_CUDA", "Kokkos_ENABLE_OPENMP"}
        for line in cache.read_text(encoding="utf-8").splitlines():
            key = line.split(":", 1)[0]
            if key in keys and "=" in line:
                info["build"][key] = line.split("=", 1)[1]
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        info["gpu"] = result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass  # Le TP autorise aussi le CPU si aucun GPU n'est disponible.
    return info


def run_command(command, directory, name):
    log_path = directory / f"{name}.log"
    with log_path.open("w", encoding="utf-8") as log:
        log.write(shlex.join(command) + "\n")
        log.flush()
        try:
            subprocess.run(command, cwd=directory, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"Échec de {name} ; consulter {log_path}") from error


def read_metrics(path):
    report = json.loads(path.read_text(encoding="utf-8"))
    values = {}
    for name, parts in METRICS.items():
        value = report
        for part in parts:
            value = value[part]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Mesure absente ou invalide : {name} dans {path}")
        values[name] = value
    if values["compute_seconds"] <= 0:
        raise ValueError(f"Temps de calcul nul dans {path}")
    # E/S = écritures de simulation + lectures Python ; pas l'analyse ni le rendu.
    values["io_compute_ratio"] = sum(values[name] for name in (
        "snapshot_write_seconds", "sismogram_write_seconds",
        "snapshots_read_seconds", "sismos_read_seconds",
    )) / values["compute_seconds"]
    return values


def summarize(records, config, output):
    summary = {}
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.writer(target)
        writer.writerow(["size", "nodes", "metric", "median", "min", "max", "unit"])
        for case in ("S", "M", "L"):
            nodes = (config["sizes"][case] * config["parameters"]["order"] + 1) ** 3
            summary[case] = {}
            for name in records[case][0]:
                values = [record[name] for record in records[case]]
                stats = (median(values), min(values), max(values))
                summary[case][name] = stats
                unit = "MB" if name.endswith("_mb") else (
                    "ratio" if name == "io_compute_ratio" else "s"
                )
                writer.writerow([case, nodes, name, *stats, unit])
    return summary


def plot_summary(summary, config, output, plt):
    nodes = [(config["sizes"][case] * config["parameters"]["order"] + 1) ** 3
             for case in ("S", "M", "L")]
    groups = (
        ("times.svg", [name for name in METRICS
                       if name.endswith("_seconds") and name != "total_seconds"],
         "Temps par étape (s)"),
        ("volumes.svg", ["snapshot_mb", "sismogram_mb"], "Volume écrit (MB)"),
        ("io_compute.svg", ["io_compute_ratio"], "Temps E/S / temps de calcul"),
    )
    for filename, names, ylabel in groups:
        fig, ax = plt.subplots(figsize=(9, 5))
        for name in names:
            stats = [summary[case][name] for case in ("S", "M", "L")]
            middle = [row[0] for row in stats]
            errors = [[row[0] - row[1] for row in stats],
                      [row[2] - row[0] for row in stats]]
            ax.errorbar(nodes, middle, yerr=errors, marker="o", capsize=3, label=name)
        ax.set(xlabel="Nombre de nœuds", ylabel=ylabel)
        ax.set_xscale("log")
        if filename == "times.svg":
            ax.set_yscale("log")
        ax.grid(True)
        ax.legend(fontsize="small")
        fig.tight_layout()
        fig.savefig(output / filename)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    try:
        config_path = args.config.resolve()
        config = load_config(config_path)
        # Vérifier la dépendance existante avant de lancer une simulation coûteuse.
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        output = RESULTS_DIR / datetime.now().strftime("%Y%m%d-%H%M%S")
        output.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(config["receivers"], output / "receivers.txt")
        config["receivers"] = str(output / "receivers.txt")
        (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        (output / "environment.json").write_text(
            json.dumps(environment(config["executable"]), indent=2) + "\n", encoding="utf-8"
        )
        visualizer = str(Path(__file__).resolve().parents[1] / "visualisation" / "visualize.py")
        records = {case: [] for case in ("S", "M", "L")}
        for case in ("S", "M", "L"):
            size = config["sizes"][case]
            for repeat in range(1, config["repetitions"] + 1):
                directory = output / case / f"run_{repeat:02d}"
                directory.mkdir(parents=True)
                print(f"{case} : essai {repeat}/{config['repetitions']} — {directory}", flush=True)
                command = [config["executable"], "--ex", str(size), "--ey", str(size),
                           "--ez", str(size), "--mesh", "cartesian",
                           "--method", "sem", "--implem", "makutu"]
                for name, value in config["parameters"].items():
                    command.extend([f"--{name}", str(value)])
                command.extend(["--recepteur", config["receivers"],
                                "--is-snapshots", "--snap-folder", "snapshots"])
                run_command(command, directory, "simulation")
                run_command([sys.executable, visualizer, "snapshots", "snapshots",
                             "--out-dir", "figures"], directory, "snapshots")
                run_command([sys.executable, visualizer, "sismos", "sismos.bin",
                             "--output", "sismos.svg"], directory, "sismos")
                records[case].append(read_metrics(directory / "metrics.json"))
        summary = summarize(records, config, output)
        plot_summary(summary, config, output, plt)
        print(f"Résultats, synthèse et figures : {output}")
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError) as error:
        parser.exit(1, f"Erreur : {error}\n")


if __name__ == "__main__":
    main()
