"""Run and compare real ad-hoc and in-situ TP2 measurements."""

import argparse
import csv
from html import escape
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
from statistics import fmean, median
import sys
from time import perf_counter

from benchmark import environment, run_command
from estimate_tp2_storage import check_storage_budget
from estimate_storage import estimate


FIELDS = (
    "minimum_pressure", "maximum_pressure", "maximum_absolute_pressure",
    "mean_pressure", "pressure_variance",
)
OPERATIONS = ("write", "read", "analysis")
COLORS = {"write": "#2878b5", "read": "#2b9348", "analysis": "#e58525"}
STYLES = {"ad-hoc": "-", "in-situ": "--"}
MARKERS = {"ad-hoc": "o", "in-situ": "^"}


def load_config(path, cpu_gpu=False):
    data = json.loads(path.read_text(encoding="utf-8"))
    expected = {"executable", "repetitions", "parameters", "sizes"}
    if not cpu_gpu:
        expected.update(("receivers", "precision_intervals"))
    if set(data) != expected:
        raise ValueError("Champs requis : " + ", ".join(sorted(expected)))
    if type(data["repetitions"]) is not int or data["repetitions"] < 2:
        raise ValueError("repetitions doit être un entier >= 2")
    sizes = data["sizes"]
    if set(sizes) != {"S", "M", "L"}:
        raise ValueError("sizes doit contenir S, M et L")
    previous = 0
    for case in ("S", "M", "L"):
        values = sizes[case]
        if (not isinstance(values, list) or not values or
                any(type(n) is not int or n <= previous for n in values) or
                values != sorted(set(values))):
            raise ValueError("Tailles positives, distinctes et croissantes dans S, M, L")
        previous = values[-1]
    params = data["parameters"]
    if set(params) != {"order", "lx", "ly", "lz", "dt", "timemax", "snap-time-interval"}:
        raise ValueError("Paramètres requis : order, lx, ly, lz, dt, timemax, snap-time-interval")
    if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
           for v in params.values()):
        raise ValueError("Les paramètres doivent être des nombres positifs et finis")
    if type(params["order"]) is not int or params["order"] not in (1, 2, 3):
        raise ValueError("order doit être 1, 2 ou 3")
    if type(params["snap-time-interval"]) is not int or params["timemax"] < params["dt"]:
        raise ValueError("Intervalle entier et au moins un pas de temps requis")
    if not cpu_gpu:
        intervals = data["precision_intervals"]
        if (not isinstance(intervals, list) or not intervals or
                any(type(k) is not int or k <= 0 for k in intervals) or
                len(intervals) != len(set(intervals))):
            raise ValueError("precision_intervals : liste d'entiers positifs distincts")
        if 1 not in intervals or params["snap-time-interval"] not in intervals:
            raise ValueError("precision_intervals doit contenir 1 et snap-time-interval")
    for name in (("executable",) if cpu_gpu else ("executable", "receivers")):
        target = (path.parent / data[name]).resolve()
        if not target.is_file():
            raise ValueError(f"Fichier introuvable : {target}")
        data[name] = str(target)
    if not os.access(data["executable"], os.X_OK):
        raise ValueError("semproxy n'est pas exécutable")
    return data


def number(value, label):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"Valeur absente ou invalide : {label}")
    return float(value)


def nested(report, *keys):
    value = report
    for key in keys:
        value = value[key]
    return number(value, ".".join(keys))


def read_run(directory, kind, mode, wall_seconds, size, config, interval):
    report = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    sim = report["simulation"]
    if sim.get("outputs") != kind:
        raise ValueError("Recompiler semproxy : outputs dans metrics.json doit correspondre au workflow")
    insitu = mode == "in-situ"
    if (report["in_situ"] is not None) != insitu:
        raise ValueError("Mode incorrect dans metrics.json")
    filename = "statistics_in-situ.json" if insitu else "statistics_ad-hoc.json"
    stats = json.loads((directory / filename).read_text(encoding="utf-8"))
    other = "sismograms" if kind == "snapshots" else "snapshots"
    if stats.get(other) is not None or not stats.get(kind):
        raise ValueError("Les sorties ne sont pas isolées")
    samples = sim["sample_count"]
    if type(samples) is not int or samples <= 0:
        raise ValueError("Nombre de pas physiques invalide")
    rows = stats[kind]["per_snapshot" if kind == "snapshots" else "per_receiver"]
    if not rows:
        raise ValueError("Statistiques vides")
    if kind == "snapshots":
        if [r["timestep_index"] for r in rows] != list(range(0, samples, interval)):
            raise ValueError("Pas analysés incorrects")
        # An integer keeps the existing cubic cases; a triplet supports weak scaling.
        dimensions = (size, size, size) if isinstance(size, int) else size
        nodes = math.prod(n * config["parameters"]["order"] + 1 for n in dimensions)
        if any(r["node_count"] != nodes or not math.isfinite(r["time_seconds"]) for r in rows):
            raise ValueError("Métadonnées de snapshots incorrectes")
    elif any(r["receiver_index"] != i or r["sample_count"] != samples
             for i, r in enumerate(rows)):
        raise ValueError("Métadonnées des récepteurs incorrectes")
    if any(not math.isfinite(r[key]) for r in rows for key in FIELDS):
        raise ValueError("Statistiques non finies")
    prefix = "snapshot" if kind == "snapshots" else "sismogram"
    raw_mb = nested(report, "simulation", prefix + "_mb")
    raw_write = nested(report, "simulation", prefix + "_write_seconds")
    if nested(report, "simulation", ("sismogram" if kind == "snapshots" else "snapshot") + "_mb") != 0:
        raise ValueError("Données brutes de l'autre workflow présentes")
    measurements = {
        "workflow_seconds": number(wall_seconds, "workflow_seconds"),
        "simulation_seconds": nested(report, "simulation", "total_seconds"),
        "compute_seconds": nested(report, "simulation", "compute_seconds"),
        "write_raw_seconds": raw_write,
        "raw_mb": raw_mb,
    }
    timing = report["in_situ"] if insitu else report["ad_hoc"]["statistics"]
    if insitu:
        if raw_mb != 0 or raw_write != 0 or (directory / "sismos.bin").exists() or (directory / "snapshots").exists():
            raise ValueError("Données brutes présentes en in-situ")
        measurements["read_seconds"] = 0.0
    else:
        measurements["read_seconds"] = number(timing[kind]["read_seconds"], "read_seconds")
    measurements["analysis_seconds"] = number(timing[kind]["analysis_seconds"], "analysis_seconds")
    measurements["write_statistics_seconds"] = number(timing["output_file"]["write_seconds"], "write_statistics_seconds")
    measurements["statistics_mb"] = number(timing["output_file"]["size_mb"], "statistics_mb")
    if round(measurements["statistics_mb"] * 1e6) != (directory / filename).stat().st_size:
        raise ValueError("Taille du JSON incohérente")
    measurements["write_seconds"] = raw_write + measurements["write_statistics_seconds"]
    measurements["total_mb"] = raw_mb + measurements["statistics_mb"]
    return measurements, stats, samples


def compare_statistics(a, b, kind, case, size, repeat):
    key = "per_snapshot" if kind == "snapshots" else "per_receiver"
    left, right = a[kind][key], b[kind][key]
    if len(left) != len(right):
        raise ValueError("Nombre de résultats différent")
    result = []
    metadata = ("timestep_index", "node_count") if kind == "snapshots" else ("receiver_index", "sample_count")
    for index, (x, y) in enumerate(zip(left, right)):
        if any(x[name] != y[name] for name in metadata):
            raise ValueError("Instants ou dimensions incompatibles")
        for name in FIELDS:
            va, vb = x[name], y[name]
            if not all(map(math.isfinite, (va, vb))):
                raise ValueError("Résultat non fini")
            result.append({"data": kind, "case": case, "elements": size,
                           "repeat": repeat, "index": index, "statistic": name,
                           "ad_hoc": va, "in_situ": vb, "absolute_error": abs(va-vb)})
    return result


def precision_row(dense, sparse, case, size, interval, samples):
    full = dense["snapshots"]["per_snapshot"]
    rows = sparse["snapshots"]["per_snapshot"]
    if ([r["timestep_index"] for r in full] != list(range(samples)) or
            [r["timestep_index"] for r in rows] != list(range(0, samples, interval))):
        raise ValueError("Référence temporelle incompatible")
    return {"case": case, "elements": size, "interval_steps": interval,
            "analyzed_snapshots": len(rows), "reference_snapshots": len(full),
            "maximum_absolute_pressure": max(r["maximum_absolute_pressure"] for r in rows),
            "reference_maximum_absolute_pressure": max(r["maximum_absolute_pressure"] for r in full),
            "temporal_mean_pressure": fmean(r["mean_pressure"] for r in rows),
            "reference_temporal_mean_pressure": fmean(r["mean_pressure"] for r in full)}


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(records):
    groups = {}
    for record in records:
        key = (record["data"], record["case"], record["elements"], record["mode"])
        groups.setdefault(key, []).append(record)
    return [{"data": kind, "case": case, "elements": size, "mode": mode,
             "repetitions": len(group), "metric": metric,
             "unit": "MB" if metric.endswith("_mb") else "s",
             "median": median(row["measurements"][metric] for row in group)}
            for (kind, case, size, mode), group in sorted(groups.items())
            for metric in group[0]["measurements"]]


def plot_results(summary, output, config, plt):
    labels = {"read_seconds": "Lecture brute", "write_raw_seconds": "Écriture brute",
              "write_statistics_seconds": "Écriture des statistiques", "analysis_seconds": "Calcul des statistiques"}
    colors = dict(zip(labels, ("#2b9348", "#2878b5", "#9467bd", "#e58525")))
    titles = {"snapshots": "Snapshots", "sismograms": "Sismogrammes"}
    def curve(ax, selected, metric, mode, color, label, log=False):
        rows = sorted((r for r in selected if r["metric"] == metric and r["mode"] == mode
                       and (not log or r["median"] > 0)), key=lambda r: r["elements"])
        if rows:
            ax.plot([r["elements"] for r in rows], [r["median"] for r in rows],
                    linestyle=STYLES[mode], marker=MARKERS[mode], color=color, label=label)
    k = config["parameters"]["snap-time-interval"]
    for kind, title in titles.items():
        selected = [r for r in summary if r["data"] == kind]
        if not selected:
            continue
        cadence = f"mêmes champs analysés, k={k}" if kind == "snapshots" else "mêmes récepteurs, chaque pas"
        fig, ax = plt.subplots(figsize=(12, 6))
        for metric, label in labels.items():
            for mode in STYLES:
                curve(ax, selected, metric, mode, colors[metric], f"{label} — {mode}", log=True)
        ax.set_yscale("log")
        ax.set(xlabel="Éléments par direction N", ylabel="Temps médian sur une simulation (s, log)",
               title=f"{title} : coûts détaillés — {cadence}")
        ax.grid(alpha=.25, which="both"); ax.legend(fontsize=8, handlelength=4, ncol=2)
        fig.text(.5,.02,"Ad-hoc : continu + ronds ; in-situ : discontinu + triangles.\n"
                 "Lecture et écriture brutes in-situ : 0 s, hors axe log. Lecture en mémoire incluse dans le calcul.",ha="center",fontsize=9)
        fig.tight_layout(rect=(0,.1,1,1)); fig.savefig(output/f"costs_{kind}.svg"); plt.close(fig)
        fig, ax = plt.subplots(figsize=(10,5))
        for mode, color in (("ad-hoc","#2878b5"),("in-situ","#e58525")):
            curve(ax,selected,"workflow_seconds",mode,color,mode)
        ax.set(xlabel="Éléments par direction N",ylabel="Temps complet médian (s)",
               title=f"Workflow {title.lower()} seul — {cadence}")
        ax.set_ylim(bottom=0); ax.grid(alpha=.25); ax.legend(handlelength=4)
        fig.text(.5,.02,"Simulation + calcul des statistiques + toutes les écritures ; lecture Python en ad-hoc.\n"
                 "Ad-hoc : continu + ronds ; in-situ : discontinu + triangles.",ha="center",fontsize=9)
        fig.tight_layout(rect=(0,.1,1,1)); fig.savefig(output/f"total_{kind}.svg"); plt.close(fig)
    kinds = [kind for kind in titles if any(r["data"]==kind for r in summary)]
    fig, axes = plt.subplots(len(kinds),2,figsize=(13,4*len(kinds)),squeeze=False)
    for row, kind in enumerate(kinds):
        selected=[r for r in summary if r["data"]==kind]
        sizes=sorted({r["elements"] for r in selected})
        for col,(metric,label) in enumerate((("write_seconds","Temps d’écriture (s)"),("total_mb","Volume écrit (MB)"))):
            ax=axes[row,col]
            for mode,offset,color in (("ad-hoc",-.18,"#2878b5"),("in-situ",.18,"#e58525")):
                values=[next(r["median"] for r in selected if r["elements"]==size and r["mode"]==mode and r["metric"]==metric) for size in sizes]
                bars=ax.bar([i+offset for i in range(len(sizes))],values,width=.36,label=mode,color=color)
                ax.bar_label(bars,fmt="%.3g",padding=3,fontsize=8)
            ax.set_xticks(range(len(sizes)),sizes)
            ax.set(xlabel="Éléments par direction N",ylabel=label,title=titles[kind])
            ax.margins(y=.2); ax.legend(); ax.grid(axis="y",alpha=.2)
    fig.suptitle("Écritures : ad-hoc = brut + statistiques ; in-situ = statistiques. Médianes des essais.")
    fig.tight_layout(rect=(0,0,1,.94)); fig.savefig(output/"writes.svg"); plt.close(fig)


def plot_peak(dense, sparse, size, k, output, plt):
    fig,ax=plt.subplots(figsize=(11,5))
    rows=dense["snapshots"]["per_snapshot"]
    ax.plot([r["time_seconds"] for r in rows],[r["maximum_absolute_pressure"] for r in rows],
            linestyle="--",label="In-situ : chaque pas")
    rows=sparse["snapshots"]["per_snapshot"]
    ax.plot([r["time_seconds"] for r in rows],[r["maximum_absolute_pressure"] for r in rows],
            linestyle="None",marker="o",label=f"Ad-hoc : un snapshot tous les {k} pas")
    ax.set(xlabel="Temps physique (s)",ylabel="Maximum spatial de |pression|",title=f"Évolution du pic — N={size}")
    ax.legend(); ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(output/f"peak_evolution_n{size}.svg"); plt.close(fig)


def html_table(headers, rows):
    def cell(value):
        return escape(f"{value:.8g}" if isinstance(value,float) else str(value))
    return ("<table><thead><tr>"+"".join(f"<th>{cell(v)}</th>" for v in headers)+
            "</tr></thead><tbody>"+"".join("<tr>"+"".join(f"<td>{cell(v)}</td>" for v in row)+"</tr>" for row in rows)+"</tbody></table>")


def write_html(output, summary, validation, precision, config):
    groups={}
    for r in summary:
        groups.setdefault((r["data"],r["elements"],r["mode"]),{})[r["metric"]]=r["median"]
    totals=html_table(("Données","N","Mode","Temps complet médian (s)","Brut (MB)","Statistiques (MB)"),
        ([kind,n,mode,m["workflow_seconds"],m["raw_mb"],m["statistics_mb"]] for (kind,n,mode),m in sorted(groups.items())))
    operations=html_table(("Données","N","Mode","Lecture brute (s)","Calcul des statistiques (s)","Écriture brute (s)","Écriture des statistiques (s)"),
        ([kind,n,mode,*[m[key] for key in ("read_seconds","analysis_seconds","write_raw_seconds","write_statistics_seconds")]] for (kind,n,mode),m in sorted(groups.items())))
    quality=[]
    for n in sorted({r["elements"] for r in precision}):
        rows=sorted((r for r in precision if r["elements"]==n),key=lambda r:r["interval_steps"])
        first=rows[0]
        quality.append([n,"in-situ — référence",1,first["reference_snapshots"],first["reference_maximum_absolute_pressure"],first["reference_temporal_mean_pressure"]])
        quality.extend([n,"ad-hoc",r["interval_steps"],r["analyzed_snapshots"],r["maximum_absolute_pressure"],r["temporal_mean_pressure"]] for r in rows)
    snapshots=html_table(("N","Mode","Intervalle k","Champs analysés","Pic absolu de pression","Moyenne temporelle de la moyenne spatiale"),quality)
    names=dict(zip(FIELDS,("Minimum","Maximum","Maximum absolu","Moyenne","Variance")))
    signals=html_table(("N","Récepteur (index 0…)","Statistique","Ad-hoc","In-situ"),
        ([r["elements"],r["index"],names[r["statistic"]],r["ad_hoc"],r["in_situ"]]
         for r in validation if r["data"]=="sismograms" and r["repeat"]==1))
    text=("<!doctype html><html lang='fr'><meta charset='utf-8'><title>Comparaison TP2</title>"
          "<style>body{font-family:sans-serif;margin:2rem}table{border-collapse:collapse;margin:1rem 0}"
          "th,td{border:1px solid #aaa;padding:.5rem;text-align:right}th{background:#eee}</style>"
          "<h1>Comparaison ad-hoc / in-situ</h1>"
          f"<p>{config['repetitions']} essais par taille, type et mode. Champs analysés tous les "
          f"{config['parameters']['snap-time-interval']} pas pour les temps ; récepteurs à chaque pas. "
          "Campagnes indépendantes. Les figures et ce rapport sont générés après les chronomètres.</p>"
          "<h2>1. Temps complet et stockage</h2>"+totals+
          "<h2>2. Détail des opérations — médianes</h2>"+operations+
          "<p>0 signifie une opération absente. Lecture en mémoire incluse dans le calcul. "
          "Écriture brute incluant la préparation de sortie. MB = 1 000 000 octets ; logs et rapports exclus.</p>"
          "<h2>3. Qualité des snapshots selon leur fréquence</h2>"+snapshots+
          "<p>Une exécution par intervalle, hors médianes des temps. La moyenne temporelle est la moyenne "
          "des pressions moyennes spatiales des instants retenus. Ces valeurs sont signées. Les écarts "
          "peuvent inclure des différences numériques entre exécutions. La référence n'est pas une solution physique exacte.</p>"
          "<h2>4. Statistiques des sismogrammes — premier essai</h2>"+signals+
          "<p>La variance est en unités de pression au carré, les autres statistiques en unités de pression. "
          "Les valeurs de chaque essai et les écarts absolus sont dans validation.csv. Aucun seuil d'acceptation. "
          "Python/NumPy et C++/Kokkos peuvent avoir des coûts de calcul différents. "
          "Les transferts GPU/hôte ne sont pas isolés par les chronomètres actuels.</p></html>")
    (output/"statistics.html").write_text(text,encoding="utf-8")


def save_results(output, records, validation, precision):
    write_csv(output/"runs.csv",[{k:v for k,v in r.items() if k!="measurements"}|r["measurements"] for r in records])
    summary=summarize(records)
    write_csv(output/"summary.csv",summary)
    write_csv(output/"validation.csv",validation)
    write_csv(output/"precision.csv",precision)
    return summary


def command(config,size,kind,mode,interval,receivers):
    ex,ey,ez = (size,size,size) if isinstance(size,int) else size
    result=[config["executable"],"--outputs",kind,"--mode",mode,"--ex",str(ex),"--ey",str(ey),"--ez",str(ez),
            "--mesh","cartesian","--method","sem","--implem","makutu"]
    for key,value in config["parameters"].items():
        result.extend([f"--{key}",str(interval if key=="snap-time-interval" else value)])
    if kind=="sismograms":
        result.extend(["--recepteur",str(receivers)])
    elif mode=="ad-hoc":
        result.extend(["--is-snapshots","--snap-folder","snapshots"])
    return result


def remove_raw_output(directory, kind):
    """Call only after validation and CSV persistence, outside all timed regions."""
    folder = directory / "snapshots" if kind == "snapshots" else directory
    if folder.is_symlink():
        raise ValueError(f"Nettoyage refusé : lien symbolique {folder}")
    files = ([folder / "mesh.bin"] + [p for p in folder.glob("snapshot_*.bin")
              if p.stem.removeprefix("snapshot_").isdigit()]
             if kind == "snapshots" else [directory / "sismos.bin"])
    # Check every candidate before deleting anything; keep all reports and logs.
    if any(p.is_symlink() or not p.is_file() for p in files):
        raise ValueError(f"Sortie brute invalide pour le nettoyage : {directory}")
    for path in files:
        path.unlink()


def execute(config,size,kind,mode,interval,directory,receivers,analyzer):
    check_storage_budget(config,size,kind,mode,interval,directory,receivers)
    directory.mkdir()
    start=perf_counter()
    run_command(command(config,size,kind,mode,interval,receivers),directory,"simulation")
    if mode=="ad-hoc":
        args=[sys.executable,str(analyzer)]
        args.extend(["--snapshots","snapshots","--dt",str(config["parameters"]["dt"])] if kind=="snapshots" else ["--sismos","sismos.bin"])
        run_command(args,directory,"analysis")
    elapsed=perf_counter()-start
    return read_run(directory,kind,mode,elapsed,size,config,interval)


def run_extrema_case(config, size, backend, directory):
    interval = config["parameters"]["snap-time-interval"]
    check_storage_budget(config,size,"snapshots","in-situ",interval,directory,None)
    directory.mkdir()
    cmd = command(config, size, "snapshots", "in-situ", interval, None)
    cmd.extend(("--extrema-backend", backend))
    start = perf_counter()
    run_command(cmd, directory, "simulation")
    wall = perf_counter() - start
    _, stats, samples = read_run(directory, "snapshots", "in-situ", wall,
                                 size, config, interval)
    report = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    measured = report["extrema_comparison"]
    if measured["backend"] != backend or "cuda" not in measured["execution_space"].lower():
        raise ValueError("La comparaison CPU/GPU exige une simulation CUDA dans les deux cas")
    nodes = (config["parameters"]["order"] * size + 1) ** 3
    calls = len(stats["snapshots"]["per_snapshot"])
    if measured["calls"] != calls:
        raise ValueError("Nombre de mesures des extrema incohérent")
    values = {key: number(measured[key], key) for key in (
        "prepare_seconds", "compute_seconds", "result_seconds", "total_seconds",
        "input_bytes", "result_bytes")}
    if any(type(measured[key]) is not int for key in ("input_bytes", "result_bytes", "calls")):
        raise ValueError("Compteurs d'octets/appels invalides")
    if backend == "cpu" and (measured["input_bytes"] != calls * nodes * 4 or measured["result_bytes"] != 0):
        raise ValueError("Volume de copie du champ incohérent")
    if backend == "kokkos" and (measured["input_bytes"] != 0 or measured["result_bytes"] <= 0):
        raise ValueError("Volume de récupération GPU incohérent")
    if measured["total_seconds"] < sum(measured[key] for key in ("prepare_seconds", "compute_seconds", "result_seconds")):
        raise ValueError("Chronomètres du bloc incohérents")
    values.update(workflow_seconds=wall, calls=calls, nodes=nodes, samples=samples,
                  execution_space=measured["execution_space"])
    return values, stats


def extrema_summary(records):
    groups = {}
    for row in records:
        groups.setdefault((row["case"], row["elements"], row["backend"]), []).append(row)
    return [{"case": case, "elements": size, "backend": backend,
             "repetitions": len(rows), "metric": metric,
             "median": median(row[metric] for row in rows)}
            for (case, size, backend), rows in sorted(groups.items())
            for metric in ("prepare_seconds", "compute_seconds", "result_seconds",
                           "total_seconds", "input_bytes", "result_bytes", "workflow_seconds")]


def plot_extrema_comparison(summary, output, plt):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (metric, title) in zip(axes, (
        ("compute_seconds", "Calcul des extrema seul"),
        ("total_seconds", "Total du bloc : préparation + calcul + récupération"),
    )):
        for backend, label, color, style, marker in (
            ("cpu", "CPU hôte séquentiel", "#2878b5", "-", "o"),
            ("kokkos", "GPU — réduction Kokkos/CUDA", "#e58525", "--", "^"),
        ):
            rows = sorted((r for r in summary if r["backend"] == backend and r["metric"] == metric),
                          key=lambda r: r["elements"])
            ax.plot([r["elements"] for r in rows], [r["median"] for r in rows],
                    label=label, color=color, linestyle=style, marker=marker)
        ax.set(xlabel="Éléments par direction N", ylabel="Temps cumulé médian (s)", title=title)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.25)
        ax.legend(fontsize=8, handlelength=4)
    fig.suptitle("Simulation GPU dans les deux cas — extrema CPU ou GPU aux mêmes pas")
    fig.tight_layout(rect=(0, 0, 1, .92))
    fig.savefig(output / "cpu_gpu_extrema.svg")
    plt.close(fig)


def write_extrema_html(output, summary, config):
    groups = {}
    for row in summary:
        groups.setdefault((row["elements"], row["backend"]), {})[row["metric"]] = row["median"]
    table = html_table(
        ("N", "Calcul des extrema", "Préparation (s)", "Calcul (s)", "Récupération (s)",
         "Total du bloc (s)", "Entrée copiée (octets)", "Résultat copié (octets)", "Workflow (s)"),
        ([size, backend, *[values[key] for key in (
            "prepare_seconds", "compute_seconds", "result_seconds", "total_seconds",
            "input_bytes", "result_bytes", "workflow_seconds")]]
         for (size, backend), values in sorted(groups.items())))
    text = (
        '<!doctype html><html lang="fr"><meta charset="utf-8"><title>Extrema CPU/GPU</title>'
        '<style>body{font-family:sans-serif;margin:2rem}table{border-collapse:collapse}'
        'td,th{border:1px solid #aaa;padding:.5rem;text-align:right}</style>'
        '<h1>Extrema CPU hôte / GPU dans une simulation CUDA</h1>'
        f'<p>Médianes de {config["repetitions"]} essais, champs analysés tous les '
        f'{config["parameters"]["snap-time-interval"]} pas. Chaque temps est cumulé '
        'sur une exécution. Le bloc ne comprend ni le calcul physique ni le JSON.</p>'
        + table + '<p>cpu : parcours séquentiel après copie vers un tampon hôte ; '
        'kokkos : réduction CUDA et récupération de min/max et du drapeau de validité. '
        'Le maximum absolu est déduit des extrema. Les allocations sont hors du bloc.</p>'
        '<p>Le total est mesuré indépendamment : sa médiane ne se déduit pas en additionnant '
        'les médianes des composantes. Les octets indiquent les copies explicites cumulées, '
        'pas le trafic physique UVM/PCIe. Les mises à disposition peuvent inclure des migrations '
        'et de la synchronisation. Le workflow complet inclut aussi moyenne/variance et les écritures.</p>'
        '<p>Les résultats des extrema à chaque pas sont dans extrema.csv, sans seuil arbitraire. '
        'Des simulations indépendantes ne garantissent pas des champs bit à bit identiques ; '
        'une différence entre résultats ne suffit pas à incriminer la réduction.</p>'
        '<p>Figure : <a href="figures/cpu_gpu_extrema.svg">calcul et coût total</a>.</p></html>')
    (output / "cpu_gpu.html").write_text(text, encoding="utf-8")


def compare_cpu_gpu(config_path, config, cases, help_text):
    if "--extrema-backend" not in help_text:
        raise ValueError("Recompiler semproxy : --extrema-backend est absent")
    if "kokkos execution space: cuda" not in help_text.lower():
        raise ValueError("Expérience CPU/GPU : un exécutable CUDA et une machine GPU sont nécessaires")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    selected = [(case, n) for case in ("S", "M", "L") if case in cases
                for n in config["sizes"][case]]
    output = config_path.parent / "results" / "extrema-cpu-vs-gpu"
    output.mkdir(parents=True, exist_ok=False)
    figures = output / "figures"
    figures.mkdir()
    shutil.copyfile(config_path, output / "cpu_gpu_config.json")
    env = environment(config["executable"])
    env["execution_space"] = "Cuda"
    env["python"] = sys.version
    cpu_info = Path("/proc/cpuinfo")
    env["cpu_model"] = next((line.split(":", 1)[1].strip()
                             for line in cpu_info.read_text().splitlines()
                             if line.startswith("model name")), "non disponible") if cpu_info.is_file() else "non disponible"
    (output / "environment.json").write_text(json.dumps(env, indent=2) + "\n", encoding="utf-8")
    print(f"Comparaison des extrema : {len(selected) * 2 * config['repetitions']} simulations CUDA, aucun snapshot brut.", flush=True)
    if env["build"].get("CMAKE_BUILD_TYPE") != "Release":
        print("Compilation Release non confirmée par CMakeCache ; vérifier les options d'optimisation.", flush=True)
    records, extrema = [], []
    for case, n in selected:
        for repeat in range(1, config["repetitions"] + 1):
            pair = {}
            order = ("cpu", "kokkos") if repeat % 2 else ("kokkos", "cpu")
            for backend in order:
                directory = output / f"{case}-n{n}-run{repeat:02d}-{backend}"
                print(f"{case} N={n} essai {repeat} : extrema {backend}", flush=True)
                values, stats = run_extrema_case(config, n, backend, directory)
                records.append({"case": case, "elements": n, "repeat": repeat,
                                "backend": backend, **values})
                pair[backend] = stats["snapshots"]["per_snapshot"]
                for row in pair[backend]:
                    extrema.append({"case": case, "elements": n, "repeat": repeat,
                                    "backend": backend, "timestep_index": row["timestep_index"],
                                    "time_seconds": row["time_seconds"],
                                    **{key: row[key] for key in FIELDS[:3]}})
            if records[-1]["samples"] != records[-2]["samples"] or records[-1]["calls"] != records[-2]["calls"]:
                raise ValueError("Les deux chemins n'ont pas analysé les mêmes nombres de champs/pas")
            if [r["timestep_index"] for r in pair["cpu"]] != [r["timestep_index"] for r in pair["kokkos"]]:
                raise ValueError("Les instants analysés diffèrent")
            write_csv(output / "runs.csv", records)
            write_csv(output / "summary.csv", extrema_summary(records))
            write_csv(output / "extrema.csv", extrema)
    summary = extrema_summary(records)
    plot_extrema_comparison(summary, figures, plt)
    write_extrema_html(output, summary, config)
    print(f"Résultats CPU/GPU : {output}")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config",type=Path)
    parser.add_argument("--cases",nargs="+",choices=("S","M","L"),default=["S","M","L"])
    parser.add_argument("--data",nargs="+",choices=("snapshots","sismograms"),default=["snapshots","sismograms"])
    parser.add_argument("--compare-cpu-gpu", action="store_true",
                        help="Comparer les extrema CPU hôte / GPU avec cpu_gpu_config.json")
    args=parser.parse_args()
    try:
        config_path=args.config.resolve(); config=load_config(config_path, args.compare_cpu_gpu)
        if len(set(args.cases))!=len(args.cases) or len(set(args.data))!=len(args.data):
            raise ValueError("Cas ou types de données répétés")
        # Old builds accept unknown flags, so reject them before a costly run.
        help_result = subprocess.run([config["executable"], "--help"],
                                     capture_output=True, text=True, check=True)
        if "--outputs" not in help_result.stdout:
            raise ValueError("Recompiler semproxy : l'exécutable ne propose pas --outputs")
        if args.compare_cpu_gpu:
            if args.data != ["snapshots", "sismograms"]:
                raise ValueError("--data est réservé à la comparaison ad-hoc/in-situ")
            compare_cpu_gpu(config_path, config, args.cases, help_result.stdout)
            return
        import numpy  # noqa: F401 — required by analyze.py
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        selected=[(case,n) for case in ("S","M","L") if case in args.cases for n in config["sizes"][case]]
        repetitions=config["repetitions"]; k=config["parameters"]["snap-time-interval"]
        precision_runs=len(config["precision_intervals"]) if "snapshots" in args.data else 0
        print(f"Tailles : {selected} ; types : {args.data} ; essais : {repetitions}")
        print(f"Simulations prévues : {len(selected)*(2*repetitions*len(args.data)+precision_runs)}")
        if "snapshots" in args.data:
            _,_,volumes=estimate(config_path,args.cases,["snapshots"])
            mb=sum(row["snapshots_bytes"] for row in volumes)/1e6
            print(f"Volume approximatif de snapshots bruts écrits, puis supprimés : {mb/1000:.2f} GB")
        print("Budget : 20 GB ; binaires supprimés après validation et sauvegarde des mesures, hors chronométrage.")
        backend=next((line.partition(":")[2].strip().lower() for line in help_result.stdout.splitlines()
                      if line.startswith("Kokkos execution space:")),"vector")
        output=config_path.parent/"results"/f"ad-hoc-vs-in-situ-{backend}"
        output.mkdir(parents=True,exist_ok=False)
        shutil.copyfile(config_path,output/"compare_config.json")
        shutil.copyfile(config["receivers"],output/"receivers.txt")
        receivers=output/"receivers.txt"
        (output/"environment.json").write_text(json.dumps(environment(config["executable"]),indent=2)+"\n",encoding="utf-8")
        analyzer=Path(__file__).resolve().parents[1]/"analyze.py"
        records=[]; validation=[]; precision=[]; peak_inputs=[]
        for case,n in selected:
            for kind in args.data:
                first=None
                for repeat in range(1,repetitions+1):
                    pair={}
                    modes=("ad-hoc","in-situ") if repeat%2 else ("in-situ","ad-hoc")
                    for mode in modes:
                        directory=output/f"{case}-n{n}-{kind}-run{repeat:02d}-{mode}"
                        print(f"{case} N={n} {kind} essai {repeat}/{repetitions} : {mode}",flush=True)
                        values,stats,samples=execute(config,n,kind,mode,k,directory,receivers,analyzer)
                        records.append({"data":kind,"case":case,"elements":n,"mode":mode,"repeat":repeat,"samples":samples,"measurements":values})
                        pair[mode]=stats
                    validation.extend(compare_statistics(pair["ad-hoc"],pair["in-situ"],kind,case,n,repeat))
                    if repeat==1: first=pair["ad-hoc"]
                    save_results(output,records,validation,precision)
                    remove_raw_output(output/f"{case}-n{n}-{kind}-run{repeat:02d}-ad-hoc",kind)
                if kind=="snapshots":
                    print(f"{case} N={n} : référence in-situ à chaque pas",flush=True)
                    _,dense,dense_samples=execute(config,n,kind,"in-situ",1,output/f"{case}-n{n}-precision-reference",receivers,analyzer)
                    peak_inputs.append((dense,first,n,k))
                    for interval in sorted(config["precision_intervals"]):
                        sparse=first
                        if interval!=k:
                            print(f"{case} N={n} : précision ad-hoc k={interval}",flush=True)
                            _,sparse,_=execute(config,n,kind,"ad-hoc",interval,output/f"{case}-n{n}-precision-ad-hoc-k{interval}",receivers,analyzer)
                        precision.append(precision_row(dense,sparse,case,n,interval,dense_samples))
                        save_results(output,records,validation,precision)
                        if interval!=k:
                            remove_raw_output(output/f"{case}-n{n}-precision-ad-hoc-k{interval}",kind)
        summary=save_results(output,records,validation,precision)
        figures = output / "figures"
        figures.mkdir()
        plot_results(summary,figures,config,plt)
        for dense,sparse,n,k in peak_inputs:
            plot_peak(dense,sparse,n,k,figures,plt)
        write_html(output,summary,validation,precision,config)
        print(f"Campagne terminée. Tableaux : {output} ; figures : {figures}")
    except (OSError,ValueError,KeyError,TypeError,IndexError,RuntimeError,ImportError,
            subprocess.SubprocessError) as error:
        parser.exit(1,f"Erreur : {error}\n")


if __name__=="__main__":
    main()
