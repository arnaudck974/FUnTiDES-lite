#!/usr/bin/env bash
# Build each backend once, then run the existing TP2 measurement protocols.
set -Eeuo pipefail
trap 'printf "Erreur à la ligne %s : arrêt de la campagne.\n" "$LINENO" >&2' ERR

fail() {
    printf 'Erreur : %s\n' "$*" >&2
    exit 1
}

[[ $# -eq 0 ]] || fail "Ce script s'utilise sans argument : bash scripts/benchmarks/run_tp2.sh"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd -P)
BUILD_DIR="$PROJECT_ROOT/build"
RESULTS_DIR="$PROJECT_ROOT/benchmarks/results"
NVCC_WRAPPER="$PROJECT_ROOT/external/kokkos/bin/nvcc_wrapper"
cd -- "$PROJECT_ROOT"

check_directory() {
    [[ ! -L "$1" ]] || fail "Chemin refusé : $1 est un lien symbolique."
    [[ ! -e "$1" || -d "$1" ]] || fail "Un répertoire est attendu : $1"
}

clean_build() {
    # Called once, before the first configuration; both backends reuse this path.
    check_directory "$BUILD_DIR"
    printf 'Nettoyage : %s\n' "$BUILD_DIR"
    rm -rf -- "$BUILD_DIR"
}

replace_results() {
    # Replace only the selected old campaign, immediately before its new run.
    case "$1" in
        strong-scaling|weak-scaling|ad-hoc-vs-in-situ-cuda|extrema-cpu-vs-gpu) ;;
        *) fail "Campagne non autorisée : $1" ;;
    esac
    local directory="$RESULTS_DIR/$1"
    check_directory "$RESULTS_DIR"
    check_directory "$directory"
    printf 'Remplacement des anciens résultats : %s\n' "$directory"
    rm -rf -- "$directory"
}

# Check dependencies and cleanup targets before changing any generated files.
for program in cmake make c++ python3 nvcc nvidia-smi; do
    command -v "$program" >/dev/null || fail "Commande manquante : $program"
done
[[ -f "$PROJECT_ROOT/CMakeLists.txt" ]] || fail "Racine du projet introuvable."
[[ -f "$PROJECT_ROOT/external/kokkos/CMakeLists.txt" && -x "$NVCC_WRAPPER" ]] || \
    fail "Le sous-module Kokkos et son nvcc_wrapper doivent être présents."
check_directory "$BUILD_DIR"
check_directory "$RESULTS_DIR"
for name in strong-scaling weak-scaling ad-hoc-vs-in-situ-cuda extrema-cpu-vs-gpu; do
    check_directory "$RESULTS_DIR/$name"
done
CMAKE_HELP=$(cmake --help)
[[ "$CMAKE_HELP" == *"--fresh"* ]] || fail "CMake doit prendre en charge --fresh pour changer de compilateur dans build/."
GPU_NAMES=$(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ -n "$GPU_NAMES" ]] || fail "Aucun GPU NVIDIA détecté pour la campagne complète."
[[ ${CUDA_VISIBLE_DEVICES-unset} != "" && ${CUDA_VISIBLE_DEVICES-unset} != "-1" ]] || \
    fail "CUDA_VISIBLE_DEVICES masque les GPU."
printf 'GPU détecté(s) :\n%s\n' "$GPU_NAMES"

# Fixed numerical-library settings; scaling scripts set their own simulation pools.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export KOKKOS_NUM_THREADS="$OMP_NUM_THREADS"
export OMP_THREAD_LIMIT="$OMP_NUM_THREADS"
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND="${OMP_PROC_BIND:-spread}"
export OMP_PLACES="${OMP_PLACES:-cores}"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

python3 -B - "$PROJECT_ROOT" <<'PY'
import json
from pathlib import Path
import os
import shutil
import sys

try:
    import numpy
    import matplotlib
    root = Path(sys.argv[1])
    sys.path.insert(0, str(root / "scripts/benchmarks"))
    from strong_scaling import cpu_resources
    from estimate_tp2_storage import estimate_all, STORAGE_BUDGET, STORAGE_RESERVE
    peak = max(row["peak_bytes"] for row in estimate_all(root / "scripts/benchmarks"))
    if peak + STORAGE_RESERVE > STORAGE_BUDGET:
        raise ValueError(f"Pic brut prévu {peak/1e9:.3f} GB + 1 GB de réserve > budget de 20 GB")
    if shutil.disk_usage(root).free < peak + STORAGE_RESERVE:
        raise ValueError(f"Espace libre insuffisant : au moins {peak/1e9 + 1:.3f} GB nécessaires")
    print(f"Pic brut nominal : {peak/1e9:.3f} GB ; contrôle du budget avant chaque essai.", flush=True)
    resources = cpu_resources()
    limit = resources["physical_cores"] or len(resources["affinity_cpus"])
    requested = int(os.environ["OMP_NUM_THREADS"])
    if requested <= 0:
        raise ValueError("OMP_NUM_THREADS doit être positif")
    for name in ("compare_config.json", "cpu_gpu_config.json", "strong_scaling_config.json", "weak_scaling_config.json"):
        path = root / "scripts/benchmarks" / name
        config = json.loads(path.read_text(encoding="utf-8"))
        if (path.parent / config["executable"]).resolve() != root / "build/bin/semproxy":
            raise ValueError(f"{name} doit utiliser ../../build/bin/semproxy")
        if "receivers" in config and not (path.parent / config["receivers"]).is_file():
            raise ValueError(f"Fichier de récepteurs introuvable dans {name}")
        threads = config.get("threads", [case["threads"] for case in config.get("cases", [])])
        if any(type(p) is not int or p <= 0 for p in threads):
            raise ValueError(f"Threads invalides dans {name}")
        requested = max([requested, *threads])
    if requested > limit:
        raise ValueError(f"{requested} threads demandés, seulement {limit} cœurs disponibles détectés")
except (ImportError, OSError, ValueError, KeyError, TypeError) as error:
    sys.exit(f"Prérequis non satisfaits : {error}")
print("Prérequis Python, configurations et ressources CPU : OK", flush=True)
PY

build_backend() {
    local backend="$1" compiler expected
    local cuda=OFF
    local -a configure_args=()
    compiler=$(command -v c++)
    expected=OpenMP
    if [[ "$backend" == cuda ]]; then
        cuda=ON
        compiler="$NVCC_WRAPPER"
        expected=Cuda
        # Reset compiler/cache metadata while keeping the same build directory.
        configure_args+=(--fresh)
    fi
    printf 'Compilation Release/Kokkos/%s\n' "$expected"
    cmake "${configure_args[@]}" -S "$PROJECT_ROOT" -B "$BUILD_DIR" -G "Unix Makefiles" \
        -DCMAKE_BUILD_TYPE=Release -DUSE_KOKKOS=ON -DUSE_VECTOR=OFF \
        -DENABLE_CUDA="$cuda" -DCOMPILE_SEM=ON -DCMAKE_CXX_COMPILER="$compiler"
    cmake --build "$BUILD_DIR" --target semproxy --parallel 4
    local help_text
    help_text=$("$BUILD_DIR/bin/semproxy" --kokkos-num-threads=1 --help)
    [[ $'\n'"$help_text"$'\n' == *$'\n'"Kokkos execution space: $expected"$'\n'* ]] || \
        fail "Le backend compilé ne correspond pas à $expected."
}

clean_build
build_backend openmp
replace_results strong-scaling
python3 "$SCRIPT_DIR/strong_scaling.py" "$SCRIPT_DIR/strong_scaling_config.json"
replace_results weak-scaling
python3 "$SCRIPT_DIR/weak_scaling.py" "$SCRIPT_DIR/weak_scaling_config.json"

build_backend cuda
replace_results ad-hoc-vs-in-situ-cuda
python3 "$SCRIPT_DIR/compare.py" "$SCRIPT_DIR/compare_config.json"
replace_results extrema-cpu-vs-gpu
python3 "$SCRIPT_DIR/compare.py" "$SCRIPT_DIR/cpu_gpu_config.json" --compare-cpu-gpu

printf 'Campagnes terminées dans %s ; chaque dossier contient figures/.\n' "$RESULTS_DIR"
