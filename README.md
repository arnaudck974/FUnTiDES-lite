# FUnTiDES: Fast Unstructured Time Dynamic Equation Solver

**FUnTiDES** is a collection of simplified codes that represent real scientific applications. It serves as a standard tool for evaluating and comparing the performance of various high-performance computing (HPC) systems, particularly those used for scientific simulations.

---

## Included Applications

The current implementation includes two proxy applications for solving the 2nd-order acoustic wave equation in 2D and 3D:

- **SEM (Spectral Element Method)**
  A benchmark designed to simulate wave propagation using SEM, a Galerkin-based finite element method for solving partial differential equations (PDEs).

- **FD (Finite Difference Method)**
  A benchmark that uses finite-difference stencil operators to simulate wave propagation and solve PDEs.

A key feature of these proxy applications is their adaptability to different programming models and HPC architectures. They are also easy to build and run, making them accessible to both researchers and developers.

---

## Supported Programming Models

The SEM proxy currently supports:

- [Kokkos](https://kokkos.github.io/kokkos-core-wiki/) — for performance portability

> **Note**: Kokkos is included as a Git submodule and will be compiled automatically when enabled.

---

## Supported Data Containers

The current SEM proxy supports the following data container:

- `std::vector` (default for serial )

---

## Quick Start: Build and Run

### Step 1: Compile and Install

```sh
mkdir build
cd build
cmake ..
make install
```

By default, this builds the applications in sequential mode using `std::vector`.
Both SEM and FD applications are compiled.

### Step 2: Run Examples

```sh
# Run SEM simulation with 100 x 100 x 100 elements
./src/main/semproxy -ex 100

# Run FD simulation
./src/main/fdproxy
```

---

## CMake Options

The following options can be used to configure your build:

| Option                 | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| `COMPILE_FD`           | Enable compilation of the FD proxy (default: ON)                            |
| `COMPILE_SEM`          | Enable compilation of the SEM proxy (default: ON)                           |
| `ENABLE_CUDA`          | Enable CUDA backend (used by Kokkos)                                        |
| `ENABLE_PYWRAP`        | Enable Python bindings via pybind11 (experimental)                          |
| `USE_KOKKOS`           | Enable Kokkos support (serial by default, CUDA/OpenMP with flags)           |
| `USE_VECTOR`           | Use `std::vector` for data arrays (enabled by default unless Kokkos is used)|

---

## Organisation des outils Python

Les outils sont regroupés dans `scripts/` :

```text
scripts/
├── snapshot_io.py          # lecteur commun de mesh.bin et des pressions
├── analyze.py              # statistiques ad-hoc
├── benchmarks/            # campagnes, estimations et configurations JSON
└── visualisation/         # coupes, propagation et figure RLE
```

Les commandes ci-dessous s'exécutent depuis la racine du projet. Les résultats
des campagnes restent dans `benchmarks/results/` ; les résultats existants
ne sont pas déplacés. `scripts/benchmarks/snapshot_stats.py` est l'ancien outil
CSV : il est conservé sans adaptation et ne lit pas le nouveau format binaire.
Les statistiques binaires utilisent `scripts/analyze.py`.

## Lancement complet des mesures TP2

Depuis la racine, après activation de votre environnement Python :

```sh
bash scripts/benchmarks/run_tp2.sh
```

Le script ne prend aucun argument. La campagne complète nécessite CMake avec
l'option `--fresh`, Make, un compilateur C++, NumPy/Matplotlib, le sous-module
Kokkos, le toolkit
CUDA (`nvcc`) et un GPU NVIDIA accessible (`nvidia-smi`). Il vérifie les
prérequis, les chemins des exécutables dans les quatre JSON et les ressources
CPU avant le nettoyage. Aucune dépendance n'est installée automatiquement.

Il supprime `build/` **une seule fois, au début**, compile `semproxy` en
Release/Kokkos/OpenMP, puis exécute strong scaling et weak scaling. Il
reconfigure ensuite **le même dossier `build/`** avec `cmake --fresh` pour
Release/Kokkos/CUDA, recompile, puis lance la comparaison ad-hoc/in-situ et les
extrema CPU/GPU. `--fresh` réinitialise le cache et les métadonnées CMake pour
changer de compilateur ; le dossier `build/` n'est pas supprimé une deuxième
fois. Il n'y a aucun sous-dossier de compilation `build/openmp/` ou
`build/cuda/` ; les quatre JSON utilisent `../../build/bin/semproxy`.
Les campagnes sont successives ; une erreur interrompt le lancement.
Compilation, nettoyage et contrôles sont hors des chronomètres des mesures.
À la fin, `build/bin/semproxy` utilise CUDA. Aucun fichier CMake n'est modifié.

Les paramètres restent dans les quatre JSON de `scripts/benchmarks/`. Pour le workflow
CUDA, `OMP_NUM_THREADS` est conservé s'il est défini, sinon il vaut 1 ; les
bibliothèques numériques sont limitées à un thread. Les pilotes strong/weak
définissent eux-mêmes leurs pools de simulation. Avec les configurations
fournies, le lancement prévoit **545 simulations**, avec cinq répétitions :

- comparaison ad-hoc/in-situ : S = `[12,16,20]`, M = `[24,28,32]`, L = `[35,38,40]` ;
- extrema CPU/GPU : S = `[10,20,30]`, M = `[40,50]`, L = `[60,70,80]` ;
- strong scaling : N = 50, threads = `[1,2,3,4,6,8]` ;
- weak scaling : 8 000 éléments par thread, avec ces mêmes nombres de threads.

Le pic brut nominal est d'environ **3,20 GB**, lors de la comparaison de
précision à N = 40, k = 1. Les dimensions sont des nombres d'éléments ;
le nombre de nœuds vaut `(order*N+1)^3` dans les cas cubiques.
Il faut huit cœurs physiques disponibles et assez de mémoire CPU/GPU ; le
budget disque ne garantit pas la capacité mémoire ni un temps d'exécution court.

**Après chaque essai ad-hoc analysé, contrôlé et enregistré dans les CSV**, seuls
ses `snapshots/snapshot_<pas>.bin` et `snapshots/mesh.bin`, ou son `sismos.bin`,
sont supprimés. Pour les
essais chronométrés de `compare.py`, le contrôle de la paire ad-hoc/in-situ
précède la suppression. Les JSON, CSV, configurations, logs et figures restent
disponibles. Un échec avant cette étape conserve les binaires pour diagnostic.
Le nettoyage est hors des chronomètres et les volumes enregistrés restent les
volumes réellement **écrits**, pas le volume restant après suppression.
Les figures TP2 utilisent les statistiques et mesures conservées ; pour refaire
une autre analyse des champs bruts ou une visualisation spatiale, il faudra
relancer la simulation.

Avant chaque essai, le pilote contrôle le volume actuel de `build/` et des
dossiers des quatre campagnes, ajoute le brut prévu et **1 GB de réserve**
pour les nouveaux rapports et la gestion du stockage, puis refuse de lancer
l'essai si le total dépasse **20 GB** ou si l'espace libre ne suffit pas.
Le contrôle du brut prévoit un pas supplémentaire pour les arrondis float32.
Le lanceur vérifie aussi le pic nominal avant son nettoyage initial.
Ces contrôles sont prévisionnels, sans quota du système de fichiers : une autre
application qui remplit le disque peut encore provoquer une erreur d'écriture.

Les quatre dossiers de sortie sont fixes, sans date :

```text
benchmarks/results/
├── strong-scaling/
├── weak-scaling/
├── ad-hoc-vs-in-situ-cuda/
└── extrema-cpu-vs-gpu/
```

Chaque dossier contient `figures/`, ses CSV, sa configuration, l'environnement
et les fichiers des essais. **À chaque relance, l'ancienne campagne concernée
est entièrement remplacée juste avant sa nouvelle exécution.** Ses anciens
fichiers sont supprimés pour éviter de mélanger des essais de configurations
différentes. Les nouveaux résultats restent disponibles après l'exécution.
Les autres dossiers de résultats sont conservés. Les noms restent fixes.
Le lanceur refuse les liens symboliques pour `build/`, `results/` et les
quatre dossiers ciblés. Lancés directement, les scripts Python refusent aussi
un dossier de campagne déjà existant.

Pour estimer le stockage avant de lancer, sans GPU ni compilation :

```sh
python3 scripts/benchmarks/estimate_tp2_storage.py
```

Cet outil lit les quatre JSON et affiche une ligne par configuration : volume
cumulé des snapshots et sismogrammes écrits, leur total, et le pic brut
temporaire. Le volume des snapshots inclut les maillages, chacun écrit une
fois par essai ad-hoc. La somme des volumes écrits inclut les répétitions et les sorties
de précision de `compare.py` ; elle peut dépasser 20 GB puisque les binaires
sont supprimés progressivement. Le pic global est le maximum des quatre pics,
pas leur somme. Les pics bruts excluent JSON, CSV, logs, figures et `build/`,
qui sont pris en compte par les contrôles pendant l'exécution.
Le zéro de la campagne CPU/GPU signifie zéro donnée brute ; ses statistiques
et figures occupent aussi du stockage. Les arrondis float32 du simulateur
peuvent légèrement modifier le nombre réel de pas et le volume.

## Format binaire des snapshots : maillage partagé

Le maillage est fixe pendant une simulation. Les sorties utilisent deux fichiers
natifs sans en-tête ni remplissage, avec le même ordre de nœuds :

```text
snapshots/mesh.bin             : x0, y0, z0, x1, y1, z1, ... (float32)
snapshots/snapshot_<pas>.bin   : p0, p1, ...                 (float32)
```

`mesh.bin` est écrit une seule fois, au premier snapshot de chaque exécution.
Les coordonnées physiques et les pressions gardent leur précision ; les
identifiants ne sont plus écrits. La pression d'indice `i` correspond au
triplet de coordonnées d'indice `i`. Pour `n` nœuds et `T` snapshots, le volume
brut est `12*n + 4*n*T` octets. Le temps d'écriture du maillage est inclus dans
le premier snapshot et son volume est compté une fois dans `snapshot_mb`.

`scripts/snapshot_io.py` fournit le lecteur commun à `scripts/analyze.py` et
`scripts/visualisation/visualize.py`.
Le nombre de nœuds est déduit de la taille de `mesh.bin`. Les lecteurs vérifient
que chaque snapshot contient exactement une pression par nœud. La visualisation
charge le maillage une seule fois pour toute la série.

L'ancien format entrelacé `identifiant,x,y,z,pression` n'est plus pris en charge.
Les anciennes sorties restent sur disque mais doivent être régénérées dans un
nouveau dossier pour utiliser ces lecteurs. Les mesures archivées de l'ancien
format ne décrivent pas les coûts du nouveau : refaire la référence avant
d'évaluer un gain supplémentaire de compression. Les sismogrammes et le RLE
gardent leurs formats actuels.

Après recompilation, les commandes Python restent les mêmes :

```sh
python3 scripts/analyze.py --snapshots snapshots --dt 0.001
python3 scripts/visualisation/visualize.py snapshots snapshots --out-dir figures
```

`--dt` doit correspondre au pas réellement utilisé par la simulation. L'indice
du pas est conservé dans le nom du fichier. Pour vérifier les lecteurs sans
lancer de simulation :

```sh
python3 -B -m unittest discover -s tests -p test_snapshot_io.py -v
```

## Figure de propagation à trois instants

Après une simulation produisant le nouveau format `mesh.bin` et les pressions
seules, choisir trois indices de snapshots présents dans le dossier :

```sh
python3 scripts/visualisation/visualize_propagation.py snapshots \
  --steps 200 400 600 --dt 0.001 --output figures/propagation.svg
```

`--dt` doit être celui réellement utilisé par la simulation. Les titres
indiquent `t = (indice + 1) * dt`, puisque l'indice 0 est sauvegardé après le
premier pas. Le script choisit le plan disponible le plus proche du milieu
du domaine ; `--z 1000` permet de demander un autre plan. Le plan réellement
retenu apparaît dans la figure et dans le terminal.

Les trois panneaux partagent une barre de couleurs, symétrique autour de zéro,
fixée par la pression absolue maximale des trois coupes choisies. Les pressions
ne sont pas normalisées séparément. Aucune interpolation des pressions n'est
effectuée. `--output` accepte SVG, PNG ou PDF. Ce script ne modifie ni les
statistiques ni `metrics.json`. Aucun nouvel histogramme n'est ajouté.

## TP2: separate snapshot and sismogram comparisons

Rebuild `semproxy` after these C++ changes. The new simulation option is
`--outputs snapshots|sismograms|both` (default: `both`). It selects raw outputs
in ad-hoc mode and the corresponding statistics in in-situ mode. Snapshot-only
runs skip receiver allocation and interpolation; signal-only runs skip field
statistics and snapshots. The pressure solver is unchanged.

From the repository root, using your Python environment with NumPy/Matplotlib:

```sh
python3 scripts/benchmarks/compare.py scripts/benchmarks/compare_config.json --cases S
python3 scripts/benchmarks/compare.py scripts/benchmarks/compare_config.json --cases S --data snapshots
python3 scripts/benchmarks/compare.py scripts/benchmarks/compare_config.json --cases S M L --data sismograms
```

`--cases` defaults to S M L; `--data` defaults to both independent campaigns.
Edit sizes, repetitions, physics and `precision_intervals` in the configuration.
Paths are relative to that JSON. Use the same executable and resource settings
throughout; for OpenMP, set `OMP_NUM_THREADS` before launching Python. This is
a problem-size study, not strong/weak scaling.

For each type and size, the script alternates ad-hoc/in-situ order across
repetitions. Both simulate every physical step. For timing, field statistics
use the same steps as the saved snapshots (0, k, 2k, ...); receiver statistics
use every step in both modes. The full workflow clock includes process startup,
simulation, raw writes, Python reads/analysis for ad-hoc, and statistics JSON
writing. JSON reading for reporting and report/figure generation are outside
this clock. No normalization per snapshot is applied.

Writing results is distinct from calculating statistics. In-situ field statistics
are streamed into JSON at selected steps; receiver statistics are written at the
end. No raw pressure histories are kept in situ. Reads of pressure in memory
belong to statistical calculation; raw disk reads occur only in ad-hoc.

Snapshots also get one in-situ reference at every step, then actual ad-hoc
outputs for each configured precision interval. The first timed ad-hoc trial is
reused at the timing cadence. These extra runs are excluded from timing medians.
The script prints the simulation count and estimated raw snapshot storage;
saving at k=1 can require substantial space. With default configuration, group S
and both data types run 25 simulations (20 timing + 5 precision).

The fixed `benchmarks/results/ad-hoc-vs-in-situ-<backend>/` directory contains
independent runs, logs, configuration/environment copies and progressive CSV
files. `<backend>` is the actual backend, such as `cuda`, `openmp` or `vector`:

- `runs.csv`: every timed run; `summary.csv`: medians by type, size and mode.
- `validation.csv`: paired statistics and absolute differences, without an
  arbitrary acceptance threshold. Invalid files/metadata still cause an error.
- `precision.csv`: actual snapshot peaks and temporal means for each k and the
  in-situ reference; not a physical exact solution.

After all measurements, the script puts SVGs in the campaign's `figures/`
subdirectory. CSVs and `statistics.html` remain at the campaign root:

- `costs_snapshots.svg`, `costs_sismograms.svg`: log time axis, separate raw
  write, raw read, statistics calculation and statistics JSON write curves.
  Zero raw in-situ I/O is stated explicitly and omitted from the log axis.
- `total_snapshots.svg`, `total_sismograms.svg`: actual complete workflow times
  from the independent campaigns, including physics and all result writes.
- `writes.svg`: grouped bars, rows for snapshots/signals, columns for total
  write time and volume (raw + statistics). Values are labelled above bars.
- `peak_evolution_n*.svg`: in-situ maximum absolute pressure at every step versus
  actual ad-hoc sample points. No additional error-curve figure.
- `statistics.html`: four tables: full time/storage, operation medians, snapshot
  quality by cadence, and per-receiver statistical values from the first trial.
  It contains no figures and is generated once, outside the timed workflows.

Performance figures display medians only. Raw individual results remain in CSV.
Different Python/NumPy and C++/Kokkos implementations affect analysis costs;
GPU/host transfer times are not isolated. Disk caches are not explicitly
controlled. Logs, metric files and reporting artifacts are excluded from data
volume totals. The referenced measurement-methodology annex is absent from
the supplied `doc/` files. Compilation and actual backend execution remain to
be checked locally.

## Comparaison des extrema CPU hôte / GPU

Le code est fourni ; la compilation et la validation CUDA restent à effectuer
sur la machine GPU. Les vérifications locales ne constituent pas une mesure GPU.

```sh
python3 scripts/benchmarks/compare.py scripts/benchmarks/cpu_gpu_config.json --compare-cpu-gpu --cases S M L
```

Le fichier dédié contient seulement exécutable, tailles, répétitions et paramètres
physiques. Le pilote lance deux chemins in-situ sur le même exécutable CUDA :
`--extrema-backend cpu` (parcours séquentiel après mise à disposition sur l'hôte)
et `--extrema-backend kokkos` (réduction GPU et retour des résultats).
Seuls les extrema changent de lieu de calcul, pas l'onde ni moyenne/variance.
Sans l'option C++ expérimentale, le traitement habituel reste automatique.
Kokkos/OpenMP permet de vérifier les chemins localement après compilation,
mais le pilote `--compare-cpu-gpu` exige CUDA et refuse de le présenter comme GPU.

Pour trois tailles et cinq répétitions, 30 simulations sont lancées, sans
snapshots bruts. Le dossier `results/extrema-cpu-vs-gpu/` contient
`runs.csv`, `summary.csv`, `extrema.csv`, `cpu_gpu.html`,
la configuration, l'environnement et les fichiers de chaque essai.
La figure de comparaison est dans `figures/`.
La génération du rapport et des figures est hors des chronomètres.

## Strong scaling Kokkos/OpenMP

Cette expérience garde le problème fixe et fait varier seulement le nombre de
threads CPU. Utiliser un même exécutable **Release, Kokkos/OpenMP, sans CUDA**
pour toute la campagne et le recompiler après les modifications de `main.cc`.
Les fichiers CMake ne sont pas modifiés. Le script vérifie le mode Release
dans le `CMakeCache.txt` du build et le backend réellement annoncé par Kokkos.

Depuis la racine, avec NumPy et Matplotlib dans l'environnement Python :

```sh
python3 scripts/benchmarks/strong_scaling.py scripts/benchmarks/strong_scaling_config.json
```

La configuration contient une seule `size` : `ex = ey = ez = size`. Les
paramètres physiques, l'ordre et la cadence restent fixes pour toutes les
exécutions. Les chemins sont relatifs au JSON. Adapter `executable` à votre
build et `threads` aux cœurs disponibles sur la machine ; la liste commence
par 1 pour fournir la référence. Le script refuse de dépasser le nombre de
cœurs physiques détectés dans l'affinité du processus. Si cette topologie
n'est pas accessible, il vérifie les processeurs logiques et le signale.
Il ne réserve pas les ressources : utiliser une machine ou une allocation
disponible, sans autres travaux concurrents.

Pour chaque nombre de threads, snapshots et sismogrammes sont mesurés
séparément, en ad-hoc et en in-situ, cinq fois par défaut. Les snapshots sont
analysés aux mêmes indices `0, k, 2k, ...` dans les deux approches ; les
sismogrammes utilisent les mêmes récepteurs et tous les pas. Il n'y a pas de
campagne de précision supplémentaire. Le volume brut d'un essai snapshots
est `n*(12 + 4*T)` octets, avec `n=(order*N+1)^3` et `T` sorties. L'estimation
de la campagne inclut un maillage par essai ad-hoc, les répétitions et les
nombres de threads configurés. Elle utilise le nombre nominal de pas ; les
compteurs mesurés restent la référence.

Le pilote demande explicitement les threads à Kokkos et OpenMP, désactive
l'ajustement dynamique et utilise `OMP_PLACES=cores`, `OMP_PROC_BIND=spread`.
Le pool Kokkos effectif est contrôlé dans chaque log, ainsi que le backend.
Les processus Python d'analyse gardent OpenMP et les bibliothèques numériques
à un thread. C'est donc la scalabilité du **workflow avec le solveur et les
analyses Kokkos parallélisés**, avec un post-traitement Python fixe. Les E/S et
le code séquentiel ne deviennent pas automatiquement parallèles.

Chaque répétition alterne l'ordre ad-hoc/in-situ ; le parcours des nombres de
threads est également inversé une répétition sur deux. Aucun échauffement ni
vidage des caches n'est ajouté. Les appels `--help` de vérification initialisent
les pools de threads, sans exécuter la physique et hors chronométrage.

Un chronomètre externe couvre le workflow complet : démarrage et fin des
processus, simulation et écritures, puis lecture/analyse Python en ad-hoc.
Le JSON de statistiques est inclus dans les deux approches. La lecture des
rapports, les contrôles et la génération des figures sont hors de ce temps.
Les sorties de processus sont redirigées vers leurs logs dans les deux cas.

Les résultats sont dans `benchmarks/results/strong-scaling/` :

- `runs.csv` : toutes les mesures individuelles, avec nombre de threads,
  approche, répétition, type de données et compteurs de calcul/analyse/E/S ;
- configuration, récepteurs et `environment.json` : machine, affinité,
  compilation, versions Python/NumPy/Matplotlib et réglages des sous-processus ;
- un dossier par exécution contenant logs, métriques, statistiques et données
  brutes éventuelles ; aucun fichier brut n'est supprimé par le pilote ;
- `figures/strong_scaling_snapshots.svg` et `figures/strong_scaling_sismograms.svg`.

Chaque figure contient trois panneaux selon le nombre de threads `p` :
temps médian du workflow `T(p)`, accélération `S(p)=T(1)/T(p)` et efficacité
`E(p)=100*S(p)/p`. Chaque approche utilise sa propre référence à un thread.
Ad-hoc est en ligne continue et in-situ en ligne discontinue ; seules les
médianes sont représentées. Les références idéales sont `S(p)=p` et `E(p)=100 %`.
Les rapports sont calculés à partir des médianes, pas en moyennant des rapports
entre essais. Une approche peut être plus rapide en temps absolu tout en ayant
une accélération plus faible.

Un plateau doit être confronté aux compteurs : E/S, parties séquentielles,
synchronisations ou bande passante mémoire peuvent limiter le gain. Ces temps
ne suffisent pas à identifier seuls la cause. Une efficacité supérieure à
100 % est possible, notamment avec des effets de cache, et doit être examinée.
Les caches disque ne sont pas contrôlés et aucun `fsync` n'est ajouté : les
mesures d'E/S ne représentent pas un débit physique garanti du disque.
L'expérience complète l'étape 5 du TP2 ; elle ne remplace pas son modèle
d'évolution des coûts avec la taille du maillage et la contrainte CFL.

## Weak scaling Kokkos/OpenMP

Le weak scaling augmente ensemble les threads et le nombre total d'éléments.
Il utilise le même exécutable Release/Kokkos/OpenMP que le strong scaling,
sans modification C++ ou CMake supplémentaire. Depuis la racine :

```sh
python3 scripts/benchmarks/weak_scaling.py scripts/benchmarks/weak_scaling_config.json
```

Le JSON contient `executable`, `receivers`, `cases`, `repetitions` et
`parameters`. Chaque cas indique explicitement `threads`, `ex`, `ey`, `ez`.
La référence est le premier cas, à un thread. Les nombres de threads doivent
être distincts et croissants ; le script vérifie exactement
`ex*ey*ez = threads * (ex0*ey0*ez0)`.

| Threads | Maillage | Domaine physique calculé |
|---:|---|---|
| 1 | 20 × 20 × 20 | 2000 × 2000 × 2000 |
| 2 | 40 × 20 × 20 | 4000 × 2000 × 2000 |
| 4 | 40 × 40 × 20 | 4000 × 4000 × 2000 |
| 8 | 40 × 40 × 40 | 4000 × 4000 × 4000 |

Les longueurs de `parameters` décrivent le premier cas. Pour chaque cas,
le script calcule `lx = lx0 * ex/ex0`, et de même pour y et z. On conserve
ainsi les dimensions physiques des éléments, l'ordre, `dt`, `timemax` et la
cadence ; aucun raffinement spatial n'est introduit. Le pas de temps doit
être stable pour le modèle choisi. Les coordonnées de source et des récepteurs
restent fixes et doivent appartenir à chaque domaine.

Le réglage fourni garde 8000 éléments par thread. Le nombre de nœuds vaut
`(order*ex+1)*(order*ey+1)*(order*ez+1)` ; les frontières font varier légèrement
les nœuds par thread. Les charges du solveur par éléments et des statistiques
par nœuds ne sont donc pas strictement identiques. Le CSV conserve les deux
comptages et les longueurs réellement transmises au programme.

Le pilote réutilise l'exécution du strong scaling : mêmes contrôles du backend
et du pool effectif, placement OpenMP, analyse Python à un thread et chronomètre
du workflow complet. Snapshots et sismogrammes restent séparés ; les approches
analysent les mêmes instants à chaque taille. Cinq répétitions par défaut,
ordre des cas et des approches alterné, donnent **80 simulations**, sans essais
de précision supplémentaires. Environ **7,54 GB de snapshots bruts** sont
conservés avec cette configuration ; cette estimation utilise les pas nominaux.
Les ressources disponibles doivent couvrir la liste de threads configurée.

Les sorties sont dans `benchmarks/results/weak-scaling/` :

- `runs.csv` : mesures individuelles, dimensions, éléments/nœuds par thread,
  temps du workflow et compteurs de calcul, analyse et E/S ;
- configuration, récepteurs, environnement et dossiers de chaque essai ;
- `figures/weak_scaling_snapshots.svg` et `figures/weak_scaling_sismograms.svg`.

Chaque figure comporte deux panneaux : temps médian `T(p)` et efficacité faible
`100*T(1)/T(p)`. Chaque approche utilise sa propre référence à un thread ; il
n'y a pas de division supplémentaire par p. L'idéal est un temps constant et
une efficacité de 100 %. Ad-hoc est continu, in-situ discontinu. L'abscisse
précise les threads et le maillage ; les mesures individuelles restent dans le
CSV. Les figures sont produites après les mesures, sans barres min–max.

L'efficacité porte sur le workflow complet, dont certaines parties restent
séquentielles : lecture, écritures et analyse Python peuvent augmenter avec les
données, même si le calcul physique se répartit bien. Pour les sismogrammes,
le nombre de récepteurs reste fixe : cette courbe évalue surtout le workflow
avec un solveur croissant, pas une charge sismique constante par thread.
Les temps seuls ne permettent pas d'attribuer une perte à une cause précise.
On reste sur une machine à mémoire partagée ; sa bande passante mémoire et son
stockage ne grandissent pas proportionnellement aux threads. Les limites de
mesure des caches et des E/S décrites pour le strong scaling s'appliquent aussi.
Cette étude par agrandissement du domaine complète le modèle demandé à
l'étape 5 du TP2, qui considère un raffinement à domaine physique fixe.
