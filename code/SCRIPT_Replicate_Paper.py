"""
SCRIPT_Replicate_Paper.py  -  baseline runs for the three planned extensions

Put this file in code/ next to SCRIPT_Swarm.py and run it from there.

    python SCRIPT_Replicate_Paper.py --dry-run          # print plan + arena geometry, nothing simulated
    python SCRIPT_Replicate_Paper.py --smoke-test       # 5 bats, 5 steps per experiment, nothing saved
    python SCRIPT_Replicate_Paper.py --workers 4        # full run, 4 runs in parallel (resumes if interrupted)
    python SCRIPT_Replicate_Paper.py --summarize        # compare finished runs with the paper's numbers
    python SCRIPT_Replicate_Paper.py --optional         # also run perched / differential / muted / random

WHICH RESULTS OF VANDERELST & PEREMANS (2025) ARE BASELINES, AND WHY
---------------------------------------------------------------------
Only conditions that one of our extensions has to be compared against are run
by default. The perched and differential conditions are not baselines for any
extension and are therefore behind --optional.

  B1  Default, branched arena, 25 bats                          (ALL extensions)
      Paper: median 3 collisions/run, 164 steps (6.6 s), wall-distance mode
      ~1.6 m, inter-agent distance mode 0.90 m, 0.43% of steps within 13 cm of
      a wall, 0.03% within 13 cm of another bat.
      Reference point for every number we report after changing the model.

  B2  Curved and widened arenas, 25 bats                        (ARENAS)
      Paper: 2 and 1 collisions, 0.57% and 0.54% of steps within 13 cm of a wall.
      Shows how much the metrics move with geometry alone, so that the effect
      of our new arenas (height changes, urban-like, Mazar & Yovel layout) can
      be told apart from ordinary arena-to-arena variation.

  B4  "No others" (calls and body echoes of others removed)     (COHESION)
      Paper: inter-agent distance mode 0.31 m, 4.8% of steps within 13 cm.
      This is what spacing looks like with no social acoustic repulsion. A
      cohesion mechanism has to pull the swarm tighter than B1 (0.90 m) towards
      the field value (Weesner mode 0.47 m, mean 0.69 m) without collapsing to B4.

  B5  Loudness and loudest-input statistics of B1               (COHESION)
      Paper Fig 4: others' calls and the secondary body echoes dominate.
      We log the distribution of the integrated level that the controller
      thresholds ('max_integrated') and the share of steps in which each input
      type is loudest. The adaptive attraction (strong when quiet, fading when
      loud) needs a loudness scale; these quantiles are where to take it from.

  B6  Vertical baseline from B1                                 (VERTICAL MOVEMENT)
      The paper has no vertical result: bats keep a fixed height, drawn
      uniformly from -0.5 to 0.5 m, and wall reflectors span -1 to 1 m. The
      baseline we log is therefore the height spread (z_std, z_min, z_max), the
      inter-agent distance with and without the vertical component (nn_*_3d vs
      nn_*_xy) and the speed statistics, to compare with the free-height version.

HOW THE METRICS ARE COMPUTED
----------------------------
  * Distances and proximity fractions skip the first 50 steps
    (PaperSettings.throw_away_steps; the paper also discards 50 steps for its
    distance analyses).
  * The paper does not say whether collision counts are trimmed, so collisions
    are reported both ways: *_all and *_after50.
  * 'wall_crossings_*' counts every step in which a bat left the arena polygon.
    'collision_events_*' merges crossings of the swarm that are less than 5
    steps apart into one event, the same rule as Utils.count_hits; this is the
    more likely definition of the paper's "collisions per run".
    Swarm.wall_collisions is never incremented in Swarm.py, so it is not used.
  * Modes are kernel-density modes (bandwidth PaperSettings.bw, range 0-3 m).
    The authors' analysis scripts are not in the repository, so the modes can
    differ slightly from the paper's.

NOT RUN HERE
------------
  The paper's 50-bat condition (4 collisions) is left out on purpose: it is
  expensive and no extension needs it as a reference. The built-in 'cohesion'
  condition of the authors' code is not a result of the paper, so it only runs
  with --optional.

VERIFY BEFORE TRUSTING
----------------------
  * MAX_STEPS: the paper states no cap (median run 164 steps). 300 is our own limit
    to save time; runs that hit it are flagged (reached_exit = False) and their
    exit time is a lower bound. Check the count in --summarize before using them.
"""
import os

# One thread per process: the runs are independent, so parallelise over runs
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# PaperSettings opens a dialog if results/ is missing, so create the folders first
for _folder in ("results", "output", "tables"):
    os.makedirs(_folder, exist_ok=True)

import matplotlib as mpl
mpl.use("Agg")  # no GUI windows: this script only saves data to disk

import argparse
import contextlib
import copy
import csv
import multiprocessing
import random
import time
import traceback
import zlib
from collections import namedtuple
from os import path

import numpy
import pandas
from scipy.stats import gaussian_kde

from library import PaperSettings
from library import ReadArena
from library import Swarm
from library import Utils
from pyBat import SaveLoad

# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------
REPLICATIONS = 15            # runs per condition (paper: 15)
BASE_BATS = 25               # default swarm size (paper: 25)
BASE_SCALE = 0.1             # metres per pixel, as in SCRIPT_Swarm.py
MAX_STEPS = 300              # our own cap; the paper states none (median run 164 steps)
STEP_SECONDS = 1 / 25        # Swarm.pulse_interval
PROXIMITY = PaperSettings.wingspan   # 0.13 m, the paper's "within 13 cm"
TRIM = PaperSettings.throw_away_steps  # first 50 steps are not counted
EVENT_SEPARATION = 5         # steps; same rule as Utils.count_hits
COHESION_THRESHOLD = 700     # level used by the built-in 'cohesion' condition in Swarm.py
SEED_BASE = 12345
SUMMARY_FILE = "baseline_summary.csv"

# Field reference used in the paper's Discussion (Weesner et al.)
WEESNER = {"nn_mode": 0.47, "nn_mean": 0.69, "pct_within_13cm": 0.55}

PULSE_NAMES = {0: "echoes", 1: "walls", 2: "calls", 3: "echoes2", 4: "walls2", 5: "perched"}

Experiment = namedtuple("Experiment", "baseline arena condition n_bats scale")


def build_experiments(args):
    experiments = [
        # B1, B5, B6: reference for everything
        Experiment("B1/B5/B6 default", "branched", "default", BASE_BATS, BASE_SCALE),
        # B2: arena geometry
        Experiment("B2 curved", "curved", "default", BASE_BATS, BASE_SCALE),
        Experiment("B2 widened", "widened", "default", BASE_BATS, BASE_SCALE),
        # B4: no social acoustic repulsion
        Experiment("B4 no others", "branched", "no_others", BASE_BATS, BASE_SCALE),
    ]
    if args.optional:
        experiments += [
            # The authors' unpublished cohesion switch (not a paper result): turns toward the
            # louder ear below an integrated level of 700 and away from it above
            Experiment("opt cohesion prototype", "branched", "cohesion", BASE_BATS, BASE_SCALE),
            Experiment("opt perched 25%", "branched", "perched", BASE_BATS, BASE_SCALE),
            Experiment("opt perched 50%", "branched", "perched2", BASE_BATS, BASE_SCALE),
            Experiment("opt differential", "branched", "differential", BASE_BATS, BASE_SCALE),
            Experiment("opt muted", "branched", "muted", BASE_BATS, BASE_SCALE),
            Experiment("opt random", "branched", "random", BASE_BATS, BASE_SCALE),
        ]
    return experiments


# Paper medians (Table 1 and Results). Percentages are in percent.
# 'collisions' is compared with both collision_events_* columns.
PAPER_TARGETS = {
    ("branched", "default", 25): {"steps_run": 164, "collisions": 3, "wall_dist_mode": 1.6,
                                   "nn_mode_3d": 0.90, "pct_wall13": 0.43, "pct_agent13": 0.03},
    ("curved", "default", 25): {"collisions": 2, "pct_wall13": 0.57, "pct_agent13": 0.0},
    ("widened", "default", 25): {"collisions": 1, "pct_wall13": 0.54, "pct_agent13": 0.0},
    ("branched", "no_others", 25): {"nn_mode_3d": 0.31, "pct_agent13": 4.8},
    ("branched", "perched", 25): {"collisions": 1, "nn_mode_3d": 0.64, "pct_wall13": 0.21,
                                   "pct_agent13": 0.17},
    ("branched", "perched2", 25): {"collisions": 0, "pct_wall13": 0.0},
    ("branched", "differential", 25): {"collisions": 0, "nn_mode_3d": 0.52, "pct_wall13": 0.0,
                                        "pct_agent13": 0.18},
}

ID_KEYS = ["arena", "condition", "number_of_bats", "arena_scale", "replication",
           "seed", "seconds", "error"]
METRIC_KEYS = [
    # exit / throughput
    "steps_run", "exit_time_s", "reached_exit", "exited_bats",
    # collisions (two definitions, with and without the first 50 steps)
    "wall_crossings_all", "wall_crossings_after50",
    "collision_events_all", "collision_events_after50",
    # proximity (percent of bat-steps, after the first 50 steps)
    "pct_wall13", "pct_agent13",
    # distances (metres, after the first 50 steps)
    "wall_dist_mode", "nn_mode_3d", "nn_mode_xy", "nn_median_3d", "nn_median_xy", "nn_p05_3d",
    # vertical baseline
    "z_std", "z_min", "z_max",
    # speed
    "speed_mean", "frac_below_max_speed",
    # loudness of the integrated level used by the controller (B5)
    "level_q05", "level_q25", "level_q50", "level_q75", "level_q95", "frac_level_above_700",
    # share of steps in which each input type is loudest (Fig 4A)
    "loudest_echoes", "loudest_walls", "loudest_calls", "loudest_echoes2", "loudest_walls2",
]
NAN = float("nan")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def result_filename(arena_name, condition, n_bats, replication):
    """The 'base_' prefix keeps these files apart from runs made with the old script."""
    tag = "" if n_bats == BASE_BATS else "_%dbats" % n_bats
    name = "base_%s_%s%s_%0.3i.pck" % (condition, arena_name, tag, replication)
    return path.join(PaperSettings.results_folder, name)


def make_seed(arena_name, condition, n_bats, replication):
    """Stable seed per run. Swarm.py draws start positions with `random` and
    `numpy.random` and never seeds them, so without this runs cannot be repeated."""
    key = "%s|%s|%d|%d" % (arena_name, condition, n_bats, replication)
    return (SEED_BASE + zlib.crc32(key.encode())) % (2 ** 32)


def arena_file_exists(arena_name):
    return path.exists(path.join(ReadArena.arena_path, arena_name + ".svg"))


def nearest_neighbour(points):
    """Distance from every point to its nearest other point."""
    if len(points) < 2:
        return numpy.array([])
    diff = points[:, None, :] - points[None, :, :]
    dist = numpy.sqrt((diff ** 2).sum(axis=-1))
    numpy.fill_diagonal(dist, numpy.inf)
    return dist.min(axis=1)


def kde_mode(values, low=0.0, high=3.0):
    """Mode of a kernel density estimate; NaN if there is too little data."""
    values = numpy.asarray(values, dtype=float)
    values = values[numpy.isfinite(values)]
    values = values[(values >= low) & (values <= high)]
    if len(values) < 10 or numpy.ptp(values) == 0:
        return NAN
    kde = gaussian_kde(values, bw_method=PaperSettings.bw)
    grid = numpy.linspace(low, high, 600)
    return float(grid[numpy.argmax(kde(grid))])


def count_events(steps):
    """Number of collision events: crossings less than EVENT_SEPARATION steps
    apart count as one (same rule as Utils.count_hits, swarm-wide)."""
    steps = numpy.sort(numpy.asarray(steps, dtype=float))
    if len(steps) == 0:
        return 0
    return int(1 + numpy.sum(numpy.diff(steps) >= EVENT_SEPARATION))


def safe_quantile(values, q):
    values = numpy.asarray(values, dtype=float)
    values = values[numpy.isfinite(values)]
    return float(numpy.quantile(values, q)) if len(values) else NAN


def compute_metrics(swarm, nn_records, steps_run, n_bats):
    """Per-run summary numbers. Anything that cannot be computed stays NaN."""
    m = dict.fromkeys(METRIC_KEYS, NAN)
    m["steps_run"] = steps_run
    m["exit_time_s"] = steps_run * STEP_SECONDS
    m["exited_bats"] = len(swarm.inactive_bats)

    # Wall collisions: rows of outside_positions are [bat id, step, x, y]
    outside = numpy.array(swarm.outside_positions, dtype=float).reshape(-1, 4)
    col_steps = outside[:, 1]
    after = col_steps[col_steps >= TRIM]
    m["wall_crossings_all"] = len(col_steps)
    m["wall_crossings_after50"] = len(after)
    m["collision_events_all"] = count_events(col_steps)
    m["collision_events_after50"] = count_events(after)

    # Nearest-neighbour distances between bats (per bat and step)
    nn3 = [r[1] for r in nn_records if r[0] >= TRIM and len(r[1])]
    nn2 = [r[2] for r in nn_records if r[0] >= TRIM and len(r[2])]
    if nn3:
        nn3 = numpy.concatenate(nn3)
        nn2 = numpy.concatenate(nn2)
        m["nn_mode_3d"] = kde_mode(nn3)
        m["nn_mode_xy"] = kde_mode(nn2)
        m["nn_median_3d"] = float(numpy.median(nn3))
        m["nn_median_xy"] = float(numpy.median(nn2))
        m["nn_p05_3d"] = float(numpy.quantile(nn3, 0.05))

    # Quantities from the per-bat log
    try:
        data_all = swarm.export_bat_data()
        data = data_all[data_all["simulation_step"] >= TRIM]

        if len(data):
            m["pct_wall13"] = float((data["closest_wall"] < PROXIMITY).mean() * 100)
            m["pct_agent13"] = float((data["closest_echo"] < PROXIMITY).mean() * 100)
            m["wall_dist_mode"] = kde_mode(data["closest_wall"])

            level = data["max_integrated"].values
            for name, q in (("q05", 0.05), ("q25", 0.25), ("q50", 0.5), ("q75", 0.75), ("q95", 0.95)):
                m["level_" + name] = safe_quantile(level, q)
            finite = level[numpy.isfinite(level)]
            if len(finite):
                m["frac_level_above_700"] = float((finite > COHESION_THRESHOLD).mean())

            shares = data["pulse_type_code"].dropna().astype(int).value_counts(normalize=True)
            for code, name in PULSE_NAMES.items():
                key = "loudest_" + name
                if key in m:
                    m[key] = float(shares.get(code, 0.0))

            m["speed_mean"] = float(data["speed"].mean())
            m["frac_below_max_speed"] = float((data["speed"] < swarm.max_speed - 1e-6).mean())

        if "z" in data_all.columns and len(data_all):
            m["z_std"] = float(data_all["z"].std())
            m["z_min"] = float(data_all["z"].min())
            m["z_max"] = float(data_all["z"].max())
    except Exception as error:  # keep the run even if the log export fails
        print("    WARNING: could not compute log-based metrics: %r" % error)
    return m


def run_one(arena_name, condition, n_bats, arena_scale, replication,
            max_steps, save=True):
    """One simulation run. The loop mirrors SCRIPT_Swarm.py, with two changes:
    (1) perched bats are only regenerated every step for the perched conditions
    (for the others one set is created once, because Swarm.get_pulses_single_bat
    always computes the 'perched' pulses and needs a non-empty list);
    (2) nearest-neighbour distances are stored per step so the first 50 steps
    can be discarded afterwards."""
    seed = make_seed(arena_name, condition, n_bats, replication)
    random.seed(seed)
    numpy.random.seed(seed)

    arena = ReadArena.read_arena(arena_name, scale=arena_scale, plot=False)
    perched_proportion = {"perched": 0.25, "perched2": 0.5}.get(condition)

    nn_records = []   # (step, nn distance 3D, nn distance in the horizontal plane)
    number_of_active_bats = []
    # Swarm.simulation_step prints one line per bat per step; silence it
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        swarm = Swarm.Swarm(n_bats, arena=arena, condition=condition,
                            replication=replication)
        if perched_proportion is None:
            swarm.set_perched_bats(proportion=0.25)
        for step in range(max_steps):
            number_of_active_bats.append(len(swarm.active_bats))
            if len(swarm.active_bats) < n_bats / 2:
                break
            if perched_proportion is not None:
                swarm.set_perched_bats(proportion=perched_proportion)
            swarm.simulation_step()
            if arena_name != "open":
                swarm.remove_bats_in_end_region()
            positions = swarm.bat_positions
            if len(positions) >= 2:
                nn_records.append((step, nearest_neighbour(positions),
                                   nearest_neighbour(positions[:, :2])))

    steps_run = swarm.simulation_step_nr
    reached_exit = len(swarm.active_bats) < n_bats / 2
    metrics = compute_metrics(swarm, nn_records, steps_run, n_bats)
    metrics["reached_exit"] = bool(reached_exit)

    result = {
        "number_of_bats": n_bats,
        "steps": steps_run,
        "swarm": copy.copy(swarm),
        "arena_name": arena_name,
        "arena": arena,
        "replication": replication,
        "number_of_active_bats": number_of_active_bats,
        "condition": condition,
        "nn_records": nn_records,        # (step, nn_3d, nn_xy) per step, for plots
        "reached_exit": reached_exit,
        "max_steps": max_steps,
        "arena_scale": arena_scale,
        "seed": seed,
        "metrics": metrics,
    }
    if save:
        SaveLoad.pickle_save(result_filename(arena_name, condition, n_bats,
                                             replication), result)
    return result


def run_task(task):
    """Worker entry point. Returns one CSV row; never raises, so one failed run
    cannot stop the others."""
    arena_name, condition, n_bats, scale, replication, max_steps = task
    row = {"arena": arena_name, "condition": condition, "number_of_bats": n_bats,
           "arena_scale": scale, "replication": replication,
           "seed": make_seed(arena_name, condition, n_bats, replication),
           "seconds": NAN, "error": ""}
    row.update(dict.fromkeys(METRIC_KEYS, NAN))
    t0 = time.time()
    try:
        result = run_one(arena_name, condition, n_bats, scale, replication, max_steps)
        row.update(result["metrics"])
    except Exception:
        traceback.print_exc()
        row["error"] = traceback.format_exc().strip().splitlines()[-1]
    row["seconds"] = round(time.time() - t0, 1)
    return row


def append_summary(row):
    summary_file = path.join(PaperSettings.results_folder, SUMMARY_FILE)
    is_new = not path.exists(summary_file)
    with open(summary_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ID_KEYS + METRIC_KEYS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


# --------------------------------------------------------------------------
# Dry run: plan and arena geometry
# --------------------------------------------------------------------------
def describe_arenas(experiments):
    print("\nArena geometry:")
    seen = set()
    for exp in experiments:
        key = (exp.arena, exp.scale)
        if key in seen or not arena_file_exists(exp.arena):
            continue
        seen.add(key)
        arena = ReadArena.read_arena(exp.arena, scale=exp.scale, plot=False)
        edge = arena["edge_points"]
        width = edge[:, 0].max() - edge[:, 0].min()
        height = edge[:, 1].max() - edge[:, 1].min()
        surface = arena["start_surface"]
        print("  %-9s scale=%.4f  extent %.1f x %.1f m  start surface %.1f m2  "
              "-> %d bats = %.2f bats/m2 of start region"
              % (exp.arena, exp.scale, width, height, surface, exp.n_bats,
                 exp.n_bats / surface if surface else float("nan")))


# --------------------------------------------------------------------------
# Summary against the paper
# --------------------------------------------------------------------------
REPORT = [
    # (column, label, paper-target key)
    ("steps_run", "steps until >50% exited", "steps_run"),
    ("exit_time_s", "exit time (s)", None),
    ("collision_events_all", "collision events (all steps)", "collisions"),
    ("collision_events_after50", "collision events (after step 50)", "collisions"),
    ("wall_crossings_all", "wall crossings (all steps)", None),
    ("pct_wall13", "% steps < 13 cm from wall", "pct_wall13"),
    ("pct_agent13", "% steps < 13 cm from bat", "pct_agent13"),
    ("wall_dist_mode", "wall distance mode (m)", "wall_dist_mode"),
    ("nn_mode_3d", "bat-bat distance mode, 3D (m)", "nn_mode_3d"),
    ("nn_mode_xy", "bat-bat distance mode, horizontal (m)", None),
    ("nn_median_3d", "bat-bat distance median, 3D (m)", None),
    ("level_q50", "integrated level, median", None),
    ("level_q95", "integrated level, 95th percentile", None),
    ("frac_level_above_700", "share of steps with level > 700", None),
    ("loudest_calls", "share loudest = others' calls", None),
    ("loudest_echoes2", "share loudest = secondary body echoes", None),
    ("loudest_echoes", "share loudest = own body echoes", None),
    ("loudest_walls", "share loudest = wall echoes", None),
    ("z_std", "height std (m)", None),
    ("speed_mean", "mean speed (m/s)", None),
]


def summarize():
    summary_file = path.join(PaperSettings.results_folder, SUMMARY_FILE)
    if not path.exists(summary_file):
        print("No %s yet. Run the simulations first." % summary_file)
        return
    df = pandas.read_csv(summary_file)
    if "error" in df.columns:
        failed = df["error"].notna() & (df["error"].astype(str) != "")
        if failed.any():
            print("NOTE: %d failed runs are excluded." % failed.sum())
            df = df[~failed]
    rows = []
    for (arena, condition, n_bats), group in df.groupby(["arena", "condition", "number_of_bats"]):
        targets = PAPER_TARGETS.get((arena, condition, int(n_bats)), {})
        print("\n=== arena=%s condition=%s bats=%d  (%d runs, %d reached the exit) ==="
              % (arena, condition, n_bats, len(group), int(group["reached_exit"].sum())))
        print("  %-42s %-30s %s" % ("metric", "ours: median [min, max]", "paper median"))
        for column, label, target_key in REPORT:
            if column not in group.columns:
                continue
            values = group[column].dropna()
            if values.empty:
                continue
            ours = "%.3g [%.3g, %.3g]" % (values.median(), values.min(), values.max())
            target = targets.get(target_key) if target_key else None
            print("  %-42s %-30s %s" % (label, ours, "" if target is None else "%g" % target))
            rows.append({"arena": arena, "condition": condition, "number_of_bats": n_bats,
                         "metric": column, "median": values.median(), "min": values.min(),
                         "max": values.max(), "paper_median": target})
    print("\nField reference (Weesner et al., from the paper's Discussion): "
          "nearest-neighbour mode %.2f m, mean %.2f m, %.2f%% of distances within 13 cm."
          % (WEESNER["nn_mode"], WEESNER["nn_mean"], WEESNER["pct_within_13cm"]))
    print("Note: the paper does not say whether collisions exclude the first 50 steps, "
          "so both versions are shown; compare the one closer to the paper's.")
    out_file = path.join("tables", "baseline_comparison.csv")
    pandas.DataFrame(rows).to_csv(out_file, index=False)
    print("Saved %s" % out_file)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replications", type=int, default=REPLICATIONS)
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--workers", type=int, default=1,
                        help="number of runs to simulate in parallel (default 1)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the run plan and arena geometry, then exit")
    parser.add_argument("--smoke-test", action="store_true",
                        help="5 bats, 5 steps per experiment, nothing saved")
    parser.add_argument("--overwrite", action="store_true",
                        help="rerun runs whose result file already exists")
    parser.add_argument("--optional", action="store_true",
                        help="also run perched, differential, muted and random "
                             "(not baselines for the extensions)")
    parser.add_argument("--summarize", action="store_true",
                        help="compare finished runs with the paper and exit")
    args = parser.parse_args()

    if args.summarize:
        summarize()
        return

    experiments = build_experiments(args)
    total = len(experiments) * args.replications
    print("Plan: %d experiments x %d replications = %d runs, max %d steps, %d worker(s)"
          % (len(experiments), args.replications, total, args.max_steps, args.workers))
    missing = []
    for exp in experiments:
        found = arena_file_exists(exp.arena)
        if not found:
            missing.append(exp.arena)
        print("  %-24s arena=%-9s condition=%-13s bats=%-3d scale=%.4f  svg=%s"
              % (exp.baseline, exp.arena, exp.condition, exp.n_bats, exp.scale,
                 "ok" if found else "MISSING"))
    if missing:
        print("\nMissing arena files in %s: %s"
              % (path.abspath(ReadArena.arena_path), sorted(set(missing))))
        return
    if args.dry_run:
        describe_arenas(experiments)
        return

    Utils.create_folder(PaperSettings.results_folder)

    if args.smoke_test:
        for exp in experiments:
            run_one(exp.arena, exp.condition, 5, exp.scale, 0, 5, save=False)
            print("OK: %s / %s" % (exp.arena, exp.condition))
        print("Smoke test finished.")
        return

    tasks = []
    for exp in experiments:
        for replication in range(args.replications):
            out_file = result_filename(exp.arena, exp.condition, exp.n_bats, replication)
            if path.exists(out_file) and not args.overwrite:
                continue
            tasks.append((exp.arena, exp.condition, exp.n_bats, exp.scale,
                          replication, args.max_steps))
    print("\n%d runs to do, %d already finished.\n" % (len(tasks), total - len(tasks)))
    if not tasks:
        print("Nothing to run. Use --summarize to compare with the paper.")
        return

    start = time.time()

    def handle(row, done):
        append_summary(row)
        status = "FAILED: " + row["error"] if row["error"] else (
            "steps=%d exit=%s events=%d" % (row["steps_run"], row["reached_exit"],
                                            row["collision_events_all"]))
        print("[%d/%d] %s/%s/%dbats rep %d | %s | %.0f s (elapsed %.0f min)"
              % (done, len(tasks), row["arena"], row["condition"], row["number_of_bats"],
                 row["replication"], status, row["seconds"], (time.time() - start) / 60))
        if not row["error"] and not row["reached_exit"]:
            print("    WARNING: hit the step cap before half the bats exited; "
                  "consider raising --max-steps")

    if args.workers > 1:
        with multiprocessing.Pool(args.workers) as pool:
            for done, row in enumerate(pool.imap_unordered(run_task, tasks), start=1):
                handle(row, done)
    else:
        for done, task in enumerate(tasks, start=1):
            handle(run_task(task), done)

    print("\nFinished. Run 'python SCRIPT_Replicate_Paper.py --summarize' to compare with the paper.")


if __name__ == "__main__":
    main()
