"""Graphs of how training is going, saved as PNGs in runs/<run_name>/plots/.

training.png     from metrics.csv: score, game length, 2048 rate, entropy, critic loss and
                 time per update, each as the raw per-update value plus a rolling mean.
                 Several runs can be drawn on the same axes to compare them.
performance.png  (with --eval) every saved checkpoint (update_NNNNNN.pt and last.pt) plays
                 the same set of games, always taking the most likely move, like `eval` and
                 `play`. Shows the spread of scores, how often games end early, and which
                 tiles the latest checkpoint reaches. Games already played are cached in
                 runs/<run_name>/eval_games.csv, so only new checkpoints are played.
"""
import glob
import os

import matplotlib

matplotlib.use("Agg")  # files only, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
EARLY_TILE = 256  # a game whose best tile is this or lower counts as ending early

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 10,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titlelocation": "left", "axes.titlepad": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "lines.linewidth": 2, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    "legend.frameon": False, "legend.labelcolor": INK_2,
})


def training_figure(runs, window, out_path):
    """runs: {name: metrics DataFrame}. One panel per metric; each run is one colour."""
    panels = [  # (column or derived key, title)
        ("mean_score", "Mean score per update"),
        ("mean_moves", "Moves per game"),
        ("hit_2048", "Updates whose best tile reached 2048 (%)"),
        ("entropy", "Policy entropy (lower = more decisive)"),
        ("value_loss", "Critic loss"),
        ("secs", "Seconds per update"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex=True)
    for (key, title), ax in zip(panels, axes.flat):
        ax.set_title(title)
        drawn = False
        for color, (name, d) in zip(SERIES, runs.items()):
            y = (d["max_tile"] >= 2048) * 100.0 if key == "hit_2048" else d.get(key)
            if y is None:  # older runs don't log critic loss
                continue
            if key != "hit_2048":  # a 0/100 per-update value is just noise; only its average
                ax.plot(d["update"], y, color=color, linewidth=1, alpha=0.25)
            ax.plot(d["update"], y.rolling(window, min_periods=1).mean(), color=color, label=name)
            drawn = True
        if key == "hit_2048":
            ax.set_ylim(0, max(ax.get_ylim()[1], 5))
        if not drawn:
            ax.text(0.5, 0.5, "not logged by this run", transform=ax.transAxes,
                    ha="center", va="center", color=MUTED)
    for ax in axes[1]:
        ax.set_xlabel("update")
    names = " vs ".join(runs)
    fig.suptitle(f"Training: {names}", x=0.01, ha="left", fontsize=14, color=INK)
    fig.text(0.01, 0.94, "Faint line: each update's batch of games, moves sampled.  "
             f"Solid line: rolling mean over {window} updates.", color=INK_2)
    if len(runs) > 1:
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper right", ncol=len(runs))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def evaluate_checkpoints(cfg, run_dir, n_games):
    """Play n_games greedy games with every checkpoint not yet in eval_games.csv.
    Every checkpoint gets the same game seeds, so differences come from the model."""
    from .connectome import load_connectome
    from .train import load_model, play_batch

    cache = os.path.join(run_dir, "eval_games.csv")
    done = pd.read_csv(cache) if os.path.exists(cache) else pd.DataFrame(
        columns=["update", "game", "score", "max_tile", "moves"])
    counts = done.groupby("update").size()
    paths = sorted(glob.glob(os.path.join(run_dir, "update_*.pt"))) + [os.path.join(run_dir, "last.pt")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conn = load_connectome(cfg["connectome"])
    for path in paths:
        if not os.path.exists(path):
            continue
        name = os.path.basename(path)
        if name.startswith("update_") and counts.get(int(name[7:-3]), 0) == n_games:
            continue  # skip loading it at all
        try:
            model, ckpt = load_model(conn, path)
        except (RuntimeError, EOFError) as e:  # e.g. last.pt mid-write while training runs
            print(f"  skipping {name}: {e}")
            continue
        update = ckpt.get("update")
        if update is None or counts.get(update, 0) == n_games:
            continue
        print(f"  {name} (update {update}): playing {n_games} games...")
        model.to(device).eval()
        with torch.no_grad():
            games, _ = play_batch(model, n_games, np.random.default_rng(123), 100000, greedy=True)
        new = pd.DataFrame({"update": update, "game": range(n_games),
                            "score": [g.score for g in games],
                            "max_tile": [g.max_tile for g in games],
                            "moves": [g.moves for g in games]})
        done = pd.concat([done[done["update"] != update], new], ignore_index=True)
        done.sort_values(["update", "game"]).to_csv(cache, index=False)
        counts = done.groupby("update").size()
    return done.astype(int)


def performance_figure(games, run, out_path):
    """games: one row per greedy game, with the checkpoint's update."""
    by = games.groupby("update")["score"]
    q = by.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).unstack()
    early = games.groupby("update")["max_tile"].apply(lambda t: (t <= EARLY_TILE).mean() * 100)
    latest = games[games["update"] == games["update"].max()]
    n = len(latest)

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15, 4.8), width_ratios=(1.6, 1, 1))
    c = SERIES[0]
    if len(q) > 1:
        a1.fill_between(q.index, q[0.1], q[0.9], color=c, alpha=0.10, linewidth=0,
                        label="10th–90th percentile")
        a1.fill_between(q.index, q[0.25], q[0.75], color=c, alpha=0.20, linewidth=0,
                        label="25th–75th percentile")
        a1.plot(q.index, q[0.5], color=c, marker="o", markersize=8, markeredgecolor=SURFACE,
                markeredgewidth=2, label="median")
        a1.set_title("Score per game, by checkpoint")
        a1.set_xlabel("update")
        a1.legend(loc="upper left")
    else:  # one checkpoint: no trend to draw, so show how its scores spread instead
        a1.hist(latest["score"], bins=20, color=c, rwidth=0.9)
        med = latest["score"].median()
        a1.axvline(med, color=INK_2, linewidth=1, linestyle="--")
        a1.annotate(f"median {med:,.0f}", (med, 1), xycoords=("data", "axes fraction"),
                    xytext=(4, -4), textcoords="offset points", va="top", color=INK_2)
        a1.set_title(f"Score per game, update {latest['update'].iloc[0]}")
        a1.set_xlabel("score")
        a1.set_ylabel("games")

    a2.plot(early.index, early.values, color=c, marker="o", markersize=8,
            markeredgecolor=SURFACE, markeredgewidth=2)
    a2.set_title(f"Games ending at tile {EARLY_TILE} or lower (%)")
    a2.set_xlabel("update")
    a2.set_ylim(bottom=0)

    tiles, counts = np.unique(latest["max_tile"], return_counts=True)
    pct = counts / n * 100
    bars = a3.bar([str(t) for t in tiles], pct, width=0.6, color=c)
    a3.bar_label(bars, labels=[f"{p:.0f}%" for p in pct], padding=3, color=INK_2)
    a3.set_title(f"Best tile reached, update {latest['update'].iloc[0]}")
    a3.set_xlabel("best tile")
    a3.set_ylabel("% of games")
    a3.margins(y=0.15)

    fig.suptitle(f"Greedy play: {run}", x=0.01, ha="left", fontsize=14, color=INK)
    fig.text(0.01, 0.885, f"Each checkpoint plays the same {n} games, always taking its most "
             f"likely move (as in eval and play)."
             + ("" if len(q) > 1 else "  Only one checkpoint so far: set train.checkpoint_every "
                "(e.g. 50) to see progress across checkpoints."), color=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_plots(cfg, runs=None, window=25, run_eval=False, n_games=50):
    runs = runs or [cfg["train"]["run_name"]]
    metrics = {}
    for name in runs:
        path = os.path.join("runs", name, "metrics.csv")
        if not os.path.exists(path):
            raise SystemExit(f"No {path}: has `{name}` been trained?")
        metrics[name] = pd.read_csv(path)
    out_dir = os.path.join("runs", runs[0], "plots")
    os.makedirs(out_dir, exist_ok=True)

    path = os.path.join(out_dir, "training.png" if len(runs) == 1 else "training_compare.png")
    training_figure(metrics, window, path)
    print("Saved", path)

    if run_eval:
        # Only the first run: its checkpoints must match the circuit in --config.
        print(f"Evaluating checkpoints in runs/{runs[0]}/ ...")
        games = evaluate_checkpoints(cfg, os.path.join("runs", runs[0]), n_games)
        if games.empty:
            print("No checkpoints to evaluate.")
            return
        path = os.path.join(out_dir, "performance.png")
        performance_figure(games, runs[0], path)
        print("Saved", path)
