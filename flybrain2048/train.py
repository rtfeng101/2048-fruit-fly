"""Train the fly brain to play 2048 with REINFORCE (policy gradient).

Each update: play `batch_games` games in parallel until they all end, then
nudge the learnable parameters so moves followed by high reward become more
likely. Simple, noisy, and a great baseline to improve on (see README ideas).
"""
import csv
import os
import time

import numpy as np
import torch
from torch.distributions import Categorical

from .connectome import load_connectome
from .game import Game2048, encode_board
from .model import FlyBrainNet, masked_logits


def build_model(cfg, conn):
    m = cfg["model"]
    return FlyBrainNet(conn, obs_dim=256, steps=m["steps"], dt=m["dt"],
                       train_synapses=m["train_synapses"], w_scale=m["w_scale"],
                       normalize_readout=m.get("normalize_readout", False))


def save_checkpoint(model, conn, cfg, path, **extra):
    torch.save({"model": model.state_dict(), "cfg": cfg, "body_ids": conn.body_ids, **extra}, path)


def load_model(conn, path):
    """Rebuild the model a checkpoint was trained with and load its weights.

    Model settings come from the checkpoint, not config.yaml, so a run keeps
    behaving the way it was trained. Returns (model, checkpoint dict).
    """
    ckpt = torch.load(path, weights_only=False)
    ids = ckpt.get("body_ids")
    same = (np.array_equal(ids, conn.body_ids) if ids is not None
            else ckpt["model"]["bias"].shape[0] == conn.n)
    if not same:
        n_ckpt = len(ids) if ids is not None else ckpt["model"]["bias"].shape[0]
        raise SystemExit(
            f"{path} was trained on a different connectome ({n_ckpt} neurons) than the one "
            f"in your cache now ({conn.n} neurons).\nRetrain with `python -m flybrain2048 train`, "
            f"or re-fetch the connectome that checkpoint was trained on.")
    model = build_model(ckpt["cfg"], conn)
    model.load_state_dict(ckpt["model"])
    return model, ckpt


def shaped_reward(gained, done, t):
    # log2 keeps huge merges from dominating; small penalty for dying.
    r = np.log2(gained) if gained > 0 else 0.0
    return r - (10.0 if done else 0.0)


def play_batch(model, n_games, rng, max_moves, greedy=False):
    games = [Game2048(seed=int(rng.integers(1 << 31))) for _ in range(n_games)]
    logps = [[] for _ in games]
    rewards = [[] for _ in games]
    entropies = []
    W = model.effective_weights()  # computed once, reused for every move
    for t in range(max_moves):
        active = [i for i, g in enumerate(games) if not g.done]
        if not active:
            break
        obs = torch.tensor(np.stack([encode_board(games[i].board) for i in active]))
        valid = np.stack([games[i].valid_moves() for i in active])
        logits = masked_logits(model(obs, W=W), valid)
        dist = Categorical(logits=logits)
        acts = logits.argmax(-1) if greedy else dist.sample()
        lp = dist.log_prob(acts)
        entropies.append(dist.entropy().mean())
        for j, i in enumerate(active):
            _, gained, done, _ = games[i].step(int(acts[j]))
            logps[i].append(lp[j])
            rewards[i].append(shaped_reward(gained, done, t))
    return games, logps, rewards, torch.stack(entropies).mean()


def discounted(rs, gamma):
    out, g = np.zeros(len(rs), dtype=np.float32), 0.0
    for k in reversed(range(len(rs))):
        g = rs[k] + gamma * g
        out[k] = g
    return out


def train(cfg, resume=False):
    """Run `train.updates` updates, starting fresh or (resume=True) continuing from
    runs/<run_name>/last.pt, appending to the same metrics.csv."""
    tc = cfg["train"]
    conn = load_connectome(cfg["connectome"])
    print("Connectome:", conn.summary())

    run_dir = os.path.join("runs", tc["run_name"])
    last_path, metrics_path = os.path.join(run_dir, "last.pt"), os.path.join(run_dir, "metrics.csv")
    start, best = 0, -1
    if resume:
        if not os.path.exists(last_path):
            raise SystemExit(f"Nothing to resume: {last_path} doesn't exist.")
        model, ckpt = load_model(conn, last_path)
        if ckpt["cfg"]["model"] != cfg["model"]:
            print("  (using this run's original `model:` settings; changes to them in "
                  "config.yaml need a new run_name)")
            cfg = {**cfg, "model": ckpt["cfg"]["model"]}
        opt = torch.optim.Adam(model.parameters(), lr=tc["lr"])
        if "opt" in ckpt:
            opt.load_state_dict(ckpt["opt"])
        else:
            print("  (older checkpoint without optimizer state; Adam restarts from zero)")
        # Older checkpoints don't store progress; recover it from the metrics log.
        with open(metrics_path) as f:
            logged = [row for row in csv.DictReader(f)]
        start = ckpt.get("update", int(logged[-1]["update"]) if logged else 0)
        best = ckpt.get("best", max((float(r["mean_score"]) for r in logged), default=-1))
        print(f"Resuming {run_dir} from update {start} (best mean score so far {best:.0f})")
    elif os.path.exists(last_path):
        raise SystemExit(f"{run_dir}/ already has a trained model. Continue it with "
                         f"`train --resume`, or set a new train.run_name in config.yaml.")
    else:
        model = build_model(cfg, conn)
        opt = torch.optim.Adam(model.parameters(), lr=tc["lr"])

    # Offset the seed so a resumed run doesn't replay the same games.
    torch.manual_seed(tc["seed"] + start)
    rng = np.random.default_rng(tc["seed"] + start)

    os.makedirs(run_dir, exist_ok=True)
    log = open(metrics_path, "a" if resume else "w", newline="")
    writer = csv.writer(log)
    if not resume:
        writer.writerow(["update", "mean_score", "max_tile", "mean_moves", "entropy", "loss", "secs"])

    for update in range(start + 1, start + tc["updates"] + 1):
        t0 = time.time()
        games, logps, rewards, ent = play_batch(model, tc["batch_games"], rng, tc["max_moves"])

        # Returns, normalized across the whole batch (acts as a baseline).
        rets = [discounted(r, tc["gamma"]) for r in rewards]
        flat = np.concatenate(rets)
        mu, sd = flat.mean(), flat.std() + 1e-8
        loss = 0.0
        for lp, R in zip(logps, rets):
            adv = torch.tensor((R - mu) / sd)
            loss = loss - (torch.stack(lp) * adv).sum()
        loss = loss / len(flat) - tc["entropy_coef"] * ent

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        scores = [g.score for g in games]
        row = [update, np.mean(scores), max(g.max_tile for g in games),
               np.mean([g.moves for g in games]), ent.item(), loss.item(), time.time() - t0]
        writer.writerow(row)
        log.flush()
        if update % tc["print_every"] == 0 or update == 1:
            print(f"[{update:5d}] score {row[1]:8.1f} | best tile {row[2]:5d} | "
                  f"moves {row[3]:6.1f} | entropy {row[4]:.3f} | {row[6]:.1f}s")
        if np.mean(scores) > best:
            best = float(np.mean(scores))
            save_checkpoint(model, conn, cfg, os.path.join(run_dir, "best.pt"),
                            update=update, best=best)
        save_checkpoint(model, conn, cfg, last_path,
                        opt=opt.state_dict(), update=update, best=best)
        # Numbered snapshots you can go back to (last.pt is overwritten every update).
        every = tc.get("checkpoint_every", 0)
        if every and update % every == 0:
            save_checkpoint(model, conn, cfg, os.path.join(run_dir, f"update_{update:06d}.pt"),
                            opt=opt.state_dict(), update=update, best=best)
    log.close()
    print(f"Done. Checkpoints in {run_dir}/")


def evaluate(cfg, checkpoint=None, n_games=50, random_agent=False):
    """Compare the trained fly against a random-move baseline."""
    rng = np.random.default_rng(123)
    if random_agent:
        scores, tiles = [], []
        for _ in range(n_games):
            g = Game2048(seed=int(rng.integers(1 << 31)))
            while not g.done:
                g.step(int(rng.choice(np.flatnonzero(g.valid_moves()))))
            scores.append(g.score)
            tiles.append(g.max_tile)
        label = "random"
    else:
        conn = load_connectome(cfg["connectome"])
        model = load_model(conn, checkpoint)[0] if checkpoint else build_model(cfg, conn)
        model.eval()
        with torch.no_grad():
            games, *_ = play_batch(model, n_games, rng, 100000, greedy=True)
        scores, tiles = [g.score for g in games], [g.max_tile for g in games]
        label = checkpoint or "untrained fly"
    vals, counts = np.unique(tiles, return_counts=True)
    print(f"{label}: mean score {np.mean(scores):.0f}, median {np.median(scores):.0f}")
    print("  max-tile distribution:", dict(zip(vals.tolist(), counts.tolist())))
