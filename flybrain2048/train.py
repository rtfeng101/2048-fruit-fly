"""Train the fly brain to play 2048 with PPO (actor-critic policy gradient).

Each update: play `batch_games` games in parallel until they all end, then take a
few passes over those moves. The fly's "dopamine" critic predicts how
much reward is still to come; moves that turned out better than it expected become
more likely, worse ones less likely, and PPO's clipping keeps each step small.
"""
import csv
import os
import time

import numpy as np
import torch
from torch.distributions import Categorical

from .connectome import load_connectome
from .game import SYM_ACTIONS, Game2048, encode_board, transform_board
from .model import FlyBrainNet, masked_logits


def build_model(cfg, conn):
    m = cfg["model"]
    return FlyBrainNet(conn, obs_dim=256, steps=m["steps"], dt=m["dt"],
                       train_synapses=m["train_synapses"], w_scale=m["w_scale"],
                       normalize_readout=m.get("normalize_readout", False),
                       readout=m.get("readout", "grouped"), critic=m.get("critic", False))


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


def play_batch(model, n_games, rng, max_moves, greedy=False, augment=False):
    """Play n_games to the end, without gradients. With augment, each move is chosen
    on a randomly rotated/mirrored copy of the board (and mapped back).

    Returns the games and, per game, what PPO learns from: the observation the fly
    saw, its valid moves and move (both in that possibly transformed frame), the move's
    log-probability, the critic's value and the reward.
    """
    games = [Game2048(seed=int(rng.integers(1 << 31))) for _ in range(n_games)]
    keys = ("obs", "valid", "act", "logp", "value", "reward")
    traj = [{k: [] for k in keys} for _ in games]
    device = next(model.parameters()).device
    W = model.effective_weights()  # computed once, reused for every move
    for t in range(max_moves):
        active = [i for i, g in enumerate(games) if not g.done]
        if not active:
            break
        sym = rng.integers(8, size=len(active)) if augment else np.zeros(len(active), int)
        obs = np.stack([encode_board(transform_board(games[i].board, s))
                        for i, s in zip(active, sym)])
        valid = np.zeros((len(active), 4), bool)
        for j, (i, s) in enumerate(zip(active, sym)):
            valid[j, SYM_ACTIONS[s]] = games[i].valid_moves()
        if model.dopamine is not None:
            logits, value = model.policy_value(torch.tensor(obs, device=device), W)
        else:  # older checkpoints without a critic can still be played / evaluated
            logits = model(torch.tensor(obs, device=device), W=W)
            value = torch.zeros(len(active), device=device)
        logits = masked_logits(logits, valid)
        dist = Categorical(logits=logits)
        acts = logits.argmax(-1) if greedy else dist.sample()
        logp, acts, value = dist.log_prob(acts).cpu().numpy(), acts.cpu().numpy(), value.cpu().numpy()
        for j, (i, s) in enumerate(zip(active, sym)):
            real = int(np.flatnonzero(SYM_ACTIONS[s] == acts[j])[0])  # back to the real board
            _, gained, done, _ = games[i].step(real)
            for k, v in zip(keys, (obs[j], valid[j], acts[j], logp[j], value[j],
                                   shaped_reward(gained, done, t))):
                traj[i][k].append(v)
    return games, traj


def gae(rewards, values, gamma, lam):
    """Generalized advantage estimation for one game (which ends at its last move).

    delta_t = r_t + gamma * V(t+1) - V(t) is the reward-prediction error, the signal
    dopamine neurons are thought to carry. Advantages are discounted sums of it."""
    adv, last = np.zeros(len(rewards), np.float32), 0.0
    for t in reversed(range(len(rewards))):
        next_v = values[t + 1] if t + 1 < len(rewards) else 0.0
        last = rewards[t] + gamma * next_v - values[t] + gamma * lam * last
        adv[t] = last
    return adv, adv + np.asarray(values, np.float32)


def ppo_update(model, opt, batch, tc):
    """A few passes of clipped-PPO minibatch steps over one batch of moves.
    Returns mean (policy loss, value loss, entropy)."""
    n = len(batch["act"])
    stats = []
    for _ in range(tc.get("ppo_epochs", 4)):
        for mb in torch.randperm(n, device=batch["act"].device).split(tc.get("minibatch", 2048)):
            logits, value = model.policy_value(batch["obs"][mb])
            dist = Categorical(logits=masked_logits(logits, batch["valid"][mb]))
            adv = batch["adv"][mb]
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            ratio = torch.exp(dist.log_prob(batch["act"][mb]) - batch["logp"][mb])
            clip = tc.get("clip", 0.2)
            pg = -torch.min(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv).mean()
            # Huber loss: returns are large early on, and this keeps the critic from
            # swamping the policy's share of the (clipped) gradient.
            v_loss = torch.nn.functional.smooth_l1_loss(value, batch["ret"][mb])
            ent = dist.entropy().mean()
            loss = pg + tc.get("value_coef", 0.5) * v_loss - tc["entropy_coef"] * ent

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            stats.append((pg.item(), v_loss.item(), ent.item()))
    return np.mean(stats, 0)


def train(cfg, resume=False):
    """Run `train.updates` updates, starting fresh or (resume=True) continuing from
    runs/<run_name>/last.pt, appending to the same metrics.csv."""
    tc = cfg["train"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    conn = load_connectome(cfg["connectome"])
    print("Connectome:", conn.summary())

    run_dir = os.path.join("runs", tc["run_name"])
    last_path, metrics_path = os.path.join(run_dir, "last.pt"), os.path.join(run_dir, "metrics.csv")
    start, best = 0, -1
    if resume:
        if not os.path.exists(last_path):
            raise SystemExit(f"Nothing to resume: {last_path} doesn't exist.")
        model, ckpt = load_model(conn, last_path)
        if model.dopamine is None:
            raise SystemExit(f"{run_dir}/ was trained with the old REINFORCE setup (no dopamine "
                             f"critic), which PPO needs. Start a new train.run_name instead.")
        model.to(device)
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
        if not cfg["model"].get("critic", False):
            raise SystemExit("PPO training needs the dopamine critic: set model.critic: true.")
        model = build_model(cfg, conn)
        model.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=tc["lr"])

    # Offset the seed so a resumed run doesn't replay the same games.
    torch.manual_seed(tc["seed"] + start)
    rng = np.random.default_rng(tc["seed"] + start)

    os.makedirs(run_dir, exist_ok=True)
    log = open(metrics_path, "a" if resume else "w", newline="")
    writer = csv.writer(log)
    if not resume:
        writer.writerow(["update", "mean_score", "max_tile", "mean_moves", "entropy", "loss",
                         "value_loss", "secs"])

    for update in range(start + 1, start + tc["updates"] + 1):
        t0 = time.time()
        with torch.no_grad():
            games, traj = play_batch(model, tc["batch_games"], rng, tc["max_moves"],
                                     augment=tc.get("augment", False))

        # Rewards are scaled down so the critic's targets stay near unit size.
        scale, adv, ret = tc.get("reward_scale", 0.1), [], []
        for tr in traj:
            a, r = gae(scale * np.array(tr["reward"]), tr["value"], tc["gamma"],
                       tc.get("gae_lambda", 0.95))
            adv.append(a)
            ret.append(r)
        batch = {k: torch.tensor(np.concatenate([tr[k] for tr in traj]), device=device)
                 for k in ("obs", "valid", "act", "logp")}
        batch["adv"] = torch.tensor(np.concatenate(adv), device=device)
        batch["ret"] = torch.tensor(np.concatenate(ret), device=device)
        pg, v_loss, ent = ppo_update(model, opt, batch, tc)

        scores = [g.score for g in games]
        row = [update, np.mean(scores), max(g.max_tile for g in games),
               np.mean([g.moves for g in games]), ent, pg, v_loss, time.time() - t0]
        writer.writerow(row)
        log.flush()
        if update % tc["print_every"] == 0 or update == 1:
            print(f"[{update:5d}] score {row[1]:8.1f} | best tile {row[2]:5d} | "
                  f"moves {row[3]:6.1f} | entropy {row[4]:.3f} | critic loss {row[6]:.3f} | "
                  f"{row[7]:.1f}s")
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device:", device)
        conn = load_connectome(cfg["connectome"])
        model = load_model(conn, checkpoint)[0] if checkpoint else build_model(cfg, conn)
        model.to(device)
        model.eval()
        with torch.no_grad():
            games, _ = play_batch(model, n_games, rng, 100000, greedy=True)
        scores, tiles = [g.score for g in games], [g.max_tile for g in games]
        label = checkpoint or "untrained fly"
    vals, counts = np.unique(tiles, return_counts=True)
    print(f"{label}: mean score {np.mean(scores):.0f}, median {np.median(scores):.0f}")
    print("  max-tile distribution:", dict(zip(vals.tolist(), counts.tolist())))
