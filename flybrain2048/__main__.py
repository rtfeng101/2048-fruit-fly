"""Command-line entry point:  python -m flybrain2048 <command>

  fetch   build/cache the connectome sub-circuit (use --refresh to rebuild it,
          --shapes to also download real neuron shapes and synapse locations
          for the display; shapes already downloaded are reused unless
          --refresh-shapes)
  train   train with PPO, checkpoints go to runs/<run_name>/
          (--resume continues that run from its last.pt)
  eval    score a checkpoint (or --random for a baseline)
  play    open the visual display
"""
import argparse

import yaml


def main():
    ap = argparse.ArgumentParser(prog="flybrain2048")
    ap.add_argument("command", choices=["fetch", "train", "eval", "play"])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--checkpoint", default=None, help="e.g. runs/ppo_dan_2x/best.pt")
    ap.add_argument("--refresh", action="store_true", help="rebuild the cached connectome")
    ap.add_argument("--refresh-shapes", action="store_true",
                    help="with --shapes: download every shape again instead of reusing cached ones")
    ap.add_argument("--shapes", action="store_true",
                    help="also download neuron skeletons, brain outline and synapse "
                         "locations (neuprint only)")
    ap.add_argument("--resume", action="store_true",
                    help="train: continue runs/<run_name>/ from last.pt")
    ap.add_argument("--random", action="store_true", help="eval a random-move baseline")
    ap.add_argument("--games", type=int, default=50)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.command == "fetch":
        from .connectome import load_connectome, load_morphology, load_synapses
        conn = load_connectome(cfg["connectome"], refresh=args.refresh)
        print(conn.summary())
        if args.shapes:
            load_morphology(cfg["connectome"], conn, refresh=args.refresh_shapes)
            load_synapses(cfg["connectome"], conn, refresh=args.refresh_shapes)
    elif args.command == "train":
        from .train import train
        train(cfg, resume=args.resume)
    elif args.command == "eval":
        from .train import evaluate
        evaluate(cfg, args.checkpoint, args.games, random_agent=args.random)
    elif args.command == "play":
        from .visualize import run
        run(cfg, args.checkpoint)


if __name__ == "__main__":
    main()
