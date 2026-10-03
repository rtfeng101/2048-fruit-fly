"""Command-line entry point:  python -m flybrain2048 <command>

  fetch   build/cache the connectome sub-circuit (use --refresh to rebuild it,
          --shapes to also download real neuron shapes and synapse locations
          for the display; shapes already downloaded are reused unless
          --refresh-shapes)
  train   train with PPO, checkpoints go to runs/<run_name>/
          (--resume continues that run from its last.pt)
  eval    score a checkpoint (or --random for a baseline)
  play    open the visual display
  plot    save graphs of training to runs/<run_name>/plots/ (--eval to also
          score every checkpoint)
"""
import argparse

import yaml


def main():
    ap = argparse.ArgumentParser(prog="flybrain2048")
    ap.add_argument("command", choices=["fetch", "train", "eval", "play", "plot"])
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
    ap.add_argument("--runs", nargs="+", default=None,
                    help="plot: runs to draw together (default: train.run_name)")
    ap.add_argument("--window", type=int, default=25, help="plot: updates per rolling mean")
    ap.add_argument("--eval", action="store_true",
                    help="plot: also play --games games with every checkpoint of the first run")
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
    elif args.command == "plot":
        from .plots import make_plots
        make_plots(cfg, args.runs, args.window, args.eval, args.games)


if __name__ == "__main__":
    main()
