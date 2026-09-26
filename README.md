# flybrain2048 🪰

A fruit fly brain, wired according to the real **male CNS connectome**
(Janelia FlyEM + Cambridge + MRC LMB + Google Research), learning to play 2048.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m flybrain2048 fetch                  # build the (synthetic) connectome
python -m flybrain2048 eval --random          # baseline: random moves
python -m flybrain2048 play                   # watch an UNTRAINED fly play
python -m flybrain2048 train                  # train (writes runs/first_flight/)
python -m flybrain2048 train --resume         # continue that run for another `updates`
python -m flybrain2048 eval --checkpoint runs/first_flight/best.pt
python -m flybrain2048 play --checkpoint runs/first_flight/best.pt
```

Display controls: `SPACE` pause · `N` step one move · `↑/↓` speed · `R` restart · `V` rotate brain view · `ESC` quit

## Switching to the real connectome

1. Log in at https://neuprint.janelia.org, open **Account → Auth Token**, copy it.
2. `export NEUPRINT_APPLICATION_CREDENTIALS="<your token>"`
3. In `config.yaml` set `connectome.source: neuprint` and confirm the dataset name
   (e.g. `male-cns:v1.0`) matches what neuPrint lists in its dataset dropdown.
4. `python -m flybrain2048 fetch --refresh --shapes`

The result is cached in `data/`, so you only hit the server once. `--shapes` also
downloads every neuron's traced skeleton plus the brain outline, so `play` draws the
real neurons where they sit in the fly (press `V` to rotate between front, top and
side views). Without it, neurons are drawn as dots at their cell bodies.

## How it works

```
board (16 cells) ──one-hot──▶ sensory neurons ──▶ interneurons ⟲ ──▶ descending neurons ──▶ 4 moves
   game.py                     [learned mapping]      [connectome wiring, fixed]           [grouped readout]
```

| File | What it does |
|---|---|
| `game.py` | Pure-numpy 2048 engine + board encoding |
| `connectome.py` | Pulls a 3-layer sub-circuit ending on descending neurons (DNs) from neuPrint, or builds a synthetic stand-in |
| `model.py` | Leaky rate-neuron RNN. **Who connects to whom** and **excitatory/inhibitory sign** come from the connectome and never change; only synapse *strengths*, neuron biases/time constants, sensory input, and readout are learned |
| `train.py` | REINFORCE policy gradient over batches of parallel games |
| `visualize.py` | Pygame window: board, neurons lighting up over the recurrent time steps, move probabilities |

**Why descending neurons?** DNs are the bottleneck carrying the brain's motor
commands down to the nerve cord, so they're a natural "choose a move" layer.
Right now DNs are split round-robin into 4 groups (up/right/down/left). A fun
experiment is assigning them by actual function instead (e.g. turning-related
DNs for left/right) in `connectome.py → _groups_round_robin`.

## Things to know

- A random agent averages roughly 1,000 points. Beating that consistently is your first milestone.
- REINFORCE is noisy. Expect slow, bumpy progress; watch `runs/<name>/metrics.csv`.
- Memory grows with game length × batch size. If you run out, lower `batch_games`.
- Weights are dense N×N. Fine up to a few thousand neurons; beyond that,
  switch `model.py` to a sparse edge list (`torch.sparse` or index/scatter ops).
- The inhibitory-neurotransmitter rule (GABA and glutamate → negative) is a simplification.

## Ideas for where to take it next

1. **Ablation test:** set `train_synapses: false`. How much does the raw wiring do on its own?
2. **Shuffle control:** randomly permute the connectome weights and retrain.
   Does real fly wiring learn faster than a random graph with the same statistics?
3. **Better RL:** add a value network (actor-critic / PPO) to reduce variance.
4. **Evolution strategies:** perturb synapse strengths, keep what scores better. Parallelizes well and suits "fixed wiring" nicely.
5. **Different circuits:** start from visual projection neurons (e.g. LC types) as inputs, since the fly "sees" the board.
6. **Silence neurons** during play in the visualizer and watch strategy break.

Data: the Male CNS dataset is licensed CC-BY; cite the FlyEM male CNS paper if you publish results.
