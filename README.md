# flybrain2048 🪰

A fruit fly brain, wired according to the real **male CNS connectome**
(Janelia FlyEM + Cambridge + MRC LMB + Google Research), learning to play 2048.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export NEUPRINT_APPLICATION_CREDENTIALS="<your token>"   # see "Getting the connectome" below

python -m flybrain2048 fetch --shapes         # download the circuit in config.yaml + neuron shapes
python -m flybrain2048 eval --random          # baseline: random moves (~1,000 points)
python -m flybrain2048 play                   # watch an UNTRAINED fly play
python -m flybrain2048 train                  # train a new run in runs/ppo_dan_2x/
python -m flybrain2048 train --resume         # continue that run for another `updates`
python -m flybrain2048 eval --checkpoint runs/ppo_dan_2x/best.pt
python -m flybrain2048 play --checkpoint runs/ppo_dan_2x/best.pt
```

Every command reads `config.yaml` unless you pass `--config <file>`.

## Commands

| Command | What it does |
|---|---|
| `fetch` | Build the circuit described in `config.yaml` and cache it in `data/`. Uses the cache if one exists. |
| `fetch --refresh` | Rebuild the circuit, e.g. after changing its size or dopamine settings. |
| `fetch --shapes` | Also download neuron skeletons, the brain outline and synapse locations for `play`. Shapes already downloaded are reused. |
| `fetch --refresh --shapes --refresh-shapes` | Rebuild everything, re-downloading every shape. |
| `train` | Train a new run in `runs/<train.run_name>/`. Refuses if that run already exists. |
| `train --resume` | Continue `runs/<train.run_name>/` from its `last.pt` for another `train.updates` updates. |
| `eval --checkpoint <file.pt>` | Score a checkpoint over `--games` games (default 50), always taking the most likely move. |
| `eval --random` | Score random moves, as a baseline. |
| `play [--checkpoint <file.pt>]` | Open the display. Without a checkpoint, the fly is untrained. |

**Runs and configs.** A run's name is `train.run_name` in the config, and it's saved to
`runs/<run_name>/` (`best.pt`, `last.pt`, `metrics.csv`). `best.pt` is chosen by the
training batch's score, which is noisy, so compare runs with `eval`. A checkpoint only
loads with the exact circuit it was trained on:

| Config | Circuit | Runs |
|---|---|---|
| `config.yaml` (current) | 4,632 neurons: 512 inputs, 4,000 hidden, 64 descending, 64 dopamine | `ppo_dan_2x` |
| `config_2320.yaml` | The original 2,320 neurons (256 inputs, 2,000 hidden, 64 descending), backed up in `data/circuit_2320/` | `gain10`, `ppo_dopamine` |

```bash
python -m flybrain2048 eval --config config_2320.yaml --checkpoint runs/ppo_dopamine/best.pt
python -m flybrain2048 play --config config_2320.yaml --checkpoint runs/ppo_dopamine/best.pt
```

Runs trained before the switch to PPO (e.g. `gain10`) still `eval` and `play`, but can't
be resumed: they have no critic.

**Display controls:** `SPACE` pause · `N` step one move · `↑/↓` speed · `R` restart ·
`V` cycle brain views (front / top / side / rotating 3D) · `M` colour mode (firing rate /
change from usual / why this move) · `E` signal paths · `ESC` quit

## Getting the connectome

1. Log in at https://neuprint.janelia.org, open **Account → Auth Token**, copy it.
2. `export NEUPRINT_APPLICATION_CREDENTIALS="<your token>"`
3. Check that `connectome.neuprint.dataset` in `config.yaml` (e.g. `male-cns:v1.0`)
   matches what neuPrint lists in its dataset dropdown.
4. `python -m flybrain2048 fetch --shapes`

To work offline without a token, set `connectome.source: synthetic` in `config.yaml`: a
random stand-in circuit with fly-like statistics. It has no neuron shapes, so run
`fetch` without `--shapes`.

Everything is cached in `data/`, so you only hit the server once. `--shapes` downloads
every neuron's traced skeleton plus the brain outline, so `play` draws the real neurons
where they sit in the fly; without it, neurons are drawn as dots at their cell bodies.
It also fetches one synapse location for each of the `path_connections` strongest
connections (default 3000). `play` then routes the connections carrying the most signal
along the real wiring: from the sending neuron's cell body along its branches to the
synapse, then along the receiving neuron's branches to its cell body. Those paths glow
while they carry signal, with pulses running in the direction the signal travels.
Without synapse locations they're straight lines.

## How it works

```
board (16 cells) ──one-hot──▶ sensory neurons ──▶ interneurons ⟲ ──▶ descending neurons ──▶ 4 moves
   game.py                     [learned mapping]      [connectome wiring, fixed]           [learned readout]
                                                              │
                                                              └──▶ dopamine neurons ──▶ critic: reward still to come
```

| File | What it does |
|---|---|
| `game.py` | Pure-numpy 2048 engine + board encoding |
| `connectome.py` | Pulls a 3-layer sub-circuit ending on descending neurons (DNs) from neuPrint, or builds a synthetic stand-in |
| `model.py` | Leaky rate-neuron RNN. **Who connects to whom** and **excitatory/inhibitory sign** come from the connectome and never change; only synapse *strengths*, neuron biases/time constants, sensory input, and readout are learned |
| `train.py` | PPO (actor-critic) over batches of parallel games |
| `visualize.py` | Pygame window: board, neurons lighting up over the recurrent time steps, signal paths along the real wiring, move probabilities |

**Why descending neurons?** DNs are the bottleneck carrying the brain's motor
commands down to the nerve cord, so they're a natural "choose a move" layer.
DNs start out split round-robin into 4 groups (up/right/down/left); with
`model.readout: learned` training then learns how much each DN counts toward each
move. A fun experiment is assigning them by actual function instead (e.g.
turning-related DNs for left/right) in `connectome.py → _groups_round_robin`.

**The dopamine critic.** In the fly, dopamine neurons are thought to signal
reward-prediction errors: "that went better (or worse) than expected". The critic
predicts how much reward is still to come in the game, and its prediction errors
decide which moves training reinforces (PPO with generalized advantage estimation).
With `model.critic: dopamine` it reads that prediction only from real dopamine
neurons: the `connectome.neuprint.dopamine.n` dopamine-type neurons (PAM, PPL1,
PPL2, PPM) that receive the most synapses from the rest of the circuit, wired in
with all their real connections. `critic: brain` reads it from every neuron instead.
The model treats dopamine like any other excitatory transmitter; in the fly it
mostly modulates other synapses rather than exciting its targets directly.

**Symmetry (off by default).** 2048 plays the same rotated or mirrored, and
`train.augment: true` has the fly choose each move on a random one of the board's 8
equivalent versions. It sounds like free data, but it made learning much slower
here: the fly's easiest winning habit is a fixed favourite direction (push tiles
toward one corner), and random rotation makes any fixed direction useless.

## Things to know

- A random agent averages roughly 1,000 points. Beating that consistently is your first milestone.
- RL is still noisy. Expect bumpy progress; watch `runs/<name>/metrics.csv`.
- GPU memory is set by `train.minibatch` (moves per gradient step). If you run out, lower it.
- Weights are dense N×N, so memory and compute grow with N². ~4,600 neurons trains in
  about 1.7 GB of GPU memory at ~2.7x the time per update of ~2,300; much beyond
  that, switch `model.py` to a sparse edge list (`torch.sparse` or index/scatter ops).
- The inhibitory-neurotransmitter rule (GABA and glutamate → negative) is a simplification.

## Ideas for where to take it next

1. **Ablation test:** set `train_synapses: false`. How much does the raw wiring do on its own?
2. **Shuffle control:** randomly permute the connectome weights and retrain.
   Does real fly wiring learn faster than a random graph with the same statistics?
3. **Dopamine as modulation:** let dopamine neurons scale the strength of the synapses
   they sit near (as in the mushroom body) instead of acting as ordinary excitatory inputs.
4. **Evolution strategies:** perturb synapse strengths, keep what scores better. Parallelizes well and suits "fixed wiring" nicely.
5. **Different circuits:** start from visual projection neurons (e.g. LC types) as inputs, since the fly "sees" the board.
6. **Silence neurons** during play in the visualizer and watch strategy break.

Data: the Male CNS dataset is licensed CC-BY; cite the FlyEM male CNS paper if you publish results.
