# Changes: visualization, training and circuit work

Changes made on top of commit `89335d4` ("cuda changes") during one working session with
Claude Code, 2026-09-28, in the order they were made, with the reason for each and the
measurements that drove the decisions. None of it is committed yet.

**Where things stand**

| | Before | Now |
|---|---|---|
| Training | REINFORCE, 8 games/update | PPO with a dopamine critic, 128 games/update |
| Motor readout | Fixed round-robin average of descending neurons (DNs) | Learned per-DN weights, starting from that average |
| Circuit | 2,320 neurons | 4,632 neurons, including 64 real dopamine neurons |
| Best eval score (50 games) | 4,218 (`gain10/best.pt`) | 5,893 (`ppo_dopamine/best.pt`); `ppo_dan_2x` not trained yet |
| Brain display | Flat dots, fixed lines from untrained weights | 4 views including rotating 3D (GPU), 3 colour modes, signal paths along real wiring |

---

## Quick reference

### Commands

```bash
# one-time: neuPrint token (neuprint.janelia.org -> Account -> Auth Token)
export NEUPRINT_APPLICATION_CREDENTIALS="<your token>"

# build the circuit in config.yaml (+ neuron shapes and synapse locations for the display)
python -m flybrain2048 fetch --refresh --shapes      # reuses shapes already downloaded
python -m flybrain2048 fetch --refresh --shapes --refresh-shapes   # re-download all shapes

# train / continue / score / watch
python -m flybrain2048 train                         # new run in runs/<train.run_name>/
python -m flybrain2048 train --resume                # another `updates` updates of that run
python -m flybrain2048 eval --checkpoint runs/ppo_dan_2x/best.pt
python -m flybrain2048 eval --random                 # random-move baseline (~1,000)
python -m flybrain2048 play --checkpoint runs/ppo_dan_2x/best.pt

# runs trained on the original 2,320-neuron circuit
python -m flybrain2048 eval --config config_2320.yaml --checkpoint runs/ppo_dopamine/best.pt
python -m flybrain2048 play --config config_2320.yaml --checkpoint runs/ppo_dopamine/best.pt
```

### Display controls (`play`)

| Key | Action |
|---|---|
| `SPACE` / `N` | pause / one move while paused |
| `↑` / `↓` | game speed, 0.25–80 moves/s |
| `R` / `ESC` | restart / quit |
| `V` | cycle views: front, top, side, rotating 3D |
| `M` | colour mode: firing rate, change from usual, why this move |
| `E` | signal paths on/off |

### Runs and which circuit they need

| Run | Trained with | Circuit | Config to use |
|---|---|---|---|
| `runs/gain10` | REINFORCE, grouped readout | 2,320 neurons | `config_2320.yaml` |
| `runs/ppo_dopamine` | PPO, whole-brain critic, learned readout | 2,320 neurons | `config_2320.yaml` |
| `runs/ppo_dan_2x` (next) | PPO, dopamine-neuron critic, 128 games/update | 4,632 neurons | `config.yaml` |

A checkpoint only loads with the exact neuron set it was trained on. The original circuit and
its shapes are backed up in `data/circuit_2320/` (`data/` is gitignored, so the backup is
local only).

### Eval scores so far (50 games, same seeds, always the most likely move)

| Checkpoint | Mean | Median |
|---|---|---|
| random moves | 1,002 | 1,030 |
| `gain10/update_001000.pt` | 2,885 | 2,612 |
| `gain10/best.pt` (update 1840) | 4,218 | 3,296 |
| `gain10/last.pt` (update 2000) | 5,340 | 5,666 |
| `ppo_dopamine/best.pt` | **5,893** | **5,924** |
| `ppo_dopamine/last.pt` | 5,783 | 5,794 |

`best.pt` is picked by the *training* batch's mean score (only 8 games per update for
`gain10`), which is noisy: `gain10/last.pt` beats its `best.pt` in eval. Always compare runs
with `eval`, not the training log.

---

## 1. Making the brain display readable

**Problem.** The display showed raw firing rates of ~2,300 neurons. Most fire about the same
on every board, so the panel looked the same from move to move. You couldn't see which move
the motor neurons favoured, how the decision built up over the 8 recurrent steps, or why the
fly picked a move. The connection lines were drawn from the *untrained* connectome weights,
so with a trained checkpoint they didn't show what the model had learned.

**Changes**

- **Race plot** (next to the move probabilities): the move probabilities *if the brain stopped
  at each time step*, drawn up to the current step.
  - *Why probabilities and not raw DN activity:* the first version plotted each move group's
    mean DN rate. It could show "down" ending highest while the fly picked "left", because
    the readout normalizes and rescales DN activity. Probabilities always end where the bars
    do.
- **Colour modes (`M`)**
  - *Firing rate:* the original view.
  - *Change from usual:* each neuron's rate minus its usual rate at the same step. "Usual" is
    measured at startup on 256 random-play boards and slowly adapts during play. Warm means
    more active than usual, cool means less. Neurons that act the same on every board fade
    out.
  - *Why this move:* rate × d log p(chosen move) / d rate. Warm pushed toward the chosen
    move, cool pushed away.
    - It's **added up over the steps so far**: at the last step only DNs can still affect the
      readout, so single-step attribution left every interneuron dark on the final frame.
    - Values are shown on a **square-root scale**, so DNs don't drown out everything else.
- **Signal lines:** each step shows the connections carrying the most signal right now
  (trained weight × presynaptic rate), green for excitatory and pink for inhibitory, instead
  of a fixed set of lines from the untrained weights.
- **Model refactor** (`model.py`): the time-step loop moved into `_dynamics()` and the readout
  into `_readout()`, shared by `forward()` and a new `explain()`. `explain()` returns
  per-step logits, rates and attributions for the display. Outputs were checked to match the
  old model exactly.
- **Descending neurons coloured by the move they vote for:** added, then **reverted** at your
  request. They're orange again. Move colours remain only on the probability bars and race
  plot, where the four lines need to be told apart.

## 2. README commands matched to the config

The Quick start used `runs/first_flight/` and called `fetch` "synthetic", but `config.yaml`
used `run_name: gain10` and `source: neuprint`. Paths and comments were fixed, plus notes
that `fetch` needs a token with `neuprint`, and that `train` refuses to overwrite a run
(use `--resume` or a new `run_name`).

## 3. Rotating 3D view

**Change.** A fourth view (`V`): the circuit slowly turns about its vertical axis (one turn
per 40 s), tilted slightly, with perspective, depth fading and a soft glow.
- All colour modes and signal paths work in it.
- Settings: `SPIN`, `TILT`, `CAMERA` at the top of `visualize.py`.

**Decisions**
- **Framing:** it rotates about, and is fitted to, the central 90% of the circuit. About 5–10%
  of neurons sit far down the nerve cord (the z extent reaches +100k voxels versus ±20k for
  the brain). Fitting everything shrank the brain to a small blob and swung it off centre.
  Those nerve-cord neurons leave the frame when it turns side-on.
- **Redraw rate:** the display redraws at least 30 times a second, while the game still
  advances at the speed chosen with ↑/↓. This keeps the rotation smooth at low game speeds.

## 4. Neuron-to-neuron signal paths along the real wiring

**Goal.** Show the "squiggly" route signal takes from one neuron to the next, lighting up as it
flows. More neurons wouldn't give this (they'd still be dots). The real traced shapes do.

**Data** (`connectome.py`)
- New `Synapses` cache (`data/synapses_neuprint.npz`), filled by `fetch --shapes`.
- It holds one representative synapse for each of the `path_connections` (3,000) strongest
  connections: the synapse nearest the middle of all of that pair's synapses, via neuPrint's
  `fetch_synapse_connections`.

**Routing** (`visualize.py → wiring_paths`)
- Each connection is traced from the sending neuron's soma along its skeleton to the synapse,
  then along the receiving neuron's skeleton to its soma. This uses shortest paths
  (`scipy` Dijkstra) over each neuron's skeleton graph.
- Skeletons are snapped to a grid, so touching segments share exact endpoints.
- Neurons without a skeleton get a straight half-line.
- Routing all 3,000 takes about 0.4 s at startup.

**Drawing**
- At each step, the 60 connections carrying the most signal (`PATHS_SHOWN`) glow green or
  pink.
- They fade in and out between steps (`GLOW_EASE`), so they don't flicker.
- Pulses run soma → branches → synapse → branches → soma in the direction signal flows,
  timed by the clock, so they keep moving while paused (`PULSE_SECONDS`).
- The neurons behind are dimmed (`PATH_BACKDROP`) so the paths stand out.
- Without synapse data, the lines fall back to straight lines.

**Testing.** Built before any real shapes existed, so it was tested on fake random skeletons,
then checked on your real data after you fetched it.

**3D speed-ups found along the way.** The 3D view took about 120 ms per frame with 10M-segment
skeletons.
- The branch point cloud is now capped. Random rounding lets the cap hold even with millions of
  tiny segments.
- Brightness math is only done on lit pixels.
- A slow `max` over a 3-wide axis was replaced.

## 5. The display was CPU-bound, and the 3D view moved to the GPU

**Finding.** Nothing in the display used the GPU. The game also advanced one simulation step per
drawn frame, so slow drawing slowed the game.

Measured with your real data (10M skeleton segments):

| Cost | Time |
|---|---|
| Flat views, per frame | 16–21 ms |
| Rotating 3D, per frame | 38–48 ms (~20 fps, so the game ran at ~2/3 speed) |
| Model per move (forward + "why" gradient) | 12–21 ms |

Four fixes were proposed; you chose #3, moving the 3D renderer to the GPU. #1 and #2 were done
later (§12). Still open:
4. Vectorize the pulse drawing.

**Change.** The 3D view's projection, point splatting, brightness compression and glow now run
in torch on CUDA (`DEVICE`). With no GPU it runs the same code on the CPU with the old
300k-point cap.

| Rotating 3D, real data | Before (CPU) | After (GPU) |
|---|---|---|
| Frame, paths on | ~48 ms | ~15 ms |
| Frame, paths off | ~38 ms | ~9 ms |
| Branch points drawn | 300k | 4M |
| GPU memory | – | ~160 MB held, ~450 MB peak |

- **Exposure:** the first GPU version overexposed the brain to white, because 13× more points
  add up 13× brighter. Each point is now dimmed so overall exposure matches the 300k-point
  look (`EXPOSURE_POINTS`). It's smoother, not brighter.

## 6. Training: PPO with a dopamine critic and a learned readout

**Starting point.** `gain10` (REINFORCE, 2,000 updates × 8 games) reached ~3,200 training
score, and the curve was still slowly rising.

The recommendations, in order of expected impact:
1. Actor-critic / PPO.
2. A learned motor readout.
3. Board-symmetry augmentation.
4. More games per update.
5. More thinking steps.
6. Weight sharing by cell type.
7. More neurons.

On the other ideas you raised:
- **Vision** would make the task *harder*: the fly would have to read tile values from an image.
- **Dopamine** helps only as the learning signal, which is what the critic provides.

**Changes**
- **PPO** (`train.py`)
  - Games are played without gradients, then there are 4 passes of 2,048-move minibatches over
    each batch.
  - Clipped policy loss, Huber value loss, advantages from GAE (λ 0.95).
  - Rewards are scaled by 0.1 so the critic's targets stay small.
  - Games per update went from 8 to 32.
- **Dopamine critic** (`model.py`): predicts reward still to come from the circuit's final
  activity. Its prediction errors (reward-prediction errors, what dopamine neurons are thought
  to signal) decide which moves get reinforced.
- **Learned readout** (`model.readout: learned`): learns how much each DN counts toward each
  move, starting exactly at the old round-robin average.
- **Board symmetries** (`game.py`): `SYM_ACTIONS` and `transform_board`. All 9,600 consistency
  checks pass.
- **Compatibility:** old checkpoints still `eval` and `play`. REINFORCE runs can't be resumed
  (they have no critic).

**The augmentation experiment.** With augmentation on, the first PPO run stalled at ~1,200 and
entropy stayed at ~0.93 (the fly stayed indecisive). Short parallel test runs showed:

| Test run (200 updates) | Games 3,201–6,400 | Entropy |
|---|---|---|
| augmentation on, lr 0.001 | ~1,230 | 0.91 |
| augmentation on, lr 0.003 | 1,211 | 0.92 |
| **augmentation off**, lr 0.001 | **2,624** | 0.66 |

- The learning rate wasn't the problem.
- Diagnostics also ruled out a broken critic (it explained 76% of return variance) and a
  crowded-out policy gradient (policy 0.63 vs critic 0.15).
- **Cause:** `gain10` had learned lopsided readout gains, right 2.49 : left 1.51 : up/down ~1.
  It wins mainly with a fixed favourite direction (push tiles into one corner). Random
  rotation makes any fixed direction useless, so the fly has to learn a much harder
  orientation-independent strategy.
- **Decision:** `augment: false` by default, kept as an option.

**Result.** Same 16,000 games as `gain10`:

| Games | `gain10` (REINFORCE) | `ppo_dopamine` (PPO) |
|---|---|---|
| 1–3,200 | 1,252 | 1,297 |
| 3,201–6,400 | 1,779 | 2,624 |
| 6,401–9,600 | 2,454 | 3,516 |
| 9,601–12,800 | 2,941 | 4,276 |
| 12,801–16,000 | 3,077 | 4,627 |

Eval: 5,893 vs 4,218 for the two `best.pt` files, and 62% vs 34% of games reached a 512 tile.

**Caveats**
- One seed per setup.
- PPO and the learned readout were tested together, so their separate contributions are
  unknown.
- The first 200 updates of `ppo_dopamine` were the no-augmentation test run, renamed and
  resumed. Its settings match the defaults.

## 7. The critic reads real dopamine neurons

**Change**
- **Choosing the neurons:** `fetch` now also picks dopamine-type neurons
  (`connectome.neuprint.dopamine.type_regex`, default PAM / PPL1 / PPL2 / PPM) and keeps the
  `n` (64) that receive the most synapses from the rest of the circuit. That way their
  activity depends on the board.
- **Wiring:** they're added with all their real connections, after the existing neurons, so
  existing indices, the DN readout and cached shapes stay valid. Ones already in the circuit
  are tagged rather than duplicated.
- **Critic:** `model.critic: dopamine` reads the prediction only from those neurons.
  `critic: brain` is the whole-circuit version used by `ppo_dopamine`.
- **Display:** dopamine neurons are magenta.
- **Shape reuse:** `fetch --refresh --shapes` now reuses already-downloaded skeletons, the brain
  outline and synapse locations by body id, and only fetches new neurons. `--refresh-shapes`
  forces a full re-download.
- **Backup:** `data/circuit_2320/` was made before refreshing, since a refresh makes older
  checkpoints unloadable.

**Why retraining is required.** The critic is only used during training, where it decides what
gets reinforced; in play only the policy picks moves. The new circuit also has a different
neuron set, which old checkpoints refuse. A warm start from `ppo_dopamine` (copy the shared
neurons' weights) was proposed but not built. It would start stronger, but muddies any
comparison.

**Testing.** No token was available in Claude's shell, so this was tested on:
- a synthetic circuit with 16 dopamine neurons (train, eval and play);
- a mocked neuPrint: selection, typing, indices, and reuse (4 of 72 skeletons fetched, 56 of
  60 synapse locations reused).

Your real fetch then produced 4,632 neurons with 64 dopamine neurons.

**Simplification.** Dopamine is treated as an ordinary excitatory transmitter. In the fly it
mostly modulates other synapses.

## 8. Doubling the circuit

**Change**
- `n_in` 256 → 512 and `n_hidden` 2,000 → 4,000.
- DNs (64) and dopamine neurons (64) are unchanged, so the move readout and the critic stay
  the same and only the circuit's size changes.
- The result is 4,632 neurons and 585,890 connections.

**You chose to skip a same-critic baseline** (`ppo_dan` at the old size). Note that a bigger
circuit isn't guaranteed to be better: the added hidden neurons are the ones more weakly
connected to the DNs, and bigger networks sometimes learn more slowly in RL. Judge it by
`eval`.

**Measured cost**

| | 2,384 neurons | 4,640 neurons |
|---|---|---|
| GPU memory (training peak) | 0.8 GB | 1.7 GB |
| Time per update (32 games, early) | ~1.4 s | ~3.8 s |

**Memory fix for `play`.** This WSL environment has ~7.6 GB of RAM, and the flat views already
peaked at ~3 GB building from 10M segments. Twice the skeletons would likely have run out.
They're now built in 1M-segment chunks, with duplicate pixels removed as they go (`SEG_CHUNK`).
Output is identical (checked view by view), and peak memory dropped from 3.0 GB to 1.4 GB.
Full `play` visiting every view now peaks at ~2.5 GB on the old circuit; expect ~5 GB on the
new one.

## 9. More games per update

`train.batch_games` 32 → 128. Measured on the doubled circuit:

| Games / update | Time / update (early) | Time / game | GPU memory |
|---|---|---|---|
| 32 | ~5.5 s | 0.17 s | 1.7 GB |
| 64 | ~7.3 s | 0.11 s | 1.7 GB |
| 128 | ~12.7 s | 0.10 s | 1.7 GB |

- Bigger batches use the GPU better, so ~1.7× more games per hour.
- GPU memory is capped by `minibatch`, not by games per update.
- Learning steps per game stay about the same.
- With `updates: 500`, `ppo_dan_2x` trains on 64,000 games (4× `ppo_dopamine`). That's several
  hours; set `updates: 125` for 16,000 games, or stop with Ctrl+C and `--resume` later.

**Naming a run:** `train.run_name` in the config. Output goes to `runs/<run_name>/`, and there's
no CLI flag for it.

## 10. Evaluating old runs

`config_2320.yaml` is a copy of `config.yaml` pointing at the backed-up 2,320-neuron circuit,
with that circuit's original sizes (and `dopamine.n: 0`), so a `fetch` with it can't touch the
new circuit. Every old checkpoint was re-scored through it (table at the top).

## 11. Flat views washed out on the doubled circuit

**Finding.** Headless renders of both circuits showed the new one's flat views (front, top,
side) pale white across the brain, with the signal paths lost in it. Two causes:
- Twice the neurons overlap each pixel (median 26 vs 13 in the front view). Even idle
  neurons glow at 4%, so a typical pixel started near white.
- With signal paths on, the neurons behind them were dimmed (`PATH_BACKDROP`) *before* the
  brightness compression, which undid most of the dimming in dense regions.

**Change** (`visualize.py`)
- Flat views dim each neuron by the view's density: `EXPOSURE_OVERLAP = 13` neurons per lit
  pixel, tuned to the old circuit, the flat-view counterpart of `EXPOSURE_POINTS`.
- `PATH_BACKDROP` is applied after compression, in the flat views and the 3D view.

| New circuit, front view, paths on | Before | After |
|---|---|---|
| Near-white pixels (> 200) | 18% | 2.4% |

The rotating 3D view was already exposure-matched and looked the same on both circuits.

## 12. The game on its own clock, and the model on the GPU

**Finding.** The game advanced one tick per drawn frame, and a move is 10 ticks (8 steps plus
a short linger), so ↑/↓ couldn't go faster than drawing. Profiled on the doubled circuit:

| Per move | Before | After |
|---|---|---|
| Model (`explain`), CPU, recomputing the weights each move | 75 ms | – |
| Model, CPU, weights cached | 40 ms | – |
| Model, GPU, weights cached | – | 5.5 ms |

The weights are an 86 MB matrix, read 16 times per move (8 steps forward, 8 back for "why").

**Change**
- **Own clock** (`visualize.py → run`): the game owes speed × ticks-per-move ticks per second of
  real time and catches up between frames, up to `SIM_BUDGET` (25 ms) per frame. Beyond
  that the backlog is dropped, so the display never freezes. Redraws are capped at
  `RENDER_FPS` (30).
- **Speeds are moves per second** (`SPEEDS`: 0.25 to 80, default 3). When the model can't keep
  up, the status line says what it's managing, e.g. `80 moves/s (managing 55)`.
- **Model on the GPU** (`DEVICE`, formerly `DEVICE_3D`), with the weights computed once at
  startup and passed to `model.explain(obs, valid, W)`. The results match the CPU to ~1e-6.

**Measured headless** (moves/s actually made while the game ran):

| Speed | Front view | Rotating 3D |
|---|---|---|
| 3 | 3.0 (29 fps) | – |
| 10 | 10.0 (26 fps) | – |
| 20 | 20.1 (23 fps) | 20.0 (30 fps) |
| 80 | ~45 (16 fps) | ~56 (23 fps) |

---

## Config reference (new keys)

| Key | Default | What it does |
|---|---|---|
| `connectome.synapse_path` | `data/synapses_{source}.npz` | synapse-location cache for signal paths |
| `connectome.synthetic.n_dopamine` | 16 | stand-in dopamine neurons in the synthetic circuit |
| `connectome.neuprint.dopamine.n` | 64 | dopamine neurons to add (0 = none) |
| `connectome.neuprint.dopamine.type_regex` | `(PAM\|PPL1\|PPL2\|PPM).*` | which cell types count as dopaminergic |
| `connectome.neuprint.path_connections` | 3000 | connections to fetch a synapse location for |
| `model.readout` | `learned` | `grouped` = fixed round-robin average of DNs |
| `model.critic` | `dopamine` | `brain` = read from all neurons; `false` = none (can't train) |
| `train.batch_games` | 128 | games per update |
| `train.gae_lambda` | 0.95 | how far prediction errors are credited back |
| `train.reward_scale` | 0.1 | keeps the critic's targets small |
| `train.ppo_epochs` / `minibatch` / `clip` | 4 / 2048 / 0.2 | PPO passes, moves per step, step limit |
| `train.value_coef` | 0.5 | critic loss weight |
| `train.augment` | false | random board rotation/mirroring per move (hurt here; see §6) |

Display constants at the top of `visualize.py`:
- 3D view: `SPIN`, `TILT`, `CAMERA`, `MAX_3D_POINTS`, `EXPOSURE_POINTS`.
- Signal paths: `PATHS_SHOWN`, `PATH_BACKDROP`, `PULSE_SECONDS`, `GLOW_EASE`.
- Flat views: `SEG_CHUNK`.

## Files touched

| File | Changes |
|---|---|
| `flybrain2048/visualize.py` | colour modes, race plot, 3D view (GPU), signal paths + routing, flow-based edges, chunked flat views, dopamine colour, density exposure, game on its own clock, model on the GPU |
| `flybrain2048/model.py` | `_dynamics`/`_readout`/`explain`/`policy_value`, learned readout, dopamine critic, tensor masks, `explain(..., W)` |
| `flybrain2048/train.py` | PPO, GAE, rollouts without gradients, symmetry option, critic checks |
| `flybrain2048/connectome.py` | dopamine neurons (neuPrint + synthetic), `Synapses` cache, shape/synapse reuse |
| `flybrain2048/game.py` | board symmetries |
| `flybrain2048/__main__.py` | `--refresh-shapes`, synapse fetch, help text |
| `config.yaml` | all keys above; doubled sizes; `run_name: ppo_dan_2x` |
| `config_2320.yaml` | new: the original circuit, for old runs |
| `README.md` | commands, controls, critic/readout/symmetry notes, memory notes, ideas |

## Known issues and open ideas

- **CPU-only machines:** GPU-trained checkpoints fail to load there. `load_model` in
  `train.py` needs `map_location` in `torch.load` (a one-line fix, not made).
- **Remaining display fix (§5):** vectorized pulses. At 80 moves/s drawing plus the model
  is the limit (~45–56 moves/s).
- **Warm start:** start a new-circuit run from a trained old-circuit run's shared weights.
- **Single seeds:** all comparisons use one seed per setup; use 2–3 before trusting small
  differences.
- **Model ideas not yet tried:** more thinking steps (`model.steps` 12–16), weight sharing by
  cell type, dopamine as synapse modulation (mushroom body), assigning DNs to moves by
  function instead of round-robin.
- **Nothing is committed.**
