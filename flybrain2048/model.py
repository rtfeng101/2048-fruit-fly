"""A rate-based recurrent network whose wiring IS the connectome.

Key idea (connectome-constrained modeling):
  - WHO connects to WHOM is fixed by the connectome (no new synapses).
  - The SIGN of each synapse is fixed by the presynaptic neurotransmitter.
  - What we learn: per-synapse strength multipliers, per-neuron bias/time
    constants, the 'sensory transduction' from board -> input neurons,
    and a small readout from descending neurons -> 4 moves.
  - Optionally a "dopamine" critic: a readout that predicts how much reward is still
    to come, from the circuit's real dopamine neurons (critic="dopamine") or from all
    of its neurons (critic="brain"). In the fly, dopamine neurons are thought to
    signal errors in exactly that kind of prediction; training (PPO) uses them to
    judge whether a move turned out better or worse than expected.

Dynamics (Euler-integrated leaky rate neurons):
  h <- h + (dt / tau) * (-h + W_eff @ r + I_ext + b)
  r  = relu(tanh(h))          # firing rate in [0, 1)
"""
import numpy as np
import torch
import torch.nn as nn


class FlyBrainNet(nn.Module):
    def __init__(self, conn, obs_dim=256, steps=8, dt=0.5, train_synapses=True, w_scale=1.0,
                 normalize_readout=False, readout="grouped", critic=False):
        """readout: "grouped" averages the DNs voting for each move (round robin);
        "learned" learns how much each DN counts toward each move, starting from that
        same averaging. critic: the critic PPO training needs, read from the dopamine
        neurons ("dopamine") or the whole circuit ("brain", or True); False = none."""
        super().__init__()
        self.steps, self.dt = steps, dt
        self.normalize_readout = normalize_readout
        n = conn.n

        W = torch.tensor(conn.weights, dtype=torch.float32)
        # Normalize so the average neuron's total |input| is ~w_scale (keeps dynamics stable).
        in_sum = W.abs().sum(1)
        W = W / (in_sum[in_sum > 0].mean() + 1e-8) * w_scale
        self.register_buffer("W_struct", W)
        self.register_buffer("input_idx", torch.tensor(conn.input_idx, dtype=torch.long))
        self.register_buffer("output_idx", torch.tensor(conn.output_idx, dtype=torch.long))

        # Group matrix: averages output neurons into 4 action channels.
        groups = torch.tensor(conn.output_groups, dtype=torch.long)
        G = torch.zeros(4, len(groups))
        G[groups, torch.arange(len(groups))] = 1.0
        self.register_buffer("G", G / G.sum(1, keepdim=True).clamp(min=1))
        self.readout_w = nn.Parameter(self.G.clone()) if readout == "learned" else None

        # Learnable parts. Log-multipliers start at 0 => the raw connectome.
        # Dense for simplicity; switch to sparse edge lists for >~5k neurons.
        self.log_syn = nn.Parameter(torch.zeros_like(W)) if train_synapses else None
        self.bias = nn.Parameter(torch.zeros(n))
        self.log_tau = nn.Parameter(torch.zeros(n))
        self.sensory = nn.Linear(obs_dim, len(conn.input_idx))
        # Normalized readout is already unit scale; the raw one needs a boost.
        self.readout_gain = nn.Parameter(torch.ones(4) * (1.0 if normalize_readout else 5.0))
        self.readout_bias = nn.Parameter(torch.zeros(4))
        if critic == "dopamine":
            if not len(conn.dopamine_idx):
                raise ValueError("model.critic: dopamine, but this connectome has no dopamine "
                                 "neurons. Set connectome.neuprint.dopamine.n and run "
                                 "`fetch --refresh`.")
            critic_idx = torch.tensor(conn.dopamine_idx, dtype=torch.long)
        else:
            critic_idx = torch.arange(n)
        # Not saved in checkpoints: rebuilt from the connectome (older ones lack it).
        self.register_buffer("critic_idx", critic_idx, persistent=False)
        self.dopamine = nn.Linear(len(critic_idx), 1) if critic else None
        if critic:  # start by predicting the same for every board
            nn.init.zeros_(self.dopamine.weight)

    def effective_weights(self):
        if self.log_syn is None:
            return self.W_struct
        # exp() keeps multipliers positive, so synapse signs never flip.
        return self.W_struct * torch.exp(self.log_syn.clamp(-5, 5))

    def _dynamics(self, obs, W):
        """Yields the firing rates r (B, N) after each recurrent time step."""
        B, N = obs.shape[0], self.W_struct.shape[0]
        I = torch.zeros(B, N, device=obs.device)
        I[:, self.input_idx] = torch.relu(self.sensory(obs))
        tau = torch.exp(self.log_tau).clamp(0.2, 10.0)

        h = torch.zeros(B, N, device=obs.device)
        r = torch.zeros_like(h)
        for _ in range(self.steps):
            h = h + (self.dt / tau) * (-h + r @ W.T + I + self.bias)
            r = torch.relu(torch.tanh(h))
            yield r

    def _readout(self, r):
        G = self.G if self.readout_w is None else self.readout_w
        motor = r[:, self.output_idx] @ G.T          # (B, 4)
        if self.normalize_readout:
            # DN rates are small and differ only slightly between the 4 moves; rescale
            # so those differences, not the overall activity level, decide the move.
            motor = motor - motor.mean(1, keepdim=True)
            motor = motor / (motor.std(1, keepdim=True) + 1e-4)
        return motor * self.readout_gain + self.readout_bias

    def forward(self, obs, return_trace=False, W=None):
        """obs: (B, obs_dim). Returns logits (B, 4) [and activity trace (steps, B, N)].

        Pass a precomputed W (from effective_weights()) when calling many times
        per update: it saves a lot of memory in the autograd graph.
        """
        W = self.effective_weights() if W is None else W
        trace = []
        for r in self._dynamics(obs, W):
            if return_trace:
                trace.append(r.detach())
        logits = self._readout(r)
        if return_trace:
            return logits, torch.stack(trace)
        return logits

    def policy_value(self, obs, W=None):
        """logits (B, 4) and the dopamine critic's predicted future reward (B,)."""
        W = self.effective_weights() if W is None else W
        for r in self._dynamics(obs, W):
            pass
        return self._readout(r), self.dopamine(r[:, self.critic_idx])[:, 0]

    def explain(self, obs, valid_mask, W=None):
        """For the display: one board (1, obs_dim) -> logits (steps, 4), i.e. the move
        scores if the brain stopped at each step, rates (steps, N) and attribution (steps, N).

        Attribution is rate x d log p(chosen move) / d rate at each step: positive
        where a neuron's activity pushed the brain toward the move it picked.
        Pass a precomputed W (from effective_weights()) when explaining move after move.
        """
        W = self.effective_weights() if W is None else W
        with torch.enable_grad():
            rates = list(self._dynamics(obs, W))
            logits = masked_logits(self._readout(torch.cat(rates)), valid_mask)
            logp = torch.log_softmax(logits[-1], -1)[logits[-1].argmax()]
            grads = torch.autograd.grad(logp, rates)
        rates = torch.cat(rates).detach()
        return logits.detach(), rates, rates * torch.cat(grads)


def masked_logits(logits, valid_mask):
    """Stop the fly from choosing moves that don't change the board."""
    if not torch.is_tensor(valid_mask):
        valid_mask = np.asarray(valid_mask)
    mask = torch.as_tensor(valid_mask, dtype=torch.bool, device=logits.device)
    return logits.masked_fill(~mask, -1e9)
