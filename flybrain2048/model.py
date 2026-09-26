"""A rate-based recurrent network whose wiring IS the connectome.

Key idea (connectome-constrained modeling):
  - WHO connects to WHOM is fixed by the connectome (no new synapses).
  - The SIGN of each synapse is fixed by the presynaptic neurotransmitter.
  - What we learn: per-synapse strength multipliers, per-neuron bias/time
    constants, the 'sensory transduction' from board -> input neurons,
    and a small readout from descending neurons -> 4 moves.

Dynamics (Euler-integrated leaky rate neurons):
  h <- h + (dt / tau) * (-h + W_eff @ r + I_ext + b)
  r  = relu(tanh(h))          # firing rate in [0, 1)
"""
import numpy as np
import torch
import torch.nn as nn


class FlyBrainNet(nn.Module):
    def __init__(self, conn, obs_dim=256, steps=8, dt=0.5, train_synapses=True, w_scale=1.0,
                 normalize_readout=False):
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

        # Learnable parts. Log-multipliers start at 0 => the raw connectome.
        # Dense for simplicity; switch to sparse edge lists for >~5k neurons.
        self.log_syn = nn.Parameter(torch.zeros_like(W)) if train_synapses else None
        self.bias = nn.Parameter(torch.zeros(n))
        self.log_tau = nn.Parameter(torch.zeros(n))
        self.sensory = nn.Linear(obs_dim, len(conn.input_idx))
        # Normalized readout is already unit scale; the raw one needs a boost.
        self.readout_gain = nn.Parameter(torch.ones(4) * (1.0 if normalize_readout else 5.0))
        self.readout_bias = nn.Parameter(torch.zeros(4))

    def effective_weights(self):
        if self.log_syn is None:
            return self.W_struct
        # exp() keeps multipliers positive, so synapse signs never flip.
        return self.W_struct * torch.exp(self.log_syn.clamp(-5, 5))

    def forward(self, obs, return_trace=False, W=None):
        """obs: (B, obs_dim). Returns logits (B, 4) [and activity trace (steps, B, N)].

        Pass a precomputed W (from effective_weights()) when calling many times
        per update: it saves a lot of memory in the autograd graph.
        """
        B, N = obs.shape[0], self.W_struct.shape[0]
        W = self.effective_weights() if W is None else W
        I = torch.zeros(B, N, device=obs.device)
        I[:, self.input_idx] = torch.relu(self.sensory(obs))
        tau = torch.exp(self.log_tau).clamp(0.2, 10.0)

        h = torch.zeros(B, N, device=obs.device)
        r = torch.zeros_like(h)
        trace = []
        for _ in range(self.steps):
            h = h + (self.dt / tau) * (-h + r @ W.T + I + self.bias)
            r = torch.relu(torch.tanh(h))
            if return_trace:
                trace.append(r.detach())

        motor = r[:, self.output_idx] @ self.G.T          # (B, 4)
        if self.normalize_readout:
            # DN rates are small and differ only slightly between the 4 moves; rescale
            # so those differences, not the overall activity level, decide the move.
            motor = motor - motor.mean(1, keepdim=True)
            motor = motor / (motor.std(1, keepdim=True) + 1e-4)
        logits = motor * self.readout_gain + self.readout_bias
        if return_trace:
            return logits, torch.stack(trace)
        return logits


def masked_logits(logits, valid_mask):
    """Stop the fly from choosing moves that don't change the board."""
    mask = torch.as_tensor(np.asarray(valid_mask), dtype=torch.bool, device=logits.device)
    return logits.masked_fill(~mask, -1e9)
