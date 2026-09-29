"""Watch the fly brain play 2048.

Left:   the game board.
Right:  every neuron in the sub-circuit at its real place in the fly's nervous
        system (its traced branches, if `fetch --shapes` was run), lighting up as
        activity propagates from sensory neurons (blue) through interneurons
        (grey) to descending/motor neurons (orange), plus dopamine neurons
        (magenta) if the circuit has them. The connections carrying the most signal
        right now glow, with pulses running from the sending neuron's cell body
        along its branches to the synapse and on along the receiving neuron's
        branches (once `fetch --shapes` has fetched synapse locations;
        straight lines otherwise). V cycles front, top and side views and a slowly
        rotating 3D view.
        M switches what the colours mean: firing rate, change from the usual
        firing for that time step, or why (which neurons pushed toward the move).
Bottom: the probability the brain assigns to each move, and how those
        probabilities build up over the recurrent time steps.

Controls: SPACE pause | N single move while paused | UP/DOWN speed | R restart |
          V cycle brain views | M colour mode | E signal paths | ESC quit
"""
import time

import numpy as np
import pygame
import torch
import torch.nn.functional as F
from scipy import sparse
from scipy.sparse.csgraph import dijkstra

from .connectome import load_connectome, load_morphology, load_synapses
from .game import ACTIONS, Game2048, encode_board
from .train import build_model, load_model

W_WIN, H_WIN = 1280, 760
BOARD_PX, PAD = 460, 40
BG = (250, 248, 239)
TILE_COLORS = {0: (205, 193, 180), 2: (238, 228, 218), 4: (237, 224, 200), 8: (242, 177, 121),
               16: (245, 149, 99), 32: (246, 124, 95), 64: (246, 94, 59), 128: (237, 207, 114),
               256: (237, 204, 97), 512: (237, 200, 80), 1024: (237, 197, 63), 2048: (237, 194, 46)}
ROLE_COLORS = {"in": np.array([70, 140, 255]), "hid": np.array([170, 170, 185]),
               "out": np.array([255, 140, 40]), "da": np.array([235, 80, 200])}
# One colour per move (up, right, down, left): probability bars and race lines.
MOVE_COLORS = np.array([[235, 90, 90], [80, 200, 120], [245, 170, 40], [180, 110, 245]])
# Signed colour modes: warm = more / toward the chosen move, cool = less / away from it.
POS_COLOR, NEG_COLOR = np.array([255, 120, 70]), np.array([80, 170, 255])
MODES = ["activity", "change", "why"]
MODE_TITLES = {"activity": "firing rate", "change": "change from usual", "why": "why this move"}
# (name, data axis shown left->right, data axis shown top->bottom); x/y/z are neuPrint voxels.
# The 3D view has no fixed axes: it turns about the brain's vertical axis.
VIEWS = [("front", 0, 1), ("top", 0, 2), ("side", 2, 1), ("rotating 3D", None, None)]
SPIN = 2 * np.pi / 40      # 3D view: radians per second (one turn every 40 s)
TILT = 0.35                # 3D view: look down on the brain slightly (radians)
CAMERA = 4.0               # 3D view: camera distance in brain radii (smaller = more perspective)
# The 3D view is drawn with torch, on the GPU when there is one: there it can afford
# many more branch sample points per frame, so branches look solid instead of dotted.
# The model runs there too (~14x faster per move), so high game speeds are reachable.
DEVICE =torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_3D_POINTS = 4_000_000 if DEVICE.type == "cuda" else 300_000
SEG_CHUNK = 1_000_000      # flat views: branch segments sampled at a time (caps memory use)
MAX_OUTLINE_POINTS = 200_000 if DEVICE.type == "cuda" else 40_000
EXPOSURE_POINTS = 300_000  # 3D view: branch brightness is tuned for this many points; more
                           # points are each dimmed to match, so they look smoother, not brighter
EXPOSURE_OVERLAP = 13      # flat views: brightness is tuned for this many neurons on a typical
                           # lit pixel (the 2,320-neuron circuit); denser circuits are dimmed to
                           # match, so more neurons don't wash the brain out to white
OUTLINE_FILL, OUTLINE_EDGE = np.array([28, 30, 46]), np.array([72, 78, 112])
EXC_COLOR, INH_COLOR = np.array([90, 200, 120]), np.array([220, 80, 110])
PULSE_SECONDS = 1.5        # signal paths: time for a pulse to run from one soma to the next
GLOW_EASE = 0.2            # signal paths: how fast they fade in/out (per frame, 0..1)
PATH_BACKDROP = 0.35       # signal paths: dim the neurons behind them to this brightness
PATHS_SHOWN = 60           # signal paths: how many of the strongest to draw at each step
RENDER_FPS = 30            # redraw rate cap; the game keeps its own clock, whatever the redraw rate
SPEEDS = (0.25, 0.5, 1, 2, 3, 5, 10, 20, 40, 80)  # game speeds for UP/DOWN, in moves per second
SIM_BUDGET = 0.025         # seconds of each frame the game may spend catching up; past that it
                           # runs as fast as the model allows instead of freezing the display


class BrainView:
    """Neurons drawn where they really are in the fly's nervous system.

    With neuron shapes (`fetch --shapes`) each neuron's traced branches are drawn
    over an outline of the brain; without them each neuron is a dot at its soma.
    Each flat view is precomputed as a sparse (pixels x neurons) matrix, so lighting
    up every branch each frame is a single matrix multiply. The 3D view re-projects a
    point cloud of the neurons every frame (on the GPU if there is one), dimmer with
    depth, with a soft glow.
    """

    def __init__(self, conn, rect, weights, usual, morph=None, paths=None, n_edges=150):
        """weights: the model's effective (post, pre) synapse weights.
        usual: (steps, N) typical firing at each time step, the baseline for "change".
        paths: optional (pre (C,), post (C,), one (P, 3) polyline per connection) routing
        connections along real wiring (see `wiring_paths`); only these get drawn."""
        self.rect = pygame.Rect(rect)
        self.conn, self.morph = conn, morph
        self.view, self._layouts, self._cloud = 0, {}, None
        self.mode, self.show_edges, self.n_edges = 0, True, n_edges
        self.surface = pygame.Surface(self.rect.size)
        self.font = pygame.font.SysFont(None, 22)

        self.role = np.full(conn.n, "hid", dtype=object)
        self.role[conn.input_idx] = "in"
        self.role[conn.output_idx] = "out"
        self.role[conn.dopamine_idx] = "da"
        self.base = np.stack([ROLE_COLORS[r] for r in self.role]).astype(np.float32)
        self.radius = np.select([self.role == "out", self.role == "da", self.role == "in"], [5, 4, 3], 2)
        self.has_pos = np.isfinite(conn.positions).all(1)

        # Every drawable connection; lines go to those carrying the most signal each step.
        self.paths = paths is not None
        if self.paths:
            self.n_edges = PATHS_SHOWN  # each is far busier than a straight line
            pre, post, lines = paths
            self.path_pts = np.concatenate(lines)
            self.path_start = np.cumsum([0] + [len(p) for p in lines])
            # How far along its path (0..1) each point is, for placing the pulses.
            frac = []
            for p in lines:
                d = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
                frac.append(d / max(d[-1], 1e-6))
            self.path_frac = np.concatenate(frac)
            self.glow = np.zeros(len(lines))
        else:
            post, pre = np.nonzero(weights * self.has_pos[:, None] * self.has_pos[None, :])
        self.syn_pre, self.syn_post, self.syn_w = pre, post, weights[post, pre]
        self.usual = usual.copy()
        self.set_trace(np.zeros((1, conn.n)), np.zeros((1, conn.n)), 0)

    def cycle_view(self):
        self.view = (self.view + 1) % len(VIEWS)

    def cycle_mode(self):
        self.mode = (self.mode + 1) % len(MODES)

    def set_trace(self, rates, attr, chosen):
        """Take in one move's (steps, N) firing rates and attributions."""
        self.rates, self.chosen = rates, chosen
        change = rates - self.usual[:len(rates)] if len(rates) == len(self.usual) else 0 * rates
        # Signed modes share one scale per move, so later steps can visibly grow.
        # "why" adds up each step's push: interneurons only matter at earlier steps (at the
        # last one only DNs can still reach the readout). sqrt lifts small values so DNs
        # don't outshine every interneuron.
        self.signed = {m: np.sign(v) * np.sqrt(np.abs(v) / max(np.percentile(np.abs(v), 99.5), 1e-9))
                       for m, v in [("change", change), ("why", attr.cumsum(0))]}
        if len(rates) == len(self.usual):  # slowly adapt "usual" to the boards seen in play
            self.usual += 0.05 * (rates - self.usual)

        self.edges = []
        for r in rates:
            flow = self.syn_w * r[self.syn_pre]
            top = np.argpartition(-np.abs(flow), min(self.n_edges, len(flow) - 1))[:self.n_edges]
            top = top[flow[top] != 0]
            self.edges.append((self.syn_pre[top], self.syn_post[top], flow[top], top))
        self.max_flow = max([np.abs(e[2]).max() for e in self.edges if len(e[2])], default=1.0)

    def _layout(self):
        """(pixels x neurons) matrix, outline background, soma screen positions and
        voxel->screen fn for this view."""
        if self.view in self._layouts:
            return self._layouts[self.view]
        conn, morph = self.conn, self.morph
        _, ax, ay = VIEWS[self.view]
        w, h = self.rect.size
        pts = [conn.positions[self.has_pos][:, [ax, ay]]]
        if morph is not None:
            pts += [morph.seg_a[:, [ax, ay]], morph.mesh_verts[:, [ax, ay]]]
        lo = np.min([p.min(0) for p in pts if len(p)], 0)
        hi = np.max([p.max(0) for p in pts if len(p)], 0)
        m = 24
        # One scale for both axes keeps the brain's true proportions.
        scale = min((w - 2 * m) / max(hi[0] - lo[0], 1e-6), (h - 2 * m) / max(hi[1] - lo[1], 1e-6))
        off = (np.array([w, h]) - (hi - lo) * scale) / 2 - lo * scale

        def project(p):
            return p[:, [ax, ay]] * scale + off

        def keys(pix, owner):
            """Unique (neuron, pixel) keys of the points that land on screen."""
            pix = np.round(pix).astype(np.int64)
            ok = (pix[:, 0] >= 0) & (pix[:, 0] < w) & (pix[:, 1] >= 0) & (pix[:, 1] < h)
            return np.unique(owner[ok].astype(np.int64) * (w * h) + pix[ok, 0] * h + pix[ok, 1])

        key = []
        drawn = np.zeros(conn.n, bool)
        if morph is not None and len(morph.seg_a):
            # Sample the branches in chunks: all at once takes several GB for big circuits.
            for c in range(0, len(morph.seg_a), SEG_CHUNK):
                a, b = project(morph.seg_a[c:c + SEG_CHUNK]), project(morph.seg_b[c:c + SEG_CHUNK])
                n_samp = np.ceil(np.linalg.norm(b - a, axis=1)).astype(int) + 1
                seg = np.repeat(np.arange(len(a)), n_samp)
                k = np.arange(n_samp.sum()) - np.repeat(np.cumsum(n_samp) - n_samp, n_samp)
                t = k / np.repeat(np.maximum(n_samp - 1, 1), n_samp)
                key.append(keys(a[seg] + (b - a)[seg] * t[:, None],
                                morph.seg_neuron[c:c + SEG_CHUNK][seg]))
            drawn[morph.seg_neuron] = True
        soma = project(np.nan_to_num(conn.positions))
        o, d = self._soma_kernels(np.flatnonzero(self.has_pos & ~drawn))
        key.append(keys(soma[o] + d, o))

        key = np.unique(np.concatenate(key))
        overlap = np.bincount(key % (w * h))
        exposure = min(1.0, EXPOSURE_OVERLAP / max(np.median(overlap[overlap > 0]), 1))
        mat = sparse.csr_matrix((np.full(len(key), exposure, np.float32),
                                 (key % (w * h), key // (w * h))), shape=(w * h, conn.n))

        bg = np.zeros((w, h, 3), np.float32)
        if morph is not None and len(morph.mesh_faces):
            v = project(morph.mesh_verts)
            for part in np.unique(morph.mesh_part):
                mask_surf = pygame.Surface((w, h))
                for tri in v[morph.mesh_faces[morph.mesh_part == part]]:
                    pygame.draw.polygon(mask_surf, (255, 255, 255), tri)
                mask = pygame.surfarray.array_red(mask_surf) > 0
                inner = mask.copy()
                inner[1:] &= mask[:-1]; inner[:-1] &= mask[1:]
                inner[:, 1:] &= mask[:, :-1]; inner[:, :-1] &= mask[:, 1:]
                bg[mask] = np.maximum(bg[mask], OUTLINE_FILL)
                bg[mask & ~inner] = OUTLINE_EDGE

        self._layouts[self.view] = (mat, bg, soma, project)
        return self._layouts[self.view]

    def _soma_kernels(self, idx):
        """Pixel offsets of a disc per neuron in idx: (neuron (P,), offset (P, 2))."""
        owner, off = [], []
        for i in idx:
            r = self.radius[i]
            dx, dy = np.mgrid[-r:r + 1, -r:r + 1].reshape(2, -1)
            keep = dx ** 2 + dy ** 2 <= r * r
            owner.append(np.full(keep.sum(), i))
            off.append(np.column_stack([dx[keep], dy[keep]]))
        if not owner:
            return np.zeros(0, int), np.zeros((0, 2), int)
        return np.concatenate(owner), np.concatenate(off)

    def _cloud_3d(self):
        """Points for the 3D view, centred on the circuit and in units of its radius:
        points (P, 3) with owner neuron (P,), pixel offset (P, 2) and brightness (P,)
        (branch samples, or a disc per soma) and outline points (Q, 3), all as tensors
        on DEVICE;
        somas (N, 3), pixels per unit and the voxels -> units fn."""
        if self._cloud is not None:
            return self._cloud
        conn, morph = self.conn, self.morph
        w, h = self.rect.size
        extent = [conn.positions[self.has_pos]]
        if morph is not None:
            extent += [morph.seg_a, morph.mesh_verts]
        extent = np.concatenate(extent)
        # Turn about, and fit, the bulk of the circuit: the neurons far down the nerve cord
        # (~5-10%) would otherwise shrink the brain to a speck and swing it off centre.
        # They leave the frame when the circuit turns side-on.
        centre = np.median(extent, 0)
        radius = max(np.percentile(np.linalg.norm(extent - centre, axis=1), 90), 1e-6)

        def unit(p):
            return (np.asarray(p, np.float32) - centre) / radius

        # Keep that bulk on screen all the way round: the nearest points are magnified
        # by CAMERA / (CAMERA - 1).
        e = unit(extent)
        r_side = max(np.percentile(np.hypot(e[:, 0], e[:, 2]), 90), 1e-6)
        r_up = max(np.percentile(np.abs(e[:, 1]), 90), 1e-6)
        px = min(0.46 * w / r_side, 0.46 * h / r_up) * (CAMERA - 1) / CAMERA

        pts, owner, off, bright = [], [], [], []
        drawn = np.zeros(conn.n, bool)
        if morph is not None and len(morph.seg_a):
            a, b = unit(morph.seg_a), unit(morph.seg_b)
            want = np.linalg.norm(b - a, axis=1) * px + 1  # ~1 sample per pixel
            want *= min(1.0, MAX_3D_POINTS / want.sum())
            # Random rounding keeps the total near the cap even with many tiny segments.
            rng = np.random.default_rng(0)
            n_samp = (want + rng.random(len(want))).astype(int)
            seg = np.repeat(np.arange(len(a)), n_samp)
            k = np.arange(n_samp.sum()) - np.repeat(np.cumsum(n_samp) - n_samp, n_samp)
            t = (k + 0.5) / np.repeat(np.maximum(n_samp, 1), n_samp)
            pts.append(a[seg] + (b - a)[seg] * t[:, None])
            owner.append(morph.seg_neuron[seg])
            off.append(np.zeros((len(seg), 2), int))
            bright.append(np.full(len(seg), min(1.0, EXPOSURE_POINTS / max(len(seg), 1))))
            drawn[morph.seg_neuron] = True
        soma = unit(np.nan_to_num(conn.positions))
        o, d = self._soma_kernels(np.flatnonzero(self.has_pos & ~drawn))
        pts.append(soma[o]); owner.append(o); off.append(d); bright.append(np.ones(len(o)))

        outline = np.zeros((0, 3), np.float32)
        if morph is not None and len(morph.mesh_verts):
            rng = np.random.default_rng(0)
            v = morph.mesh_verts
            outline = unit(v[rng.choice(len(v), min(len(v), MAX_OUTLINE_POINTS), replace=False)])

        def gpu(x, dtype=torch.float32):
            return torch.as_tensor(np.asarray(x), dtype=dtype, device=DEVICE)
        self._cloud = (gpu(np.concatenate(pts)), gpu(np.concatenate(owner), torch.long),
                       gpu(np.concatenate(off)), gpu(np.concatenate(bright)), gpu(outline),
                       soma, px, unit)
        return self._cloud

    def _render_3d(self, colors, dim=1.0):
        """Image (w, h, 3) uint8, soma screen positions (N, 2), depth brightness (N,) and
        a voxels -> (screen positions, depth brightness) fn. The point cloud is drawn
        with torch on DEVICE; somas and signal paths are few, so they stay in numpy.
        dim scales the neurons' brightness after compression (the outline is untouched)."""
        pts, owner, off, bright, outline, soma, px, unit = self._cloud_3d()
        w, h = self.rect.size
        yaw = pygame.time.get_ticks() / 1000 * SPIN
        cy, sy, ct, st = np.cos(yaw), np.sin(yaw), np.cos(TILT), np.sin(TILT)
        rot = (np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]])
               @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])).astype(np.float32)

        def project(p):
            q = np.asarray(p, np.float32) @ rot.T
            persp = CAMERA / (CAMERA + q[:, 2])
            xy = q[:, :2] * (px * persp)[:, None] + np.array([w / 2, h / 2], np.float32)
            # Depth cue: the far side of the brain fades into the dark.
            return xy, 0.25 + 0.75 * np.clip((1 - q[:, 2]) / 2, 0, 1)

        with torch.no_grad():
            rot_t = torch.from_numpy(rot).to(DEVICE)
            centre = torch.tensor([w / 2, h / 2], device=DEVICE)

            def project_t(p):
                q = p @ rot_t.T
                persp = CAMERA / (CAMERA + q[:, 2])
                return (q[:, :2] * (px * persp)[:, None] + centre,
                        0.25 + 0.75 * ((1 - q[:, 2]) / 2).clamp(0, 1))

            def splat(xy, vals):
                xy = xy.round().long()
                ok = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
                idx = xy[ok, 0] * h + xy[ok, 1]
                return torch.zeros(w * h, 3, device=DEVICE).index_add_(0, idx, vals[ok])

            xy, fog = project_t(pts)
            col = torch.as_tensor(colors, dtype=torch.float32, device=DEVICE)
            light = splat(xy + off, col[owner] * (fog * bright)[:, None])
            # Same brightness compression as _compress, done on the device.
            peak = light.amax(1, keepdim=True)
            light *= 255 * dim * (1 - torch.exp(-1.5 * peak)) / peak.clamp_min(1e-6)
            if len(outline):
                oxy, ofog = project_t(outline)
                edge = torch.as_tensor(OUTLINE_EDGE * 0.6, dtype=torch.float32, device=DEVICE)
                light += splat(oxy, edge * ofog[:, None]).clamp(max=90)
            # Soft glow: a blurred copy (shrunk 6x and blown back up) added at half strength.
            img = light.clamp(0, 255).reshape(w, h, 3).permute(2, 0, 1)[None]
            glow = F.interpolate(F.interpolate(img, size=(w // 6, h // 6), mode="area"),
                                 size=(w, h), mode="bilinear", align_corners=False)
            img = (img + 0.5 * glow).clamp(0, 255)[0].permute(1, 2, 0).to(torch.uint8).cpu().numpy()
        soma_xy, soma_fog = project(soma)
        return img, soma_xy.astype(float), soma_fog, lambda p: project(unit(p))

    @staticmethod
    def _compress(light):
        """Overlapping branches add, then brightness is compressed (keeping the colour
        mix) so dense regions don't blow out to white."""
        light = np.asarray(light, np.float32)
        peak = np.maximum(np.maximum(light[:, 0], light[:, 1]), light[:, 2])  # >> light.max(1)
        lit = np.flatnonzero(peak > 0)  # usually a small part of the frame: skip the rest
        light[lit] *= (255 * (1 - np.exp(-1.5 * peak[lit])) / peak[lit])[:, None]
        return light

    def _colors(self, step):
        """(N, 3) colour per neuron in 0..1 for the current mode."""
        mode = MODES[self.mode]
        if mode == "activity":
            # Idle neurons glow faintly (dense neuropil adds up brighter); active ones light up.
            return self.base / 255 * (0.04 + self.rates[step][:, None])
        v = np.clip(self.signed[mode][step], -1, 1)[:, None]
        return (np.where(v > 0, POS_COLOR, NEG_COLOR) * np.abs(v)
                + ROLE_COLORS["hid"] * 0.03) / 255

    def _draw_paths(self, surf, step, to_screen):
        """Glow along the real wiring of the connections carrying the most signal, with a
        pulse running soma -> branches -> synapse -> branches -> soma on each."""
        target = np.zeros(len(self.glow))
        _, _, flow, idx = self.edges[step]
        target[idx] = np.minimum(np.abs(flow) / self.max_flow, 1)
        self.glow += GLOW_EASE * (target - self.glow)  # fade in and out between steps
        live = np.flatnonzero(self.glow > 0.02)
        if not len(live):
            return
        start, end = self.path_start[live], self.path_start[live + 1]
        pts = np.concatenate([np.arange(a, b) for a, b in zip(start, end)])
        xy, depth = to_screen(self.path_pts[pts])
        xy = xy.astype(float)  # pygame wants float64 points
        frac = self.path_frac[pts]
        at = np.concatenate([[0], np.cumsum(end - start)])
        now = pygame.time.get_ticks() / 1000

        # Drawn on black and added on top, so overlapping paths glow brighter.
        layer = pygame.Surface(self.rect.size)
        for k, i in enumerate(live):
            p, f = xy[at[k]:at[k + 1]], frac[at[k]:at[k + 1]]
            if len(p) < 2:
                continue
            base = EXC_COLOR if self.syn_w[i] > 0 else INH_COLOR
            g = self.glow[i] * depth[at[k]:at[k + 1]].mean()
            pygame.draw.lines(layer, np.minimum(255, base * 1.2 * g).astype(int), False, p,
                              2 if g > 0.25 else 1)
            phase = (now / PULSE_SECONDS + 0.618 * i) % 1  # spread pulses out across paths
            for j, lag in enumerate((0, 0.015, 0.03, 0.045, 0.06)):  # bright head, fading tail
                if phase - lag < 0:
                    break
                c = (np.interp(phase - lag, f, p[:, 0]), np.interp(phase - lag, f, p[:, 1]))
                col = np.minimum(255, (base * 0.5 + 160) * (0.4 + 0.6 * g) * (1 - j / 5))
                pygame.draw.circle(layer, col.astype(int), c, 4 if j == 0 else 3 - j // 2)
        # A blurred copy underneath gives the paths and pulses a soft halo.
        w, h = self.rect.size
        halo = pygame.transform.smoothscale(pygame.transform.smoothscale(layer, (w // 4, h // 4)),
                                            (w, h))
        surf.blit(halo, self.rect.topleft, special_flags=pygame.BLEND_ADD)
        surf.blit(layer, self.rect.topleft, special_flags=pygame.BLEND_ADD)

    def _legend(self):
        move = ACTIONS[self.chosen].upper()
        mode = MODES[self.mode]
        if mode == "activity":
            rows = [("sensory (board input)", ROLE_COLORS["in"]),
                    ("interneurons", ROLE_COLORS["hid"]),
                    ("descending / motor", ROLE_COLORS["out"])]
            if len(self.conn.dopamine_idx):
                rows.append(("dopamine", ROLE_COLORS["da"]))
        elif mode == "change":
            rows = [("more active than usual", POS_COLOR), ("less active than usual", NEG_COLOR)]
        else:
            rows = [(f"pushed toward {move}", POS_COLOR), (f"pushed away from {move}", NEG_COLOR)]
        return rows

    def draw(self, surf, step):
        step = min(step, len(self.rates) - 1)
        colors = self._colors(step)
        # Let the signal paths stand out. Dimmed after compression: before it, dense
        # regions would still saturate and the paths drawn over them wash out to white.
        dim = PATH_BACKDROP if self.show_edges and self.paths else 1.0
        is_3d = VIEWS[self.view][1] is None
        if is_3d:
            img, soma, depth, to_screen = self._render_3d(colors, dim)
        else:
            mat, bg, soma, project = self._layout()
            img = bg + dim * self._compress(mat @ colors).reshape(bg.shape)
            depth = np.ones(len(soma))

            def to_screen(p):
                return project(p), np.ones(len(p))
        pygame.surfarray.blit_array(self.surface, img.clip(0, 255).astype(np.uint8))
        surf.blit(self.surface, self.rect.topleft)

        if self.show_edges and self.paths:
            self._draw_paths(surf, step, to_screen)
        elif self.show_edges:
            edge_layer = pygame.Surface(self.rect.size, pygame.SRCALPHA)
            for pre, post, f in zip(*self.edges[step][:3]):
                a = (30 + 190 * min(abs(f) / self.max_flow, 1)) * (depth[pre] + depth[post]) / 2
                col = (*EXC_COLOR, int(a)) if f > 0 else (*INH_COLOR, int(a))
                pygame.draw.line(edge_layer, col, soma[pre], soma[post], 1)
            surf.blit(edge_layer, self.rect.topleft)

        rows = self._legend()
        if self.show_edges:
            rows += [("excitatory signal", EXC_COLOR), ("inhibitory signal", INH_COLOR)]
        for k, (label, col) in enumerate(rows):
            y = self.rect.y + 14 + k * 20
            pygame.draw.circle(surf, col, (self.rect.x + 20, y + 7), 5)
            surf.blit(self.font.render(label, True, (220, 220, 230)), (self.rect.x + 32, y))
        for k, text in enumerate([f"{VIEWS[self.view][0]} view  (V)",
                                  f"{MODE_TITLES[MODES[self.mode]]}  (M)",
                                  f"step {step + 1}/{len(self.rates)}"]):
            t = self.font.render(text, True, (150, 150, 170))
            surf.blit(t, (self.rect.right - t.get_width() - 14, self.rect.y + 14 + k * 20))


def draw_board(surf, font, board, origin):
    x0, y0 = origin
    n = board.shape[0]
    cell = BOARD_PX // n
    pygame.draw.rect(surf, (187, 173, 160), (x0, y0, BOARD_PX, BOARD_PX), border_radius=10)
    for r in range(n):
        for c in range(n):
            v = int(board[r, c])
            rect = pygame.Rect(x0 + c * cell + 6, y0 + r * cell + 6, cell - 12, cell - 12)
            pygame.draw.rect(surf, TILE_COLORS.get(v, (60, 58, 50)), rect, border_radius=6)
            if v:
                txt = font.render(str(v), True, (119, 110, 101) if v <= 4 else (249, 246, 242))
                surf.blit(txt, txt.get_rect(center=rect.center))


def draw_probs(surf, font, probs, chosen, origin):
    x0, y0 = origin
    for a, name in enumerate(ACTIONS):
        y = y0 + a * 30
        col = MOVE_COLORS[a] if a == chosen else (MOVE_COLORS[a] + 2 * np.array(BG)) / 3
        surf.blit(font.render(name, True, (80, 80, 80)), (x0, y))
        pygame.draw.rect(surf, col, (x0 + 60, y + 4, int(probs[a] * 150), 18), border_radius=4)
        surf.blit(font.render(f"{probs[a]:.2f}", True, (80, 80, 80)), (x0 + 220, y))


def draw_race(surf, font, step_probs, chosen, step, rect):
    """The move probabilities if the brain had stopped at each time step so far."""
    rect = pygame.Rect(rect)
    pygame.draw.rect(surf, (238, 232, 220), rect, border_radius=6)
    surf.blit(font.render("probability by time step", True, (130, 120, 110)),
              (rect.x + 6, rect.y + 4))
    if step_probs is None:
        return
    plot = rect.inflate(-16, -34).move(0, 9)
    lo, hi = 0.0, 1.0
    n = len(step_probs)
    step = min(step, n - 1)

    def xy(t, v):
        return (plot.x + plot.w * t / max(n - 1, 1), float(plot.bottom - plot.h * (v - lo) / (hi - lo)))

    x = xy(step, lo)[0]
    pygame.draw.line(surf, (200, 192, 180), (x, plot.y), (x, plot.bottom), 1)
    for a in sorted(range(4), key=lambda a: a == chosen):  # chosen move drawn on top
        pts = [xy(t, step_probs[t, a]) for t in range(step + 1)]
        width = 3 if a == chosen else 1
        if len(pts) > 1:
            pygame.draw.lines(surf, MOVE_COLORS[a], False, pts, width)
        pygame.draw.circle(surf, MOVE_COLORS[a], pts[-1], width + 1)


def wiring_paths(conn, morph, syn):
    """Route each connection in `syn` along the real wiring: from the sending neuron's
    soma along its branches to the synapse, then along the receiving neuron's branches
    to its soma. Returns one (P, 3) polyline (voxels) per connection. Without a traced
    skeleton, that half is a straight line between the soma and the synapse."""
    trees = {}
    if morph is not None:
        order = np.argsort(morph.seg_neuron, kind="stable")
        bounds = np.searchsorted(morph.seg_neuron[order], np.arange(conn.n + 1))

    def tree(i):
        """(skeleton nodes (V, 3), distance from the soma (V,), predecessor (V,)) or None."""
        if i not in trees:
            seg = order[bounds[i]:bounds[i + 1]] if morph is not None else []
            if not len(seg):
                trees[i] = None
                return None
            a, b = morph.seg_a[seg], morph.seg_b[seg]
            # Skeletons are snapped to a grid, so segments that touch share exact endpoints.
            nodes, inv = np.unique(np.concatenate([a, b]), axis=0, return_inverse=True)
            inv = inv.ravel()
            n = len(nodes)
            g = sparse.csr_matrix((np.linalg.norm(a - b, axis=1) + 1e-3, (inv[:len(a)], inv[len(a):])),
                                  shape=(n, n))
            soma = np.argmin(np.linalg.norm(nodes - conn.positions[i], axis=1))
            dist, pred = dijkstra(g, directed=False, indices=soma, return_predecessors=True)
            trees[i] = (nodes, dist, pred)
        return trees[i]

    def to_synapse(i, site):
        """Neuron i's soma, then along its branches to the point nearest the synapse."""
        t = tree(i)
        if t is None:
            return conn.positions[i][None]
        nodes, dist, pred = t
        reach = np.flatnonzero(np.isfinite(dist))  # skip skeleton fragments cut off from the soma
        route = [reach[np.argmin(np.linalg.norm(nodes[reach] - site, axis=1))]]
        while pred[route[-1]] >= 0:
            route.append(pred[route[-1]])
        return np.concatenate([conn.positions[i][None], nodes[route[::-1]]])

    return [np.concatenate([to_synapse(i, s), s[None], to_synapse(j, s)[::-1]]).astype(np.float32)
            for i, j, s in zip(syn.pre, syn.post, syn.site)]


def usual_activity(model, n_boards=256, seed=0):
    """(steps, N) average firing over boards from random play: what "usual" looks like."""
    game, boards = Game2048(seed=seed), []
    while len(boards) < n_boards:
        if game.done:
            game.reset()
        boards.append(encode_board(game.board))
        game.step(game.rng.choice(np.flatnonzero(game.valid_moves())))
    obs = torch.tensor(np.array(boards), device=next(model.parameters()).device)
    with torch.no_grad():
        return model(obs, return_trace=True)[1].mean(1).cpu().numpy()


def run(cfg, checkpoint=None, max_frames=None):
    conn = load_connectome(cfg["connectome"])
    morph = load_morphology(cfg["connectome"])
    if morph is not None and not np.array_equal(morph.body_ids, conn.body_ids):
        print("Neuron shapes are for a different connectome (re-run: fetch --shapes); showing dots.")
        morph = None
    syn = load_synapses(cfg["connectome"])
    if syn is not None and not np.array_equal(syn.body_ids, conn.body_ids):
        print("Synapse locations are for a different connectome (re-run: fetch --shapes); "
              "drawing straight connection lines.")
        syn = None
    paths = None
    if syn is not None:
        print(f"Routing {len(syn.pre)} connections along the neurons' branches...")
        paths = (syn.pre, syn.post, wiring_paths(conn, morph, syn))
    model = load_model(conn, checkpoint)[0] if checkpoint else build_model(cfg, conn)
    model.to(DEVICE).eval()

    pygame.init()
    screen = pygame.display.set_mode((W_WIN, H_WIN))
    pygame.display.set_caption("Fly brain plays 2048")
    big, small, tiny = (pygame.font.SysFont(None, s) for s in (52, 26, 20))
    clock = pygame.time.Clock()
    with torch.no_grad():
        W = model.effective_weights()  # fixed while playing: computed once, reused every move
    brain = BrainView(conn, (BOARD_PX + 2 * PAD, PAD, W_WIN - BOARD_PX - 3 * PAD, H_WIN - 2 * PAD),
                      W.cpu().numpy(), usual_activity(model), morph, paths)

    game = Game2048()
    trace, probs, chosen, frame, race = None, np.full(4, 0.25), -1, 0, None
    paused, speed, step_once, frames = False, SPEEDS.index(3), False, 0
    # Game clock: animation ticks owed, when moves were made, and when the speed last changed
    # (the moves/s actually managed is measured from then).
    due, made, since = 0.0, [], time.perf_counter()
    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_SPACE, pygame.K_UP, pygame.K_DOWN, pygame.K_r):
                    since = time.perf_counter()
                if e.key == pygame.K_SPACE: paused = not paused
                elif e.key == pygame.K_n: step_once = paused
                elif e.key == pygame.K_UP: speed = min(speed + 1, len(SPEEDS) - 1)
                elif e.key == pygame.K_DOWN: speed = max(speed - 1, 0)
                elif e.key == pygame.K_r: game.reset(); trace = None
                elif e.key == pygame.K_v: brain.cycle_view()
                elif e.key == pygame.K_m: brain.cycle_mode()
                elif e.key == pygame.K_e: brain.show_edges = not brain.show_edges

        # The game runs on its own clock, not the redraw rate: a move is its recurrent steps
        # plus a short linger, and the game owes SPEEDS[speed] moves' worth of those ticks per
        # second of real time. Several ticks (even whole moves) can run between two frames;
        # whatever the model can't get through in SIM_BUDGET is dropped rather than piling up.
        dt = clock.tick(RENDER_FPS) / 1000
        if not paused or step_once:
            due += dt * SPEEDS[speed] * ((len(trace) if trace is not None else model.steps) + 2)
            deadline = time.perf_counter() + SIM_BUDGET
            while due >= 1 and time.perf_counter() < deadline:
                due -= 1
                if trace is None:  # start a move: run the brain once, then animate its steps
                    if game.done:
                        step_once = False
                        break
                    obs = torch.tensor(encode_board(game.board), device=DEVICE)[None]
                    logits, rates, attr = (x.cpu() for x in
                                           model.explain(obs, game.valid_moves()[None], W))
                    race = torch.softmax(logits, -1).numpy()
                    probs = race[-1]
                    chosen = int(probs.argmax())
                    trace, frame = rates.numpy(), 0
                    brain.set_trace(trace, attr.numpy(), chosen)
                else:
                    frame += 1
                    if frame > len(trace):  # after lingering a moment on the final state
                        game.step(chosen)
                        trace = None
                        made.append(time.perf_counter())
                        if step_once:
                            step_once, due = False, 0.0
                            break
            due = min(due, 1.0)

        # Between moves (game over, or before the first one) keep showing the last state.
        step = min(frame, len(trace) - 1) if trace is not None else 10 ** 9

        screen.fill(BG)
        draw_board(screen, big, game.board, (PAD, PAD + 60))
        screen.blit(big.render(f"Score {game.score}", True, (119, 110, 101)), (PAD, PAD))
        draw_probs(screen, small, probs, chosen, (PAD, PAD + BOARD_PX + 90))
        draw_race(screen, tiny, race, chosen, step,
                  (PAD + 280, PAD + BOARD_PX + 84, BOARD_PX - 280, 126))
        status = ("GAME OVER - press R" if game.done else
                  "PAUSED" if paused else f"{SPEEDS[speed]:g} moves/s")
        # At high speeds the model itself is the limit: say what it is actually managing.
        now = time.perf_counter()
        window = max(now - 4, since)
        made = [t for t in made if t > window]
        if not (paused or game.done) and SPEEDS[speed] >= 5 and now - since > 1.5:
            actual = len(made) / (now - window)
            if actual < 0.8 * SPEEDS[speed]:
                status += f" (managing {actual:.0f})"
        screen.blit(small.render(status, True, (150, 60, 60)), (PAD, H_WIN - 40))
        brain.draw(screen, step)
        pygame.display.flip()
        frames += 1
        if max_frames and frames >= max_frames:
            break
    pygame.quit()
