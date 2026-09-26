"""Watch the fly brain play 2048.

Left:   the game board.
Right:  every neuron in the sub-circuit at its real place in the fly's nervous
        system (its traced branches, if `fetch --shapes` was run), lighting up as
        activity propagates from sensory neurons (blue) through interneurons
        (grey) to descending/motor neurons (orange).
Bottom: the probability the brain assigns to each move.

Controls: SPACE pause | N single move while paused | UP/DOWN speed | R restart |
          V rotate brain view | ESC quit
"""
import numpy as np
import pygame
import torch
from scipy import sparse

from .connectome import load_connectome, load_morphology
from .game import ACTIONS, Game2048, encode_board
from .model import masked_logits
from .train import build_model, load_model

W_WIN, H_WIN = 1280, 760
BOARD_PX, PAD = 460, 40
BG = (250, 248, 239)
TILE_COLORS = {0: (205, 193, 180), 2: (238, 228, 218), 4: (237, 224, 200), 8: (242, 177, 121),
               16: (245, 149, 99), 32: (246, 124, 95), 64: (246, 94, 59), 128: (237, 207, 114),
               256: (237, 204, 97), 512: (237, 200, 80), 1024: (237, 197, 63), 2048: (237, 194, 46)}
ROLE_COLORS = {"in": np.array([70, 140, 255]), "hid": np.array([170, 170, 185]),
               "out": np.array([255, 140, 40])}
# (name, data axis shown left->right, data axis shown top->bottom); x/y/z are neuPrint voxels.
VIEWS = [("front", 0, 1), ("top", 0, 2), ("side", 2, 1)]
OUTLINE_FILL, OUTLINE_EDGE = np.array([28, 30, 46]), np.array([72, 78, 112])


class BrainView:
    """Neurons drawn where they really are in the fly's nervous system.

    With neuron shapes (`fetch --shapes`) each neuron's traced branches are drawn
    over an outline of the brain; without them each neuron is a dot at its soma.
    Each view is precomputed as a sparse (pixels x neurons) matrix, so lighting up
    every branch each frame is a single matrix multiply.
    """

    def __init__(self, conn, rect, morph=None, n_edges=600):
        self.rect = pygame.Rect(rect)
        self.conn, self.morph = conn, morph
        self.view, self._layouts = 0, {}
        self.surface = pygame.Surface(self.rect.size)
        self.font = pygame.font.SysFont(None, 22)

        self.role = np.full(conn.n, "hid", dtype=object)
        self.role[conn.input_idx] = "in"
        self.role[conn.output_idx] = "out"
        self.base = np.stack([ROLE_COLORS[r] for r in self.role]).astype(np.float32)
        self.radius = np.where(self.role == "out", 5, np.where(self.role == "in", 3, 2))
        self.has_pos = np.isfinite(conn.positions).all(1)

        # Without shapes, show the strongest connections as lines (or the panel becomes solid ink).
        self.edges = []
        if morph is None:
            w = np.abs(conn.weights)
            flat = np.argsort(w, axis=None)[::-1][:n_edges]
            post, pre = np.unravel_index(flat, w.shape)
            self.edges = [(a, b, conn.weights[b, a] > 0) for a, b in zip(pre, post)
                          if w[b, a] > 0 and self.has_pos[a] and self.has_pos[b]]

    def cycle_view(self):
        self.view = (self.view + 1) % len(VIEWS)

    def _layout(self):
        """(pixels x neurons) matrix, outline background and neuron->screen fn for this view."""
        if self.view in self._layouts:
            return self._layouts[self.view]
        conn, morph = self.conn, self.morph
        _, ax, ay = VIEWS[self.view]
        w, h = self.rect.size
        pts = [conn.positions[self.has_pos]]
        if morph is not None:
            pts += [morph.seg_a, morph.mesh_verts]
        pts = np.concatenate(pts)[:, [ax, ay]]
        lo, hi = pts.min(0), pts.max(0)
        m = 24
        # One scale for both axes keeps the brain's true proportions.
        scale = min((w - 2 * m) / max(hi[0] - lo[0], 1e-6), (h - 2 * m) / max(hi[1] - lo[1], 1e-6))
        off = (np.array([w, h]) - (hi - lo) * scale) / 2 - lo * scale

        def project(p):
            return p[:, [ax, ay]] * scale + off

        pix, owner = [], []
        drawn = np.zeros(conn.n, bool)
        if morph is not None and len(morph.seg_a):
            a, b = project(morph.seg_a), project(morph.seg_b)
            n_samp = np.ceil(np.linalg.norm(b - a, axis=1)).astype(int) + 1
            seg = np.repeat(np.arange(len(a)), n_samp)
            k = np.arange(n_samp.sum()) - np.repeat(np.cumsum(n_samp) - n_samp, n_samp)
            t = k / np.repeat(np.maximum(n_samp - 1, 1), n_samp)
            pix.append(a[seg] + (b - a)[seg] * t[:, None])
            owner.append(morph.seg_neuron[seg])
            drawn[morph.seg_neuron] = True
        soma = project(np.nan_to_num(conn.positions))
        for i in np.flatnonzero(self.has_pos & ~drawn):
            r = self.radius[i]
            dx, dy = np.mgrid[-r:r + 1, -r:r + 1].reshape(2, -1)
            keep = dx ** 2 + dy ** 2 <= r * r
            pix.append(soma[i] + np.column_stack([dx[keep], dy[keep]]))
            owner.append(np.full(keep.sum(), i))

        pix = np.round(np.concatenate(pix)).astype(int) if pix else np.zeros((0, 2), int)
        owner = np.concatenate(owner) if owner else np.zeros(0, int)
        ok = (pix[:, 0] >= 0) & (pix[:, 0] < w) & (pix[:, 1] >= 0) & (pix[:, 1] < h)
        key = np.unique(owner[ok] * (w * h) + pix[ok, 0] * h + pix[ok, 1])
        mat = sparse.csr_matrix((np.ones(len(key), np.float32), (key % (w * h), key // (w * h))),
                                shape=(w * h, conn.n))

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

        self._layouts[self.view] = (mat, bg, soma)
        return self._layouts[self.view]

    def draw(self, surf, activity):
        mat, bg, soma = self._layout()
        w, h = self.rect.size
        # Idle neurons glow faintly (dense neuropil adds up brighter); active ones light up.
        # Overlapping branches add, then brightness is compressed (keeping the colour mix)
        # so dense regions don't blow out to white.
        light = mat @ (self.base / 255 * (0.04 + activity[:, None]))
        peak = light.max(1, keepdims=True)
        light *= 255 * (1 - np.exp(-1.5 * peak)) / np.maximum(peak, 1e-6)
        img = (bg + light.reshape(w, h, 3)).clip(0, 255).astype(np.uint8)
        pygame.surfarray.blit_array(self.surface, img)
        surf.blit(self.surface, self.rect.topleft)

        if self.edges:
            edge_layer = pygame.Surface(self.rect.size, pygame.SRCALPHA)
            for pre, post, exc in self.edges:
                a = int(25 + 150 * activity[pre])
                col = (90, 200, 120, a) if exc else (220, 80, 110, a)
                pygame.draw.line(edge_layer, col, soma[pre], soma[post], 1)
            surf.blit(edge_layer, self.rect.topleft)

        for k, (label, role) in enumerate([("sensory (board input)", "in"),
                                           ("interneurons", "hid"),
                                           ("descending / motor", "out")]):
            y = self.rect.y + 14 + k * 20
            pygame.draw.circle(surf, ROLE_COLORS[role], (self.rect.x + 20, y + 7), 5)
            surf.blit(self.font.render(label, True, (220, 220, 230)), (self.rect.x + 32, y))
        view = self.font.render(f"{VIEWS[self.view][0]} view  (V to rotate)", True, (150, 150, 170))
        surf.blit(view, (self.rect.right - view.get_width() - 14, self.rect.y + 14))


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
        surf.blit(font.render(name, True, (80, 80, 80)), (x0, y))
        w = int(probs[a] * 300)
        col = (255, 140, 40) if a == chosen else (200, 190, 180)
        pygame.draw.rect(surf, col, (x0 + 80, y + 4, w, 18), border_radius=4)
        surf.blit(font.render(f"{probs[a]:.2f}", True, (80, 80, 80)), (x0 + 390, y))


def run(cfg, checkpoint=None, max_frames=None):
    conn = load_connectome(cfg["connectome"])
    morph = load_morphology(cfg["connectome"])
    if morph is not None and not np.array_equal(morph.body_ids, conn.body_ids):
        print("Neuron shapes are for a different connectome (re-run: fetch --shapes); showing dots.")
        morph = None
    model = load_model(conn, checkpoint)[0] if checkpoint else build_model(cfg, conn)
    model.eval()

    pygame.init()
    screen = pygame.display.set_mode((W_WIN, H_WIN))
    pygame.display.set_caption("Fly brain plays 2048")
    big, small = pygame.font.SysFont(None, 52), pygame.font.SysFont(None, 26)
    clock = pygame.time.Clock()
    brain = BrainView(conn, (BOARD_PX + 2 * PAD, PAD, W_WIN - BOARD_PX - 3 * PAD, H_WIN - 2 * PAD),
                      morph)

    game = Game2048()
    trace, probs, chosen, frame = None, np.full(4, 0.25), -1, 0
    paused, fps, step_once, frames = False, 30, False, 0
    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_SPACE: paused = not paused
                elif e.key == pygame.K_n: step_once = True
                elif e.key == pygame.K_UP: fps = min(240, fps * 2)
                elif e.key == pygame.K_DOWN: fps = max(2, fps // 2)
                elif e.key == pygame.K_r: game.reset(); trace = None
                elif e.key == pygame.K_v: brain.cycle_view()

        # Each move: run the brain once, then animate its internal time steps.
        advance = not paused or step_once
        if trace is None and not game.done and advance:
            with torch.no_grad():
                obs = torch.tensor(encode_board(game.board))[None]
                logits, tr = model(obs, return_trace=True)
                logits = masked_logits(logits, game.valid_moves()[None])
                probs = torch.softmax(logits, -1)[0].numpy()
            chosen = int(probs.argmax())
            trace, frame = tr[:, 0].numpy(), 0

        activity = trace[min(frame, len(trace) - 1)] if trace is not None else np.zeros(conn.n)
        if trace is not None and advance:
            frame += 1
            if frame >= len(trace) + 2:  # linger a moment on the final state
                game.step(chosen)
                trace, step_once = None, False

        screen.fill(BG)
        draw_board(screen, big, game.board, (PAD, PAD + 60))
        screen.blit(big.render(f"Score {game.score}", True, (119, 110, 101)), (PAD, PAD))
        draw_probs(screen, small, probs, chosen, (PAD, PAD + BOARD_PX + 90))
        status = "GAME OVER - press R" if game.done else ("PAUSED" if paused else f"{fps} fps")
        screen.blit(small.render(status, True, (150, 60, 60)), (PAD, H_WIN - 40))
        brain.draw(screen, activity)
        pygame.display.flip()
        clock.tick(fps)
        frames += 1
        if max_frames and frames >= max_frames:
            break
    pygame.quit()
