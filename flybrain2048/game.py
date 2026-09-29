"""A minimal, fast 2048 engine (numpy only, no graphics).

Actions: 0=up, 1=right, 2=down, 3=left
"""
import numpy as np

ACTIONS = ["up", "right", "down", "left"]
# np.rot90 rotates counter-clockwise; we rotate the board so every move becomes "left".
_ROT = {0: 1, 1: 2, 2: 3, 3: 0}


class Game2048:
    def __init__(self, size=4, seed=None):
        self.size = size
        self.rng = np.random.default_rng(seed)
        self.reset()

    # ------------------------------------------------------------------ core
    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=np.int64)
        self.score = 0
        self.moves = 0
        self.done = False
        self._spawn()
        self._spawn()
        return self.board.copy()

    def _spawn(self):
        empty = np.argwhere(self.board == 0)
        if len(empty) == 0:
            return
        r, c = empty[self.rng.integers(len(empty))]
        self.board[r, c] = 2 if self.rng.random() < 0.9 else 4

    @staticmethod
    def _merge_left(row):
        tiles = [x for x in row if x]
        out, gained, i = [], 0, 0
        while i < len(tiles):
            if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
                out.append(tiles[i] * 2)
                gained += tiles[i] * 2
                i += 2
            else:
                out.append(tiles[i])
                i += 1
        out += [0] * (len(row) - len(out))
        return out, gained

    def _slide(self, board, action):
        k = _ROT[action]
        b = np.rot90(board, k)
        new, gained = [], 0
        for row in b:
            r, g = self._merge_left(row)
            new.append(r)
            gained += g
        return np.rot90(np.array(new, dtype=np.int64), -k), gained

    def valid_moves(self):
        """Boolean mask of length 4: which actions actually change the board."""
        return np.array([not np.array_equal(self._slide(self.board, a)[0], self.board)
                         for a in range(4)])

    def step(self, action):
        """Returns (board, reward, done, info). Invalid moves do nothing."""
        new, gained = self._slide(self.board, action)
        valid = not np.array_equal(new, self.board)
        if valid:
            self.board = new
            self.score += gained
            self.moves += 1
            self._spawn()
        self.done = not self.valid_moves().any()
        return self.board.copy(), gained, self.done, {"valid": valid}

    @property
    def max_tile(self):
        return int(self.board.max())


# 2048 plays the same rotated or mirrored, so every board has 8 equivalent versions.
# Symmetry s rotates the board s % 4 quarter-turns counter-clockwise, then mirrors it
# left-right if s >= 4. SYM_ACTIONS[s][a] is where move a ends up on the transformed board.
_MIRROR = np.array([0, 3, 2, 1])
SYM_ACTIONS = np.array([_MIRROR[(np.arange(4) - s % 4) % 4] if s >= 4 else (np.arange(4) - s) % 4
                        for s in range(8)])


def transform_board(board, s):
    b = np.rot90(board, s % 4)
    return np.fliplr(b) if s >= 4 else b


def encode_board(board, levels=16):
    """One-hot encode each cell by log2(tile): 16 cells x 16 levels = 256 floats.

    This is the 'sensory input' the fly brain sees.
    """
    b = np.asarray(board).flatten()
    lv = np.zeros_like(b)
    nz = b > 0
    lv[nz] = np.log2(b[nz]).astype(np.int64)
    lv = np.clip(lv, 0, levels - 1)
    out = np.zeros((b.size, levels), dtype=np.float32)
    out[np.arange(b.size), lv] = 1.0
    return out.flatten()
