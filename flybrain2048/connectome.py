"""Load a sub-circuit of the fly connectome as a signed weight matrix.

Two sources:
  - "synthetic": a random layered graph with fly-like statistics, so you can
    develop offline with zero setup.
  - "neuprint":  the real Janelia/Google male CNS connectome via neuprint-python.

Everything ends up in a `Connectome` object that is cached to an .npz file.
Convention: weights[post, pre] = signed synapse strength from pre -> post.
"""
import os
from dataclasses import dataclass

import numpy as np

# Neurotransmitters treated as inhibitory (in the fly, glutamate is mostly
# inhibitory in the central brain via GluCl receptors; this is a simplification).
INHIBITORY_NT = {"gaba", "glutamate"}


class _NpzCache:
    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez_compressed(path, **self.__dict__)

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=True)
        return cls(**{k: d[k] for k in d.files})


@dataclass
class Connectome(_NpzCache):
    body_ids: np.ndarray      # (N,) neuron ids
    types: np.ndarray         # (N,) cell-type strings
    positions: np.ndarray     # (N, 3) xyz soma positions (NaN if unknown), for the display
    weights: np.ndarray       # (N, N) float32, signed, weights[post, pre]
    input_idx: np.ndarray     # indices of "sensory" neurons that receive the board
    output_idx: np.ndarray    # indices of "motor" (descending) neurons
    output_groups: np.ndarray # (len(output_idx),) action 0..3 each output votes for

    @property
    def n(self):
        return len(self.body_ids)

    def summary(self):
        nnz = int((self.weights != 0).sum())
        inh = int((self.weights < 0).sum())
        return (f"{self.n} neurons | {nnz} connections ({inh} inhibitory) | "
                f"{len(self.input_idx)} inputs, {len(self.output_idx)} outputs")


# --------------------------------------------------------------------- utils
def _signed_log_weights(counts, signs):
    """counts[post, pre] synapse counts -> signed, compressed weights."""
    w = np.log1p(counts.astype(np.float32)) * signs[None, :]
    return w.astype(np.float32)


def _groups_round_robin(n_out):
    return np.arange(n_out) % 4


# ----------------------------------------------------------------- synthetic
def build_synthetic(n_in=128, n_hidden=512, n_out=32, p=0.05, frac_inhib=0.3, seed=0):
    rng = np.random.default_rng(seed)
    n = n_in + n_hidden + n_out
    layer = np.array([0] * n_in + [1] * n_hidden + [2] * n_out)

    # Mostly feed-forward (in->hidden->out) with recurrence inside hidden.
    allowed = ((layer[None, :] == 0) & (layer[:, None] == 1)) \
            | ((layer[None, :] == 1) & (layer[:, None] == 1)) \
            | ((layer[None, :] == 1) & (layer[:, None] == 2))
    conn = (rng.random((n, n)) < p) & allowed
    np.fill_diagonal(conn, False)
    counts = np.where(conn, rng.lognormal(1.5, 1.0, (n, n)).astype(int) + 1, 0)

    signs = np.where(rng.random(n) < frac_inhib, -1.0, 1.0).astype(np.float32)
    signs[:n_in] = 1.0  # sensory neurons excitatory

    # Fake 3D positions: layers laid out left -> right, like a brain-to-VNC axis.
    pos = np.column_stack([
        layer * 4.0 + rng.normal(0, 0.4, n),
        rng.normal(0, 1, n),
        rng.normal(0, 1, n),
    ])
    types = np.array(["SENS"] * n_in + ["INT"] * n_hidden + ["DN"] * n_out)
    return Connectome(
        body_ids=np.arange(n),
        types=types,
        positions=pos.astype(np.float32),
        weights=_signed_log_weights(counts, signs),
        input_idx=np.arange(n_in),
        output_idx=np.arange(n_in + n_hidden, n),
        output_groups=_groups_round_robin(n_out),
    )


# ------------------------------------------------------------------ neuPrint
def build_from_neuprint(cfg):
    """Build a 3-layer sub-circuit ending on descending neurons (DNs).

    Strategy (a reasonable starting point; tweak freely):
      outputs = descending neurons matching `output_type_regex` (brain -> VNC motor commands)
      hidden  = the strongest upstream partners of those DNs
      inputs  = the strongest upstream partners of the hidden layer
    Then fetch *all* connections among the chosen neurons (including recurrence).
    """
    from neuprint import Client, NeuronCriteria as NC, fetch_neurons, \
        fetch_simple_connections, fetch_adjacencies

    token = os.environ.get("NEUPRINT_APPLICATION_CREDENTIALS")
    if not token:
        raise RuntimeError("Set NEUPRINT_APPLICATION_CREDENTIALS to your neuPrint token "
                           "(neuprint.janelia.org -> Account -> Auth Token).")
    # Keep a reference: neuprint only holds the default client weakly, so an
    # unassigned Client is garbage-collected before the queries below run.
    client = Client(cfg["server"], dataset=cfg["dataset"], token=token)

    min_w = cfg.get("min_weight", 5)

    # 1) Output layer: descending neurons
    dn_df, _ = fetch_neurons(NC(type=cfg["output_type_regex"], regex=True))
    dn_df = dn_df.dropna(subset=["type"]).sort_values("type")
    out_ids = dn_df["bodyId"].to_numpy()[: cfg["n_out"]]
    print(f"  outputs: {len(out_ids)} descending neurons")

    def top_upstream(targets, k, exclude):
        conns = fetch_simple_connections(None, NC(bodyId=list(targets)), min_weight=min_w)
        conns = conns[~conns["bodyId_pre"].isin(exclude)]
        ranked = conns.groupby("bodyId_pre")["weight"].sum().sort_values(ascending=False)
        return ranked.index.to_numpy()[:k]

    # 2) Hidden + 3) input layers
    hid_ids = top_upstream(out_ids, cfg["n_hidden"], set(out_ids))
    print(f"  hidden:  {len(hid_ids)} upstream partners")
    in_ids = top_upstream(hid_ids, cfg["n_in"], set(out_ids) | set(hid_ids))
    print(f"  inputs:  {len(in_ids)} second-order upstream partners")

    all_ids = np.concatenate([in_ids, hid_ids, out_ids])
    index = {b: i for i, b in enumerate(all_ids)}

    # 4) All connections among the chosen neurons
    _, conn_df = fetch_adjacencies(NC(bodyId=list(all_ids)), NC(bodyId=list(all_ids)),
                                   min_total_weight=1)
    conn_df = conn_df.groupby(["bodyId_pre", "bodyId_post"], as_index=False)["weight"].sum()
    n = len(all_ids)
    counts = np.zeros((n, n), dtype=np.int64)
    counts[conn_df["bodyId_post"].map(index), conn_df["bodyId_pre"].map(index)] = conn_df["weight"]

    # 5) Neuron properties: type, neurotransmitter sign, soma position
    props, _ = fetch_neurons(NC(bodyId=list(all_ids)))
    props = props.set_index("bodyId").loc[all_ids]
    nt_col = next((c for c in ["consensusNt", "predictedNt", "celltypePredictedNt"]
                   if c in props.columns), None)
    if nt_col is None:
        print("  ! no neurotransmitter column found; treating all neurons as excitatory")
        signs = np.ones(n, dtype=np.float32)
    else:
        nts = props[nt_col].fillna("").str.lower()
        signs = np.where(nts.isin(INHIBITORY_NT), -1.0, 1.0).astype(np.float32)

    pos = np.full((n, 3), np.nan)
    if "somaLocation" in props.columns:
        for i, loc in enumerate(props["somaLocation"]):
            if isinstance(loc, (list, tuple)) and len(loc) == 3:
                pos[i] = loc

    return Connectome(
        body_ids=all_ids,
        types=props["type"].fillna("?").astype(str).to_numpy(),
        positions=pos.astype(np.float32),
        weights=_signed_log_weights(counts, signs),
        input_idx=np.arange(len(in_ids)),
        output_idx=np.arange(len(in_ids) + len(hid_ids), n),
        output_groups=_groups_round_robin(len(out_ids)),
    )


# ---------------------------------------------------------------- morphology
@dataclass
class Morphology(_NpzCache):
    """Real neuron shapes and brain outline for the display (neuPrint only)."""
    body_ids: np.ndarray    # (N,) the Connectome these shapes were fetched for
    seg_a: np.ndarray       # (S, 3) float32 skeleton segment start points
    seg_b: np.ndarray       # (S, 3) float32 skeleton segment end points
    seg_neuron: np.ndarray  # (S,) int32 neuron index (Connectome order) of each segment
    mesh_verts: np.ndarray  # (V, 3) float32 brain-region mesh vertices
    mesh_faces: np.ndarray  # (F, 3) int32 triangles into mesh_verts
    mesh_part: np.ndarray   # (F,) int32 which brain region each triangle belongs to


def _parse_obj(data):
    verts, faces = [], []
    for line in data.decode().splitlines():
        if line.startswith("v "):
            verts.append([float(t) for t in line.split()[1:4]])
        elif line.startswith("f "):
            idx = [int(t.split("/")[0]) - 1 for t in line.split()[1:]]
            faces += [[idx[0], idx[k], idx[k + 1]] for k in range(1, len(idx) - 1)]
    return np.array(verts, np.float32).reshape(-1, 3), np.array(faces, np.int32).reshape(-1, 3)


def build_morphology(cfg, conn):
    """Download every neuron's skeleton plus the top-level brain-region meshes.

    Skeleton nodes are snapped to a `skeleton_grid`-voxel grid and duplicate
    segments dropped, which shrinks the data ~10x with no visible difference.
    """
    from concurrent.futures import ThreadPoolExecutor

    from neuprint import Client, fetch_roi_hierarchy, fetch_skeleton

    token = os.environ.get("NEUPRINT_APPLICATION_CREDENTIALS")
    if not token:
        raise RuntimeError("Set NEUPRINT_APPLICATION_CREDENTIALS to your neuPrint token.")
    # Kept referenced for the same weak-default-client reason as build_from_neuprint.
    client = Client(cfg["server"], dataset=cfg["dataset"], token=token)
    grid = cfg.get("skeleton_grid", 100)

    def segments(i):
        try:
            df = fetch_skeleton(int(conn.body_ids[i]), format="pandas")
        except Exception:
            return None
        xyz = np.round(df[["x", "y", "z"]].to_numpy() / grid) * grid
        pos = dict(zip(df["rowId"], xyz))
        pairs = [(pos[r], pos[l]) for r, l in zip(df["rowId"], df["link"]) if l in pos]
        if not pairs:
            return None
        seg = np.unique(np.array(pairs, np.float32).reshape(-1, 6), axis=0)
        seg = seg[(seg[:, :3] != seg[:, 3:]).any(1)]
        return seg

    print(f"Fetching {conn.n} neuron skeletons...")
    with ThreadPoolExecutor(8) as pool:
        segs = list(pool.map(segments, range(conn.n)))
    missing = sum(s is None for s in segs)
    print(f"  got {conn.n - missing} skeletons ({missing} unavailable, shown as soma dots)")
    parts = [np.full(len(s), i, np.int32) for i, s in enumerate(segs) if s is not None]
    segs = np.concatenate([s for s in segs if s is not None])

    hierarchy = fetch_roi_hierarchy(False, False, "dict")
    root = next(iter(hierarchy))
    rois = cfg.get("outline_rois") or list(hierarchy[root])
    print(f"Fetching {len(rois)} brain-region meshes for the outline...")
    verts, faces, face_part, offset = [], [], [], 0
    for k, roi in enumerate(rois):
        try:
            v, f = _parse_obj(client.fetch_roi_mesh(roi))
        except Exception as e:
            print(f"  ! skipping {roi}: {e}")
            continue
        verts.append(v)
        faces.append(f + offset)
        face_part.append(np.full(len(f), k, np.int32))
        offset += len(v)

    return Morphology(
        body_ids=conn.body_ids,
        seg_a=segs[:, :3], seg_b=segs[:, 3:], seg_neuron=np.concatenate(parts),
        mesh_verts=np.concatenate(verts) if verts else np.zeros((0, 3), np.float32),
        mesh_faces=np.concatenate(faces) if faces else np.zeros((0, 3), np.int32),
        mesh_part=np.concatenate(face_part) if face_part else np.zeros(0, np.int32),
    )


def load_morphology(cfg, conn=None, refresh=False):
    """Cached neuron shapes, or None if they haven't been fetched (or aren't available)."""
    path = cfg.get("morphology_path", "data/morphology_{source}.npz").format(source=cfg["source"])
    if os.path.exists(path) and not refresh:
        m = Morphology.load(path)
        if conn is None or np.array_equal(m.body_ids, conn.body_ids):
            return m
        print("Cached neuron shapes are for a different connectome; re-fetching.")
    if conn is None:
        return None
    if cfg["source"] != "neuprint":
        raise ValueError("Neuron shapes are only available with connectome.source: neuprint")
    m = build_morphology(cfg["neuprint"], conn)
    m.save(path)
    print(f"Saved to {path}: {len(m.seg_a)} skeleton segments, {len(m.mesh_faces)} mesh triangles")
    return m


# -------------------------------------------------------------------- entry
def load_connectome(cfg, refresh=False):
    path = cfg["cache_path"].format(source=cfg["source"])
    if os.path.exists(path) and not refresh:
        return Connectome.load(path)
    print(f"Building connectome from source '{cfg['source']}'...")
    if cfg["source"] == "synthetic":
        c = build_synthetic(**cfg["synthetic"])
    elif cfg["source"] == "neuprint":
        c = build_from_neuprint(cfg["neuprint"])
    else:
        raise ValueError(f"Unknown connectome source: {cfg['source']}")
    c.save(path)
    print(f"Saved to {path}: {c.summary()}")
    return c
