"""Load a sub-circuit of the fly connectome as a signed weight matrix.

Two sources:
  - "synthetic": a random layered graph with fly-like statistics, so you can
    develop offline with zero setup.
  - "neuprint":  the real Janelia/Google male CNS connectome via neuprint-python.

Everything ends up in a `Connectome` object that is cached to an .npz file.
Convention: weights[post, pre] = signed synapse strength from pre -> post.
"""
import os
from dataclasses import dataclass, field

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
    # Dopamine neurons, which the critic can read (model.critic: dopamine). Empty in
    # connectomes cached before these were added.
    dopamine_idx: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))

    @property
    def n(self):
        return len(self.body_ids)

    def summary(self):
        nnz = int((self.weights != 0).sum())
        inh = int((self.weights < 0).sum())
        return (f"{self.n} neurons | {nnz} connections ({inh} inhibitory) | "
                f"{len(self.input_idx)} inputs, {len(self.output_idx)} outputs, "
                f"{len(self.dopamine_idx)} dopamine")


# --------------------------------------------------------------------- utils
def _signed_log_weights(counts, signs):
    """counts[post, pre] synapse counts -> signed, compressed weights."""
    w = np.log1p(counts.astype(np.float32)) * signs[None, :]
    return w.astype(np.float32)


def _groups_round_robin(n_out):
    return np.arange(n_out) % 4


# ----------------------------------------------------------------- synthetic
def build_synthetic(n_in=128, n_hidden=512, n_out=32, p=0.05, frac_inhib=0.3, seed=0,
                    n_dopamine=0):
    rng = np.random.default_rng(seed)
    n = n_in + n_hidden + n_out + n_dopamine
    layer = np.array([0] * n_in + [1] * n_hidden + [2] * n_out + [3] * n_dopamine)

    # Mostly feed-forward (in->hidden->out) with recurrence inside hidden; dopamine
    # neurons listen to the hidden layer and feed back into it.
    allowed = ((layer[None, :] == 0) & (layer[:, None] == 1)) \
            | ((layer[None, :] == 1) & (layer[:, None] == 1)) \
            | ((layer[None, :] == 1) & (layer[:, None] == 2)) \
            | ((layer[None, :] == 1) & (layer[:, None] == 3)) \
            | ((layer[None, :] == 3) & (layer[:, None] == 1))
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
    types = np.array(["SENS"] * n_in + ["INT"] * n_hidden + ["DN"] * n_out + ["DAN"] * n_dopamine)
    return Connectome(
        body_ids=np.arange(n),
        types=types,
        positions=pos.astype(np.float32),
        weights=_signed_log_weights(counts, signs),
        input_idx=np.arange(n_in),
        output_idx=np.arange(n_in + n_hidden, n_in + n_hidden + n_out),
        output_groups=_groups_round_robin(n_out),
        dopamine_idx=np.arange(n - n_dopamine, n),
    )


# ------------------------------------------------------------------ neuPrint
def build_from_neuprint(cfg):
    """Build a 3-layer sub-circuit ending on descending neurons (DNs).

    Strategy (a reasonable starting point; tweak freely):
      outputs  = descending neurons matching `output_type_regex` (brain -> VNC motor commands)
      hidden   = the strongest upstream partners of those DNs
      inputs   = the strongest upstream partners of the hidden layer
      dopamine = the dopamine neurons (`dopamine.type_regex`) that get the most synapses
                 from those three layers, so their activity can depend on the board
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

    # 4) Dopamine neurons, added after the others so existing neurons keep their indices.
    base = set(in_ids) | set(hid_ids) | set(out_ids)
    da_ids, new_da = [], []
    dcfg = cfg.get("dopamine") or {}
    if dcfg.get("n", 0) > 0:
        conns = fetch_simple_connections(NC(bodyId=list(base)),
                                         NC(type=dcfg["type_regex"], regex=True), min_weight=min_w)
        ranked = conns.groupby("bodyId_post")["weight"].sum().sort_values(ascending=False)
        da_ids = ranked.index.to_numpy()[: dcfg["n"]]
        if not len(da_ids):
            raise RuntimeError(f"No neurons of types {dcfg['type_regex']!r} receive synapses from "
                               f"the circuit. Check the dopamine type names on neuPrint.")
        new_da = [b for b in da_ids if b not in base]
        print(f"  dopamine: {len(da_ids)} neurons driven by the circuit "
              f"({len(da_ids) - len(new_da)} already in it)")

    all_ids = np.concatenate([in_ids, hid_ids, out_ids, np.array(new_da, dtype=in_ids.dtype)])
    index = {b: i for i, b in enumerate(all_ids)}

    # 5) All connections among the chosen neurons
    _, conn_df = fetch_adjacencies(NC(bodyId=list(all_ids)), NC(bodyId=list(all_ids)),
                                   min_total_weight=1)
    conn_df = conn_df.groupby(["bodyId_pre", "bodyId_post"], as_index=False)["weight"].sum()
    n = len(all_ids)
    counts = np.zeros((n, n), dtype=np.int64)
    counts[conn_df["bodyId_post"].map(index), conn_df["bodyId_pre"].map(index)] = conn_df["weight"]

    # 6) Neuron properties: type, neurotransmitter sign, soma position
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
        if len(da_ids):
            n_da = int((nts.to_numpy()[[index[b] for b in da_ids]] == "dopamine").sum())
            print(f"  ({n_da} of the {len(da_ids)} dopamine-type neurons have dopamine as "
                  f"their {nt_col})")

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
        output_idx=np.arange(len(in_ids) + len(hid_ids), len(in_ids) + len(hid_ids) + len(out_ids)),
        output_groups=_groups_round_robin(len(out_ids)),
        dopamine_idx=np.array([index[b] for b in da_ids], dtype=np.int64),
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


def build_morphology(cfg, conn, reuse=None):
    """Download every neuron's skeleton plus the top-level brain-region meshes.

    Skeleton nodes are snapped to a `skeleton_grid`-voxel grid and duplicate
    segments dropped, which shrinks the data ~10x with no visible difference.
    reuse: shapes cached for another connectome; neurons (by body id) and the brain
    outline found there are copied instead of downloaded again.
    """
    from concurrent.futures import ThreadPoolExecutor

    from neuprint import Client, fetch_roi_hierarchy, fetch_skeleton

    token = os.environ.get("NEUPRINT_APPLICATION_CREDENTIALS")
    if not token:
        raise RuntimeError("Set NEUPRINT_APPLICATION_CREDENTIALS to your neuPrint token.")
    # Kept referenced for the same weak-default-client reason as build_from_neuprint.
    client = Client(cfg["server"], dataset=cfg["dataset"], token=token)
    grid = cfg.get("skeleton_grid", 100)

    old = {}
    if reuse is not None and len(reuse.seg_a):
        order = np.argsort(reuse.seg_neuron, kind="stable")
        bounds = np.searchsorted(reuse.seg_neuron[order], np.arange(len(reuse.body_ids) + 1))
        for j, body in enumerate(reuse.body_ids):
            seg = order[bounds[j]:bounds[j + 1]]
            if len(seg):
                old[int(body)] = np.concatenate([reuse.seg_a[seg], reuse.seg_b[seg]], 1)

    def segments(i):
        if int(conn.body_ids[i]) in old:
            return old[int(conn.body_ids[i])]
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

    n_new = sum(int(b) not in old for b in conn.body_ids)
    print(f"Fetching {n_new} neuron skeletons ({conn.n - n_new} reused from the cache)...")
    with ThreadPoolExecutor(8) as pool:
        segs = list(pool.map(segments, range(conn.n)))
    missing = sum(s is None for s in segs)
    print(f"  got {conn.n - missing} skeletons ({missing} unavailable, shown as soma dots)")
    parts = [np.full(len(s), i, np.int32) for i, s in enumerate(segs) if s is not None]
    segs = np.concatenate([s for s in segs if s is not None])

    verts, faces, face_part, offset = [], [], [], 0
    if reuse is not None and len(reuse.mesh_faces):
        rois = []
        verts, faces, face_part = [reuse.mesh_verts], [reuse.mesh_faces], [reuse.mesh_part]
        print("Reusing the cached brain outline.")
    else:
        hierarchy = fetch_roi_hierarchy(False, False, "dict")
        root = next(iter(hierarchy))
        rois = cfg.get("outline_rois") or list(hierarchy[root])
        print(f"Fetching {len(rois)} brain-region meshes for the outline...")
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
    """Cached neuron shapes, or None if they haven't been fetched (or aren't available).
    Shapes cached for a different connectome are reused for the neurons they share,
    unless refresh."""
    path = cfg.get("morphology_path", "data/morphology_{source}.npz").format(source=cfg["source"])
    reuse = None
    if os.path.exists(path) and not refresh:
        m = Morphology.load(path)
        if conn is None or np.array_equal(m.body_ids, conn.body_ids):
            return m
        print("Cached neuron shapes are for a different connectome; fetching the new neurons.")
        reuse = m
    if conn is None:
        return None
    if cfg["source"] != "neuprint":
        raise ValueError("Neuron shapes are only available with connectome.source: neuprint")
    m = build_morphology(cfg["neuprint"], conn, reuse)
    m.save(path)
    print(f"Saved to {path}: {len(m.seg_a)} skeleton segments, {len(m.mesh_faces)} mesh triangles")
    return m


# ------------------------------------------------------------------ synapses
@dataclass
class Synapses(_NpzCache):
    """Where the circuit's strongest connections physically are, so the display can
    route each one along real wiring (neuPrint only)."""
    body_ids: np.ndarray    # (N,) the Connectome these were fetched for
    pre: np.ndarray         # (C,) int32 sending neuron index (Connectome order)
    post: np.ndarray        # (C,) int32 receiving neuron index
    site: np.ndarray        # (C, 3) float32 one synapse of that connection, in voxels


def build_synapses(cfg, conn, reuse=None):
    """Fetch one representative synapse for each of the `path_connections` strongest
    connections between neurons with a known soma: the synapse nearest the middle
    of all of that pair's synapses. reuse: synapse locations cached for another
    connectome; connections (by body ids) found there are copied, not re-fetched."""
    from concurrent.futures import ThreadPoolExecutor

    from neuprint import Client, fetch_synapse_connections

    token = os.environ.get("NEUPRINT_APPLICATION_CREDENTIALS")
    if not token:
        raise RuntimeError("Set NEUPRINT_APPLICATION_CREDENTIALS to your neuPrint token.")
    client = Client(cfg["server"], dataset=cfg["dataset"], token=token)

    has_pos = np.isfinite(conn.positions).all(1)
    w = np.abs(conn.weights) * has_pos[:, None] * has_pos[None, :]
    k = min(cfg.get("path_connections", 3000), int((w > 0).sum()))
    flat = np.argpartition(-w.ravel(), k - 1)[:k] if k else np.zeros(0, int)
    post, pre = np.unravel_index(flat, w.shape)

    rows = []
    if reuse is not None:
        index = {int(b): i for i, b in enumerate(conn.body_ids)}
        old = {(int(reuse.body_ids[a]), int(reuse.body_ids[b])): site
               for a, b, site in zip(reuse.pre, reuse.post, reuse.site)}
        keep = np.ones(len(pre), bool)
        for k_, (i, j) in enumerate(zip(pre, post)):
            key = (int(conn.body_ids[i]), int(conn.body_ids[j]))
            if key in old:
                rows.append((index[key[0]], index[key[1]], old[key]))
                keep[k_] = False
        print(f"  reusing {len(rows)} cached synapse locations")
        pre, post = pre[keep], post[keep]

    def sites(i):
        targets = post[pre == i]
        try:
            df = fetch_synapse_connections(int(conn.body_ids[i]), conn.body_ids[targets].tolist(),
                                           client=client)
        except Exception:
            return []
        out = []
        for body, g in df.groupby("bodyId_post"):
            xyz = g[["x_post", "y_post", "z_post"]].to_numpy(np.float32)
            mid = np.median(xyz, 0)
            out.append((i, int(np.flatnonzero(conn.body_ids == body)[0]),
                        xyz[np.argmin(np.linalg.norm(xyz - mid, axis=1))]))
        return out

    senders = np.unique(pre)
    print(f"Fetching synapse locations for {len(pre)} connections from {len(senders)} neurons...")
    with ThreadPoolExecutor(8) as pool:
        rows += [r for rs in pool.map(sites, senders) for r in rs]
    print(f"  got {len(rows)} of {k}")
    return Synapses(
        body_ids=conn.body_ids,
        pre=np.array([r[0] for r in rows], np.int32),
        post=np.array([r[1] for r in rows], np.int32),
        site=np.array([r[2] for r in rows], np.float32).reshape(-1, 3),
    )


def load_synapses(cfg, conn=None, refresh=False):
    """Cached synapse locations, or None if they haven't been fetched (or aren't available)."""
    path = cfg.get("synapse_path", "data/synapses_{source}.npz").format(source=cfg["source"])
    reuse = None
    if os.path.exists(path) and not refresh:
        s = Synapses.load(path)
        if conn is None or np.array_equal(s.body_ids, conn.body_ids):
            return s
        print("Cached synapse locations are for a different connectome; fetching the new ones.")
        reuse = s
    if conn is None:
        return None
    if cfg["source"] != "neuprint":
        raise ValueError("Synapse locations are only available with connectome.source: neuprint")
    s = build_synapses(cfg["neuprint"], conn, reuse)
    s.save(path)
    print(f"Saved to {path}: {len(s.pre)} synapse locations")
    return s


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
