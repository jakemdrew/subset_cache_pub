"""
Training data and batch computation for the Thesis Validation GAAP Medium
Full Set model: SEM's bidirectional encoder over the filing's facts, then the
interaction stack over the encoder's per layer, per head KV caches, with
full attention over the full set (one direct softmax) and over every sampled
subset (shared prefix states, the invention).

Worker side (DataLoader worker processes; Windows spawns workers by
importing this module, so everything they run lives here):
  gaap_keep            TV4's feature dropout, reproduced exactly.
  SubsetFilingDataset  one filing per item, addressed as (position, epoch,
                       train): in training, TV4's feature dropout first, then
                       an epoch keyed uniform stride over the full power set
                       of the RETAINED facts (local positions); in evaluation,
                       every fact and a fixed subset key. Tries are built in
                       one pass from the canonical order (no T x T mask).
  compact_items        worker side collate: tries reduced to plan arrays.
  collate_forest       pads a batch and merges the filings' subset tries into
                       one forest plan, so the level walk runs once per batch.

Main process (GPU):
  plan_to_device, batch_forward, batch_ret_loss, combine_predictions.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

from set_attention_cache import sample_subsets_powerset, build_prefix_trie_canonical
from set_attention_model import encoder_layer_kv, subset_attention_levels_exact_bwd


def gaap_keep(p, epoch, n0, seed, keep_min=0.5, min_keep=4):
    """TV4's feature dropout (Thesis Validation GAAP.ipynb, TV4), unchanged:
    keep ceil(U(keep_min, 1) x n0) facts, at least min_keep, drawn by the
    generator seeded (seed, epoch, p, 0); p = the filing's position in the
    fine tune population. Returns sorted retained positions."""
    rng0 = np.random.default_rng((seed, epoch, int(p), 0))
    k0 = max(min_keep, int(math.ceil(rng0.uniform(keep_min, 1.0) * n0)))
    return np.sort(rng0.choice(n0, size=k0, replace=False)) if k0 < n0 else np.arange(n0)


class SubsetFilingDataset(torch.utils.data.Dataset):
    """indptr/indices/data: CSR slices of the population's rows, in
    population position order (canonical order within each row: ascending
    registry column); file_names and targets aligned with positions.
    M = subset budget per filing over the full power set (0 = full set
    only; sample_subsets_powerset returns at most M). Items are addressed as
    (position, epoch, train): train = 1 applies TV4's feature dropout and
    keys the subsets by epoch; train = 0 uses every fact and eval_epoch.
    Arguments after M are keyword only."""

    def __init__(self, indptr, indices, data, file_names, targets, M, *,
                 seed, key_tag=None, keep_min=0.5, min_keep=4, eval_epoch=1000000):
        self.indptr = np.asarray(indptr, dtype=np.int64)
        self.indices = np.asarray(indices, dtype=np.int32)
        self.data = np.asarray(data, dtype=np.float32)
        self.file_names = list(file_names)
        self.targets = np.asarray(targets, dtype=np.float32)
        if not (len(self.indptr) == len(self.file_names) + 1 == len(self.targets) + 1):
            raise ValueError("indptr, file_names and targets disagree in length")
        if (np.diff(self.indptr) < 1).any():
            raise ValueError("every filing needs at least one fact")
        self.M, self.seed, self.key_tag = int(M), seed, key_tag
        self.keep_min, self.min_keep, self.eval_epoch = keep_min, int(min_keep), int(eval_epoch)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, key):
        p, epoch, train = key
        lo, hi = int(self.indptr[p]), int(self.indptr[p + 1])
        cols = self.indices[lo:hi].astype(np.int64)
        vals = self.data[lo:hi]
        if train:
            sel = gaap_keep(p, epoch, hi - lo, self.seed, self.keep_min, self.min_keep)
            cols, vals = cols[sel], vals[sel]
        n = len(cols)
        subs = (sample_subsets_powerset(n, self.file_names[p], self.M, seed=self.seed,
                                        epoch=int(epoch) if train else self.eval_epoch,
                                        key_tag=self.key_tag)
                if self.M > 0 else [])
        return {"i": int(p), "cols": cols, "vals": vals, "n": n,
                "trie": build_prefix_trie_canonical(subs) if subs else None,
                "y": float(self.targets[p])}


def forest_level_plan_np(tries, offsets):
    """Merge per filing subset tries (None = no subsets) into one forest.
    Fact positions are shifted by offsets[b] (the filing's start in the
    flattened batch); node and parent indices are shifted so each filing's
    trie stays separate. Returns a numpy plan: 'levels' (per depth: pos,
    parent_local, end_local), 'inv' (restores forest path order: filing 0's
    subsets, then filing 1's, ...), and 'counts' (subsets per filing).
    Level membership comes from one stable sort by depth, so the cost does
    not grow with nodes times levels; node order within a level is index
    order, as before."""
    pos_l, par_l, dep_l, term_l, counts = [], [], [], [], []
    base = 0
    for tr, off in zip(tries, offsets):
        if tr is None:
            counts.append(0)
            continue
        T = len(tr["pos"])
        pos_l.append(np.asarray(tr["pos"], dtype=np.int64) + int(off))
        p = np.asarray(tr["parent"], dtype=np.int64).copy()
        p[p >= 0] += base
        par_l.append(p)
        dep_l.append(np.asarray(tr["depth"], dtype=np.int64))
        terms = tr["terminals"] if "terminals" in tr else [q[-1] for q in tr["paths"]]
        term_l.append(np.asarray(terms, dtype=np.int64) + base)
        counts.append(len(terms))
        base += T
    if base == 0:
        return {"levels": [], "inv": None, "counts": counts}
    pos, parent = np.concatenate(pos_l), np.concatenate(par_l)
    depth, term = np.concatenate(dep_l), np.concatenate(term_l)
    D = int(depth.max()) + 1
    node_order = np.argsort(depth, kind="stable")                      # per level, nodes in index order
    n_off = np.concatenate([[0], np.cumsum(np.bincount(depth, minlength=D))])
    local = np.empty(len(pos), dtype=np.int64)
    local[node_order] = np.arange(len(pos), dtype=np.int64) - np.repeat(n_off[:-1], np.diff(n_off))
    term_depth = depth[term]
    t_order = np.argsort(term_depth, kind="stable")                    # per level, subsets in forest order
    t_off = np.concatenate([[0], np.cumsum(np.bincount(term_depth, minlength=D))])
    pos_by_level = pos[node_order]
    parent_local_by_level = local[parent[node_order]]                  # depth 0 entries are never read
    end_local_by_level = local[term[t_order]]
    levels = []
    for d in range(D):
        a, b, c, e = n_off[d], n_off[d + 1], t_off[d], t_off[d + 1]
        levels.append({"pos": pos_by_level[a:b],
                       "parent_local": parent_local_by_level[a:b] if d > 0 else None,
                       "end_local": end_local_by_level[c:e] if e > c else None})
    return {"levels": levels, "inv": np.argsort(t_order), "counts": counts}


def compact_items(items):
    """Worker side collate for training and evaluation loaders: each item's
    trie reduced to the arrays the forest plan needs (pos, parent, depth,
    terminals), so whole logical batches cross the process boundary cheaply.
    The main process microbatches the items and calls collate_forest."""
    out = []
    for it in items:
        tr = it["trie"]
        if tr is not None:
            terms = tr["terminals"] if "terminals" in tr else [q[-1] for q in tr["paths"]]
            tr = {"pos": np.asarray(tr["pos"], dtype=np.int32),
                  "parent": np.asarray(tr["parent"], dtype=np.int32),
                  "depth": np.asarray(tr["depth"], dtype=np.int16),
                  "terminals": np.asarray(terms, dtype=np.int32)}
        out.append({**it, "trie": tr})
    return out


def collate_forest(items):
    """Pad a batch of SubsetFilingDataset items and build its forest plan.
    Returns numpy arrays only (cols, vals, pad, y, n) plus the plan."""
    B = len(items)
    L = max(it["n"] for it in items)
    cols = np.zeros((B, L), dtype=np.int64)
    vals = np.zeros((B, L), dtype=np.float32)
    pad = np.ones((B, L), dtype=bool)
    for b, it in enumerate(items):
        k = it["n"]
        cols[b, :k] = it["cols"]
        vals[b, :k] = it["vals"]
        pad[b, :k] = False
    plan = forest_level_plan_np([it["trie"] for it in items], [b * L for b in range(B)])
    return {"idx": [it["i"] for it in items], "cols": cols, "vals": vals, "pad": pad,
            "y": np.asarray([it["y"] for it in items], dtype=np.float32),
            "n": [it["n"] for it in items], "plan": plan}


def plan_to_device(plan, device):
    """Numpy forest plan -> the tensor plan subset_attention_levels expects
    (None when the batch has no subsets). Every level array and inv go to the
    device in one copy; each level's tensors are slices of that buffer, with
    every slice starting on a 16 byte boundary."""
    if not plan["levels"]:
        return None
    parts, spans, off = [], [], 0

    def put(a):
        nonlocal off
        a = np.asarray(a, dtype=np.int64)
        if off % 2:                                                    # keep every slice 16 byte aligned
            parts.append(np.zeros(1, dtype=np.int64)); off += 1
        parts.append(a)
        s = (off, off + len(a)); off += len(a)
        return s

    for lv in plan["levels"]:
        spans.append({k: (None if lv[k] is None else put(lv[k])) for k in ("pos", "parent_local", "end_local")})
    inv_span = put(plan["inv"])
    flat = torch.as_tensor(np.concatenate(parts), device=device)
    levels = [{k: (None if s is None else flat[s[0]:s[1]]) for k, s in sp.items()} for sp in spans]
    return {"levels": levels, "inv": flat[inv_span[0]:inv_span[1]]}


def batch_forward(enc, stack, name_table_t, batch, device, dtype):
    """One batch through the encoder (once per filing), the KV caches, and the
    interaction stack. name_table_t: the concept name vector table as a tensor
    on `device`. Subsets use subset_attention_levels_exact_bwd (exact reverse
    trie walk backward, nothing saved per trie node). Returns (pred_full [B],
    pred_subs [P]) with pred_subs in forest order (filing 0's subsets first)."""
    cols = torch.as_tensor(batch["cols"], device=device)
    vals = torch.as_tensor(batch["vals"], device=device).to(dtype)
    pad = torch.as_tensor(batch["pad"], device=device)
    kv = encoder_layer_kv(enc, name_table_t[cols], vals, pad)
    B, L = pad.shape
    z, V = stack.scores_values([(K.reshape(B * L, *K.shape[2:]), Vv.reshape(B * L, *Vv.shape[2:]))
                                for K, Vv in kv])
    C, dh = V.shape[1], V.shape[2]
    w = torch.softmax(z.view(B, L, C).masked_fill(pad[..., None], float("-inf")), dim=1)
    full = torch.einsum("blc,blcd->bcd", w, V.view(B, L, C, dh))
    pred_full = stack.predict(full)
    plan = plan_to_device(batch["plan"], device)
    pred_subs = (stack.predict(subset_attention_levels_exact_bwd(z, V, plan)) if plan is not None
                 else pred_full.new_zeros(0))
    return pred_full, pred_subs


def batch_ret_loss(pred_full, pred_subs, counts, y, delta=1.0):
    """Mean over filings (equal weight) of the per filing return loss:
    0.5 x Huber(full set) + 0.5 x mean Huber over that filing's subsets,
    or the full set loss alone when it has no subsets."""
    l_full = F.huber_loss(pred_full, y, delta=delta, reduction="none")
    if torch.is_tensor(counts):                  # prebuilt long tensor on y.device, summing to pred_subs.shape[0]; no host to device copy (graph capture)
        counts_t = counts
    else:
        assert sum(int(c) for c in counts) == pred_subs.shape[0], "counts must sum to the number of subset predictions"
        counts_t = torch.as_tensor(counts, device=y.device, dtype=torch.long)
    if pred_subs.numel() == 0:
        return l_full.mean()
    fid = torch.repeat_interleave(torch.arange(len(counts), device=y.device), counts_t, output_size=pred_subs.shape[0])
    l_sub = F.huber_loss(pred_subs, y[fid], delta=delta, reduction="none")
    mean_sub = (torch.zeros(len(counts), dtype=torch.float64, device=y.device).index_add(0, fid, l_sub.double())
                / counts_t.clamp(min=1).to(torch.float64)).to(l_full.dtype)
    per = torch.where(counts_t > 0, 0.5 * l_full + 0.5 * mean_sub, l_full)
    return per.mean()


def combine_predictions(pred_full, pred_subs, counts):
    """Evaluation prediction per filing, with the loss weights: 0.5 x full
    set prediction + 0.5 x mean subset prediction, or the full set
    prediction alone when the filing has no subsets. Returns [B]."""
    counts_t = torch.as_tensor(counts, device=pred_full.device, dtype=torch.long)
    if pred_subs.numel() == 0:
        return pred_full
    fid = torch.repeat_interleave(torch.arange(len(counts), device=pred_full.device), counts_t)
    mean_sub = (torch.zeros_like(pred_full).index_add(0, fid, pred_subs)
                / counts_t.clamp(min=1).to(pred_full.dtype))
    return torch.where(counts_t > 0, 0.5 * pred_full + 0.5 * mean_sub, pred_full)