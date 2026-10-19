"""
MC2 pilot data items for DataLoader workers (Windows spawns workers by importing this module).

PilotDataset items are addressed (i, epoch, train) or (i, epoch, train, targets):
  explicit targets (column schedule): the given local positions are the targets.
  train = 1, no explicit targets: ceil(rate x n) facts drawn by a generator seeded (seed, epoch, i, 7), the SAME
             draw for every arm (occurrence schedule).
  train = 0: every fact is a target (held out leave one out scoring); no subsets.
W training items carry the powerset_runs subset plan for the filing (keyed by filing and epoch); T training items
carry one exhaustive trie per target over its ranked context (template trie relabeled). Contexts for T and T0
come from MC2b (top N positions per fact, canonical order, -1 padded).

column_epoch_plan: shuffle concepts, shuffle each concept's training occurrences, partition into batches of at
most `batch`; every training occurrence appears exactly once per epoch; the plan does not depend on the arm.
"""
import math

import numpy as np
from torch.utils.data import Dataset

from set_attention_cache import build_prefix_trie_canonical, sample_subsets_powerset_runs
from masked_concept_model import mc_plan

_TEMPLATES = {}


def template_trie(N):
    """Exhaustive canonical trie over local positions 0..N-1, built once per N."""
    if N not in _TEMPLATES:
        subs = []

        def rec(prefix, start):
            for j in range(start, N):
                s = prefix + (j,)
                subs.append(s)
                rec(s, j + 1)

        rec((), 0)
        _TEMPLATES[N] = build_prefix_trie_canonical(subs)
    return _TEMPLATES[N]


def relabel(tr, ctx):
    """Template trie over 0..N-1 mapped onto sorted filing positions ctx (monotone, so canonical order holds)."""
    return {"pos": np.asarray(ctx, dtype=np.int64)[tr["pos"]], "parent": tr["parent"], "depth": tr["depth"],
            "terminals": tr["terminals"]}


def draw_targets(i, epoch, n, seed, rate):
    rng = np.random.default_rng((seed, int(epoch), int(i), 7))
    k = max(1, math.ceil(rate * n))
    return np.sort(rng.choice(n, k, replace=False))


def column_epoch_plan(occ_concept, epoch, seed, batch):
    """occ_concept: concept of every training occurrence (index = occurrence id). Returns a list of
    (concept, occurrence ids) batches: concepts in a seeded random order, each concept's occurrences in a seeded
    random order, partitioned into batches of at most `batch`."""
    rng = np.random.default_rng((seed, int(epoch), 21))
    occ_concept = np.asarray(occ_concept)
    sort_idx = np.argsort(occ_concept, kind="stable")
    sorted_c = occ_concept[sort_idx]
    concepts = np.unique(sorted_c)
    starts = np.searchsorted(sorted_c, concepts, side="left")
    ends = np.searchsorted(sorted_c, concepts, side="right")
    out = []
    for k in rng.permutation(len(concepts)):
        occ = sort_idx[starts[k]:ends[k]]
        occ = occ[rng.permutation(len(occ))]
        for s in range(0, len(occ), batch):
            out.append((int(concepts[k]), occ[s:s + batch]))
    return out


class PilotDataset(Dataset):
    def __init__(self, indptr, indices, data, file_names, ctx, kind, *, M, run_factor, seed, rate):
        if kind not in ("S0", "W0", "W", "T0", "T"):
            raise ValueError(f"unknown arm {kind!r}")
        self.indptr = np.asarray(indptr, dtype=np.int64)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.data = np.asarray(data, dtype=np.float32)
        self.file_names = list(file_names)
        self.ctx = ctx
        self.kind, self.M, self.run_factor, self.seed, self.rate = kind, int(M), int(run_factor), seed, rate

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, key):
        i, epoch, train = key[0], key[1], key[2]
        explicit = key[3] if len(key) > 3 else None
        lo, hi = int(self.indptr[i]), int(self.indptr[i + 1])
        n = hi - lo
        if explicit is not None:
            tgt = np.asarray(sorted(int(t) for t in explicit), dtype=np.int64)
            if len(tgt) == 0 or tgt.min() < 0 or tgt.max() >= n:
                raise ValueError(f"explicit targets out of range for filing {i}")
        elif train:
            tgt = draw_targets(i, epoch, n, self.seed, self.rate)
        else:
            tgt = np.arange(n)
        item = {"i": int(i), "cols": self.indices[lo:hi], "vals": self.data[lo:hi], "n": n, "tgt": tgt}
        if self.kind in ("T0", "T"):
            rows = self.ctx[lo + tgt]
            item["ctx"] = [r[r >= 0].astype(np.int64) for r in rows]
            if self.kind == "T" and train:
                item["tries"] = [relabel(template_trie(len(c)), c) for c in item["ctx"]]
        if self.kind == "W" and train:
            subs = sample_subsets_powerset_runs(n, self.file_names[i], self.M, self.run_factor * n,
                                                seed=self.seed, epoch=int(epoch))
            item["P"] = len(subs)
            item["trie"] = build_prefix_trie_canonical(subs)
            item["plan"] = mc_plan([item["trie"]])
        return item


def collate_list(items):
    return items