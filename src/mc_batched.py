"""
Batched column step: one logical batch (one concept's occurrences across up to 32 filings) computed in one pass.
Fact rows of every filing are concatenated (the fact encoder is per fact, so rows stay independent). Each target is
one channel and, for W and T, owns one tree whose positions are global (filing offset + local position): the tree
reads only its own filing's facts and only its owner's query. W excludes each owner's own target by its global
position; T contexts never contain the target. Full contexts are dense masked attention restricted to each
target's own filing. The loss is the mean over all of the batch's targets: one backward, one update.
walk="levels" uses the certified level walk (mc_plan, mc_walk); walk="jump" uses pointer jumping over the same
trees (mc_plan_jump, mc_walk_jump).
"""
import numpy as np
import torch
import torch.nn.functional as F

import masked_concept_model as mcm
from set_attention_model import signed_log

WALKS = {"levels": (mcm.mc_plan, mcm.mc_walk), "jump": (mcm.mc_plan_jump, mcm.mc_walk_jump)}


def _shift(tr, off):
    return {"pos": np.asarray(tr["pos"], dtype=np.int64) + int(off), "parent": tr["parent"], "depth": tr["depth"],
            "terminals": tr["terminals"]}


def batch_loss(model, kind, items, NT, dev, walk="levels"):
    """Returns (loss, exposures, pred_full [C], pred_sub_usable [U]) with channels in item order, then each
    item's target order; pred_sub_usable in channel order, then subset order within each channel's tree."""
    if kind not in ("W0", "W", "T0", "T"):
        raise ValueError(f"unknown arm {kind!r}")
    if walk not in WALKS:
        raise ValueError(f"unknown walk {walk!r}")
    plan_fn, walk_fn = WALKS[walk]
    ns = np.array([it["n"] for it in items], dtype=np.int64)
    offs = np.concatenate([[0], np.cumsum(ns)[:-1]])
    cols = torch.as_tensor(np.concatenate([it["cols"] for it in items]), device=dev)
    vals = torch.as_tensor(np.concatenate([it["vals"] for it in items]), device=dev).to(NT.dtype)
    nv = NT[cols]
    ch_item = np.concatenate([np.full(len(it["tgt"]), b) for b, it in enumerate(items)])
    ch_j = np.concatenate([np.arange(len(it["tgt"])) for it in items])
    ch_local = np.concatenate([np.asarray(it["tgt"], dtype=np.int64) for it in items])
    ch_global = offs[ch_item] + ch_local
    C, Ntot = len(ch_global), int(ns.sum())
    tg = torch.as_tensor(ch_global, device=dev)
    z, V, e = model.scores(nv, vals, nv[tg])                                  # z [Ntot, C, H]
    y = signed_log(vals)[tg]
    fact_item = torch.as_tensor(np.repeat(np.arange(len(items)), ns), device=dev)
    ci = torch.as_tensor(ch_item, device=dev)
    if kind in ("W0", "W"):
        member = fact_item[None, :] == ci[:, None]
        member[torch.arange(C, device=dev), tg] = False
    else:
        member = torch.zeros(C, Ntot, dtype=torch.bool, device=dev)
        for c in range(C):
            b, j = int(ch_item[c]), int(ch_j[c])
            member[c, torch.as_tensor(offs[b] + items[b]["ctx"][j], device=dev)] = True
    fp, fL, fk, fu = mcm.mc_masked_attention(z, V, member, torch.arange(C, device=dev))
    if not bool(fu.all()):
        raise ValueError("a target has an empty full context")
    pred_full = model.predict(fp, fL, fk, e)
    if kind in ("W0", "T0"):
        return F.huber_loss(pred_full, y), C, pred_full, pred_full.new_zeros(0)
    if kind == "W":
        tries = [_shift(items[int(ch_item[c])]["trie"], offs[int(ch_item[c])]) for c in range(C)]
        plan = plan_fn(tries)
        pool, L, k, use = walk_fn(z, V, plan, owned=True, owner_target_pos=ch_global)
        wd = torch.as_tensor([mcm.w_subset_denominator(int(ns[ch_item[c]]), int(items[int(ch_item[c])]["P"]))
                              for c in range(C)], device=dev, dtype=pred_full.dtype)
    else:
        tries = [_shift(items[int(ch_item[c])]["tries"][int(ch_j[c])], offs[int(ch_item[c])]) for c in range(C)]
        plan = plan_fn(tries)
        pool, L, k, use = walk_fn(z, V, plan, owned=True)
        wd = None
    owner = torch.as_tensor(plan["term_owner"], device=dev)
    pred = model.predict(pool[:, 0], L[:, 0], k[:, 0], e[owner])
    loss = mcm.mc_target_loss(pred_full, y, fu, pred, owner, k[:, 0], use[:, 0], mode=kind, w_denom=wd)
    return loss, C + int(use[:, 0].sum()), pred_full, pred[use[:, 0]]