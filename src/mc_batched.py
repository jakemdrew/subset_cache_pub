"""
Batched column step: one logical batch (one concept's occurrences across up to 32 filings) computed in one pass.
Fact rows of every filing are concatenated (the fact encoder is per fact, so rows stay independent). Each target is
one channel and, for W and T, owns one tree whose positions are global (filing offset + local position): the tree
reads only its own filing's facts and only its owner's query. W excludes each owner's own target by its global
position; T contexts never contain the target. Full contexts are dense masked attention restricted to each
target's own filing. The loss is the mean over all of the batch's targets: one backward, one update.
walk="levels" uses the certified level walk (mc_plan, mc_walk); walk="jump" uses pointer jumping over the same
trees (mc_plan_jump, mc_walk_jump); both are release v1.0.
walk="cache" uses the shared cache, subset_attention_levels_exact_bwd with return_lse=True, the implementation the
Section 7 benchmarks measure. Each target with subsets gets its own block of rows: W, its filing's rows; T, its
ranked context's rows. Block rows carry only that target's scores (one channel, H heads) and the shared values:
    Zp = z[row_idx, chan_idx, :]  [R, H]        Vp = V[row_idx]  [R, H, dh]
so the cache's channels are the H heads and every tree reads only its own block. Advanced indexing returns the
gradients to dz[row_idx, chan_idx, :] and dV[row_idx] by scatter add. W items need tries_excl (PilotDataset with
structures "cache" or "both"): per target, the drawn subsets that do not contain it, which are exactly the subsets
the v1.0 walks count as usable; the Horvitz Thompson denominator keeps the drawn P. T uses the template trie over
0..r-1 directly. Each subset's owner is read from the forest plan's per tree counts, and k, its effective count, is
its size: the terminal's depth plus one. A target without subsets contributes the subset term zero, as in v1.0.
"""
import numpy as np
import torch
import torch.nn.functional as F

import masked_concept_model as mcm
import mc_pilot_data as mpd
from set_attention_model import signed_log, subset_attention_levels_exact_bwd
from set_attention_data import forest_level_plan_np, plan_to_device

WALKS = {"levels": (mcm.mc_plan, mcm.mc_walk), "jump": (mcm.mc_plan_jump, mcm.mc_walk_jump)}
ALL_WALKS = tuple(WALKS) + ("cache",)


def _shift(tr, off):
    return {"pos": np.asarray(tr["pos"], dtype=np.int64) + int(off), "parent": tr["parent"], "depth": tr["depth"],
            "terminals": tr["terminals"]}


def cache_subsets(z, V, blocks, dev):
    """Shared cache over one block per channel. blocks: per channel c, (row_idx global rows, trie over 0..len-1) or
    None. Returns (out [P, H, dh], L [P, H], owner [P] long, k [P]) in forest order (channel order, then each trie's
    subset order), or None when no channel has subsets. The order is checked by count here and by identity in the
    agreement gate."""
    rows, chans, tries, offs, owners = [], [], [], [], []
    base = 0
    for c, blk in enumerate(blocks):
        if blk is None:
            continue
        r, tr = blk
        pos = np.asarray(tr["pos"], dtype=np.int64)
        if len(pos) == 0 or len(tr["terminals"]) == 0:
            raise ValueError(f"cache_subsets: channel {c} has an empty trie; pass None for a channel without subsets")
        if int(pos.min()) < 0 or int(pos.max()) >= len(r):
            raise ValueError(f"cache_subsets: channel {c}'s trie reads positions outside its block of {len(r)} rows")
        rows.append(np.asarray(r, dtype=np.int64))
        chans.append(np.full(len(r), c, dtype=np.int64))
        tries.append(tr)
        offs.append(base)
        owners.append(c)
        base += len(r)
    if not tries:
        return None
    npl = forest_level_plan_np(tries, offs)
    counts = np.asarray(npl["counts"], dtype=np.int64)
    sizes = np.concatenate([np.asarray(tr["depth"], dtype=np.int64)[np.asarray(tr["terminals"], dtype=np.int64)] + 1
                            for tr in tries])
    if not np.array_equal(counts, [len(tr["terminals"]) for tr in tries]) or counts.sum() != len(sizes):
        raise ValueError("cache_subsets: forest counts disagree with the tries")
    plan = plan_to_device(npl, dev)
    if int(plan["inv"].numel()) != int(counts.sum()):
        raise ValueError("cache_subsets: plan subset count disagrees with the tries")
    row_idx = torch.as_tensor(np.concatenate(rows), device=dev)
    chan_idx = torch.as_tensor(np.concatenate(chans), device=dev)
    Zp = z[row_idx, chan_idx, :]                                                  # [R, H]
    Vp = V[row_idx]                                                               # [R, H, dh]
    out, L = subset_attention_levels_exact_bwd(Zp, Vp, plan, return_lse=True)    # [P, H, dh], [P, H]
    owner = torch.as_tensor(np.repeat(np.asarray(owners, dtype=np.int64), counts), device=dev)
    k = torch.as_tensor(sizes, device=dev, dtype=z.dtype)
    return out, L, owner, k


def batch_loss(model, kind, items, NT, dev, walk="levels"):
    """Returns (loss, exposures, pred_full [C], pred_sub_usable [U]) with channels in item order, then each
    item's target order; pred_sub_usable in channel order, then subset order within each channel's tree."""
    if kind not in ("W0", "W", "T0", "T"):
        raise ValueError(f"unknown arm {kind!r}")
    if walk not in ALL_WALKS:
        raise ValueError(f"unknown walk {walk!r}")
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
    wd = None
    if kind == "W":
        wd = torch.as_tensor([mcm.w_subset_denominator(int(ns[ch_item[c]]), int(items[int(ch_item[c])]["P"]))
                              for c in range(C)], device=dev, dtype=pred_full.dtype)
    if walk == "cache":
        blocks = []
        for c in range(C):
            b, j = int(ch_item[c]), int(ch_j[c])
            if kind == "W":
                tr = items[b]["tries_excl"][j]
                blocks.append(None if tr is None else (offs[b] + np.arange(ns[b]), tr))
            else:
                ctx = np.asarray(items[b]["ctx"][j], dtype=np.int64)
                blocks.append((offs[b] + ctx, mpd.template_trie(len(ctx))))
        res = cache_subsets(z, V, blocks, dev)
        if res is None:
            empty = pred_full.new_zeros(0)
            none_l = torch.zeros(0, dtype=torch.long, device=dev)
            loss = mcm.mc_target_loss(pred_full, y, fu, empty, none_l, empty, torch.zeros(0, dtype=torch.bool, device=dev),
                                      mode=kind, w_denom=wd)
            return loss, C, pred_full, empty
        out, L, owner, k = res
        pred = model.predict(out, L, k, e[owner])
        use = torch.ones_like(owner, dtype=torch.bool)
        loss = mcm.mc_target_loss(pred_full, y, fu, pred, owner, k, use, mode=kind, w_denom=wd)
        return loss, C + int(owner.numel()), pred_full, pred
    plan_fn, walk_fn = WALKS[walk]
    if kind == "W":
        tries = [_shift(items[int(ch_item[c])]["trie"], offs[int(ch_item[c])]) for c in range(C)]
        plan = plan_fn(tries)
        pool, L, k, use = walk_fn(z, V, plan, owned=True, owner_target_pos=ch_global)
    else:
        tries = [_shift(items[int(ch_item[c])]["tries"][int(ch_j[c])], offs[int(ch_item[c])]) for c in range(C)]
        plan = plan_fn(tries)
        pool, L, k, use = walk_fn(z, V, plan, owned=True)
    owner = torch.as_tensor(plan["term_owner"], device=dev)
    pred = model.predict(pool[:, 0], L[:, 0], k[:, 0], e[owner])
    loss = mcm.mc_target_loss(pred_full, y, fu, pred, owner, k[:, 0], use[:, 0], mode=kind, w_denom=wd)
    return loss, C + int(use[:, 0].sum()), pred_full, pred[use[:, 0]]