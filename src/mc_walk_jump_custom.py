"""
Pointer jumping walk with a closed form custom backward (v2 exploratory; not part of release v1.0).

Forward: masked_concept_model.mc_walk_jump, unchanged, run without autograd, owned mode only.
Backward: the exact gradient of pooled and L, by the method of set_attention_model._SubsetAttentionLevelsFn
(one reverse level walk with a stable reference per node), adapted to per head scores, owner channels, excluded
targets and the L output. For a usable subset P with log normalizer L_P and output o_P, every valid node i on its
path has weight w_Pi = exp(z_i - L_P). With incoming gradients g_P (pooled) and l_P (L):
    dV_i = sum_P w_Pi g_P
    dz_i = sum_P w_Pi (g_P . V_i - g_P . o_P + l_P)
The reverse walk carries, per node t, mu_t = min of L_P over the usable subsets ending in its subtree, and
GH_t = [G_t | H_t] with
    G_t = sum exp(mu_t - L_P) g_P,    H_t = sum exp(mu_t - L_P) (g_P . o_P - l_P)
(every exponent <= 0). A valid node t holding fact i for owner c contributes e = exp(z_i - mu_t) <= 1:
dV[i] += e G_t and dz[i, c] += e (V_i . G_t - H_t). These node contributions are computed for all nodes in one
batched pass after the walk, when every mu and GH is final. Excluded nodes contribute nothing. Unusable subsets enter
with zero weight: their outputs and incoming gradients are replaced with zero before any arithmetic.
Retained for backward: z, V, pooled, L and usable, plus node sized index tensors (position, owner, valid) and the
level schedule. No intermediate attention states from the pointer jumping rounds are retained. Backward allocates
node sized buffers mu [T, H] and GH [T, H, dh + 1], and node sized temporaries in the batched node pass.
"""
import numpy as np
import torch

import masked_concept_model as mcm


def level_schedule(plan, depth, dev):
    """Per depth level: the nodes (global index) and their parents, and the terminal rows ending at that level with
    their nodes. depth is each node's depth in plan order (the tries' depth arrays concatenated in trie order).
    Checked: depth matches n_levels, roots are exactly the depth 0 nodes, every nonroot's parent is one level up,
    and terminal nodes are unique."""
    depth = np.asarray(depth, dtype=np.int64)
    parent = np.asarray(plan["parent"], dtype=np.int64)
    term = np.asarray(plan["term"], dtype=np.int64)
    if len(depth) != len(parent):
        raise ValueError("level_schedule: depth length does not match the plan's nodes")
    D = int(depth.max()) + 1
    if D != int(plan["n_levels"]):
        raise ValueError("level_schedule: depth disagrees with the plan's n_levels")
    if int(depth.max()) > np.iinfo(np.int16).max:
        raise ValueError("level_schedule: depth exceeds the int16 range of the sort keys")
    depth16 = depth.astype(np.int16)                                   # narrow sort keys; performance measured below
    root = parent < 0
    if ((depth == 0) != root).any():
        raise ValueError("level_schedule: roots and depth 0 disagree")
    nr = ~root
    if (depth[parent[nr]] != depth[nr] - 1).any():
        raise ValueError("level_schedule: a nonroot node's parent is not one level up")
    if len(np.unique(term)) != len(term):
        raise ValueError("level_schedule: terminal node indices are not unique")
    nodes = np.argsort(depth16, kind="stable")
    n_cnt = np.bincount(depth, minlength=D)
    term_depth = depth[term]
    t_rows = np.argsort(depth16[term], kind="stable")
    t_cnt = np.bincount(term_depth, minlength=D)
    T, P = len(nodes), len(term)
    packed = torch.as_tensor(np.concatenate([nodes, parent[nodes], t_rows, term[t_rows]]), device=dev)   # one transfer
    nodes_t, par_t, rows_t, tnode_t = packed[:T], packed[T:2 * T], packed[2 * T:2 * T + P], packed[2 * T + P:]
    n_off = np.concatenate([[0], np.cumsum(n_cnt)])
    t_off = np.concatenate([[0], np.cumsum(t_cnt)])
    levels = []
    for d in range(D):
        a, b, c, e = n_off[d], n_off[d + 1], t_off[d], t_off[d + 1]
        has_t = e > c
        levels.append({"nodes": nodes_t[a:b], "parents": par_t[a:b] if d > 0 else None,
                       "rows": rows_t[c:e] if has_t else None, "tnodes": tnode_t[c:e] if has_t else None})
    return {"levels": levels, "rows": rows_t, "tnodes": tnode_t}


class _WalkJumpFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, V, plan, sched, owner_target_pos):
        pooled, L, k, usable = mcm.mc_walk_jump(z, V, plan, owned=True, owner_target_pos=owner_target_pos)
        dev = z.device
        pos = torch.as_tensor(plan["pos"], device=dev)
        own = torch.as_tensor(plan["owner"], device=dev)
        otp = torch.as_tensor(np.asarray(owner_target_pos), device=dev)
        ctx.save_for_backward(z, V, pooled, L, usable)
        ctx.pos, ctx.own, ctx.valid, ctx.sched = pos, own, pos != otp[own], sched
        ctx.mark_non_differentiable(k, usable)
        return pooled, L, k, usable

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, g_pool, g_L, _g_k, _g_use):
        z, V, pooled, L, usable = ctx.saved_tensors
        pos, own, valid, sched = ctx.pos, ctx.own, ctx.valid, ctx.sched
        N, C, H = z.shape
        dh = V.shape[-1]
        T, dt = pos.numel(), z.dtype
        use = usable[:, 0]                                                            # [P]
        u2, u3 = use[:, None], use[:, None, None]
        o = torch.where(u3, pooled[:, 0], 0.0)                                         # [P, H, dh]
        gP = torch.where(u3, g_pool[:, 0], 0.0) if g_pool is not None else torch.zeros_like(o)
        gL = torch.where(u2, g_L[:, 0], 0.0) if g_L is not None else torch.zeros_like(L[:, 0])
        Lt = torch.where(u2, L[:, 0], float("inf"))                                     # unusable: weight zero
        termv = torch.cat((gP, ((gP * o).sum(-1) - gL).unsqueeze(-1)), -1)            # [P, H, dh + 1]
        flat = pos * C + own
        zf = z.reshape(-1, H)
        dz = torch.zeros((N, C, H), dtype=dt, device=z.device)
        dV = torch.zeros(V.shape, dtype=dt, device=V.device)
        dzf = dz.view(-1, H)
        mu = z.new_full((T, H), float("inf"))
        mu.index_put_((sched["tnodes"],), Lt.index_select(0, sched["rows"]))         # own terminal first (unique)
        GH = z.new_zeros((T, H, dh + 1))
        zero = z.new_zeros(())
        levels = sched["levels"]
        for d in range(len(levels) - 1, -1, -1):
            lv = levels[d]
            if lv["rows"] is not None:                                                # terminals ending here
                tn, Lr = lv["tnodes"], Lt.index_select(0, lv["rows"])
                ft = torch.where(torch.isfinite(Lr), (mu.index_select(0, tn) - Lr).exp(), zero)
                GH.index_add_(0, tn, termv.index_select(0, lv["rows"]) * ft.unsqueeze(-1))
            nd = lv["nodes"]
            md = mu.index_select(0, nd)                                               # final mu of this level
            if d > 0:                                                                 # push to parents                                                                 # push to parents
                par = lv["parents"]
                mu.scatter_reduce_(0, par[:, None].expand(-1, H), md, reduce="amin", include_self=True)
                f = torch.where(torch.isfinite(md), (mu.index_select(0, par) - md).exp(), zero)
                GH.index_add_(0, par, GH.index_select(0, nd) * f.unsqueeze(-1))
        ok = valid[:, None] & torch.isfinite(mu)                                       # every node: mu and GH are final
        e = torch.where(ok, (zf.index_select(0, flat) - mu).exp(), zero)              # [T, H], <= 1
        G, Hh = GH.split((dh, 1), dim=-1)
        Vn = V.index_select(0, pos)
        dV.index_add_(0, pos, G * e.unsqueeze(-1))
        dzf.index_add_(0, flat, ((Vn * G).sum(-1) - Hh[..., 0]) * e)
        return dz, dV, None, None, None


def mc_walk_jump_custom(z, V, plan, depth, owner_target_pos):
    """Same outputs as mc_walk_jump(z, V, plan, owned=True, owner_target_pos=...), with the closed form backward.
    Builds the level schedule on every call."""
    sched = level_schedule(plan, depth, z.device)
    return _WalkJumpFn.apply(z, V, plan, sched, owner_target_pos)