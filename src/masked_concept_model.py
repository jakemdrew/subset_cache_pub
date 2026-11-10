"""
Masked concept subset attention, version 1 (reference implementation for MC1 certification).

Each fact is encoded on its own (no cross fact mixing). Each target fact's concept becomes a fixed
query (name vector plus a learned mask vector; never its value). One attention step pools a target's
supporting context, either a subset taken from a prefix trie (cached) or a dense context (direct).
A readout maps the pooled vector, the log channel m + log(s) per head, the effective count and the
target's concept embedding to a predicted signed log value.

Two designs share this module:
  W (wide, shared trie)  one trie over every fact of the filing; every node carries all C channels.
                         Channel c excludes its own target fact through a validity mask: an excluded
                         node passes its parent's state through unchanged (its own score and value get
                         no gradient; gradients still flow through the parent state), and any path that
                         contains the target is unusable for c, so each supporting context comes from
                         exactly one path.
  T (top N, owned tries) one trie per target over that target's context facts; each node carries
                         only its owner's channel; the context never contains the target.
Owned tries also serve batched W (one tree per target across several filings, global positions): pass
owner_target_pos and each owner excludes its own target with the same pass through and unusable path rules.

Every exponential is taken of a masked, finite, nonpositive argument, so extreme finite scores
cannot overflow in an unselected branch and poison gradients.
Gradients come from autograd through the level walk (exact by construction). This graph is a
correctness reference; its memory use does not predict the cost of a later custom backward.
"""
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from set_attention_model import signed_log

NAME_DIM = 384


@dataclass
class MCConfig:
    dim: int = 256
    n_heads: int = 4
    enc_layers: int = 2
    readout_hidden: int = 256
    huber_delta: float = 1.0


class FactEncoder(nn.Module):
    """One fact at a time: adapter(name) + g(value) * w_val + b_val, residual MLP blocks, then keys and
    values per head. Every operation is row wise, so a fact's keys and values never depend on other facts."""

    def __init__(self, cfg):
        super().__init__()
        d = cfg.dim
        self.cfg = cfg
        self.adapter = nn.Linear(NAME_DIM, d, bias=False)
        self.w_val = nn.Parameter(torch.randn(d) * 0.02)
        self.b_val = nn.Parameter(torch.zeros(d))
        self.blocks = nn.ModuleList(nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 4 * d), nn.GELU(),
                                                  nn.Linear(4 * d, d)) for _ in range(cfg.enc_layers))
        self.ln = nn.LayerNorm(d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        nn.init.normal_(self.adapter.weight, std=0.02)

    def forward(self, name_vecs, values):
        x = self.adapter(name_vecs) + signed_log(values)[:, None] * self.w_val + self.b_val
        for blk in self.blocks:
            x = x + blk(x)
        h = self.ln(x)
        n, H = h.shape[0], self.cfg.n_heads
        return self.k(h).view(n, H, -1), self.v(h).view(n, H, -1)


class ConceptQuery(nn.Module):
    """Target concept identity only: e = adapter(name) + mask_vec; q = Wq(LN(e)) per head."""

    def __init__(self, cfg):
        super().__init__()
        d = cfg.dim
        self.cfg = cfg
        self.adapter = nn.Linear(NAME_DIM, d, bias=False)
        self.mask_vec = nn.Parameter(torch.randn(d) * 0.02)
        self.ln = nn.LayerNorm(d)
        self.q = nn.Linear(d, d)
        nn.init.normal_(self.adapter.weight, std=0.02)

    def forward(self, target_name_vecs):
        e = self.adapter(target_name_vecs) + self.mask_vec
        C, H = e.shape[0], self.cfg.n_heads
        return self.q(self.ln(e)).view(C, H, -1), e


class MaskedConceptModel(nn.Module):
    def __init__(self, cfg: MCConfig):
        super().__init__()
        self.cfg = cfg
        d, H = cfg.dim, cfg.n_heads
        self.encoder = FactEncoder(cfg)
        self.query = ConceptQuery(cfg)
        self.readout = nn.Sequential(nn.LayerNorm(d + H + 1 + d), nn.Linear(d + H + 1 + d, cfg.readout_hidden),
                                     nn.GELU(), nn.Linear(cfg.readout_hidden, cfg.readout_hidden), nn.GELU(),
                                     nn.Linear(cfg.readout_hidden, 1))

    def scores(self, name_vecs, values, target_name_vecs):
        """Returns z [n, C, H] (fact i, channel c, head h), V [n, H, dh], e [C, d]."""
        K, V = self.encoder(name_vecs, values)
        q, e = self.query(target_name_vecs)
        z = torch.einsum("ihd,chd->ich", K, q) / math.sqrt(K.shape[-1])
        return z, V, e

    def predict(self, pooled, L, k, e_rows):
        """pooled [R, H, dh], L [R, H], k [R] effective counts, e_rows [R, d] -> predictions [R]."""
        feats = torch.cat([pooled.reshape(pooled.shape[0], -1), L, torch.log1p(k.to(L.dtype))[:, None], e_rows], 1)
        return self.readout(feats).squeeze(-1)


# ----- plans: merged tries with owners, level by level -----
def mc_plan(tries):
    """tries: list of build_prefix_trie_canonical dicts whose positions are FILING positions (no offsets).
    Trie b owns channel b in owned mode; in shared mode pass one trie. Returns levels (pos, parent_local,
    owner, end_local), inv (path order), term_owner and counts per trie. Empty input or an empty trie is
    rejected: a target without subsets is handled by passing no plan to the loss, not an empty plan."""
    if len(tries) == 0:
        raise ValueError("mc_plan needs at least one trie")
    if any(tr is None or len(tr["terminals"]) == 0 for tr in tries):
        raise ValueError("mc_plan: every trie needs at least one subset")
    pos_l, par_l, dep_l, term_l, own_l, counts = [], [], [], [], [], []
    base = 0
    for b, tr in enumerate(tries):
        T = len(tr["pos"])
        pos_l.append(np.asarray(tr["pos"], dtype=np.int64))
        p = np.asarray(tr["parent"], dtype=np.int64).copy()
        p[p >= 0] += base
        par_l.append(p)
        dep_l.append(np.asarray(tr["depth"], dtype=np.int64))
        term_l.append(np.asarray(tr["terminals"], dtype=np.int64) + base)
        own_l.append(np.full(T, b, dtype=np.int64))
        counts.append(len(tr["terminals"]))
        base += T
    pos, parent, depth = np.concatenate(pos_l), np.concatenate(par_l), np.concatenate(dep_l)
    term, owner = np.concatenate(term_l), np.concatenate(own_l)
    local = np.empty(len(pos), dtype=np.int64)
    levels, order = [], []
    term_depth = depth[term]
    for d in range(int(depth.max()) + 1):
        idx = np.flatnonzero(depth == d)
        local[idx] = np.arange(len(idx))
        ends = np.flatnonzero(term_depth == d)
        levels.append({"pos": pos[idx], "owner": owner[idx],
                       "parent_local": local[parent[idx]] if d > 0 else None,
                       "end_local": local[term[ends]] if len(ends) else None})
        order.append(ends)
    return {"levels": levels, "inv": np.argsort(np.concatenate(order)), "term_owner": owner[term],
            "counts": counts}


def mc_walk(z, V, plan, target_pos=None, owned=False, owner_target_pos=None):
    """Cached attention over every terminal of the plan.
    shared (owned False): states [nodes, C, H]; channel c excludes fact target_pos[c] by a validity mask
    and marks any path containing it unusable. owned (True): states [nodes, 1, H] for the node's owner
    channel; exclusion only when owner_target_pos gives each owner's own (global) target position, with the same
    pass through and unusable path rules as shared mode. Returns, in path order: pooled [P, C', H, dh], L [P, C', H] (m + log s),
    k [P, C'] effective counts, usable [P, C'] (C' = C shared, 1 owned)."""
    if plan is None or len(plan["levels"]) == 0:
        raise ValueError("mc_walk needs a plan with at least one subset")
    if not owned and target_pos is None:
        raise ValueError("shared mode needs target_pos")
    dev = z.device
    tgt = None if owned else torch.as_tensor(np.asarray(target_pos), device=dev)
    otp = None if owner_target_pos is None else torch.as_tensor(np.asarray(owner_target_pos), device=dev)
    prev, outs = None, []
    for lv in plan["levels"]:
        pos = torch.as_tensor(lv["pos"], device=dev)
        if owned:
            own = torch.as_tensor(lv["owner"], device=dev)
            zn = z[pos, own][:, None]                                                # [Nd, 1, H]
            if otp is None:
                valid = torch.ones(zn.shape[:2], dtype=torch.bool, device=dev)
            else:
                valid = (pos != otp[own])[:, None]                                   # each owner excludes its own target
        else:
            zn = z[pos]                                                              # [Nd, C, H]
            valid = pos[:, None] != tgt[None, :]                                     # [Nd, C]
        Vn = V[pos][:, None]                                                         # [Nd, 1, H, dh]
        if prev is None:
            m_p = torch.full_like(zn, float("-inf"))
            s_p = torch.zeros_like(zn)
            u_p = torch.zeros(*zn.shape, V.shape[-1], dtype=zn.dtype, device=dev)
            k_p = torch.zeros(zn.shape[:2], dtype=zn.dtype, device=dev)
            bad_p = torch.zeros(zn.shape[:2], dtype=torch.bool, device=dev)
        else:
            pl = torch.as_tensor(lv["parent_local"], device=dev)
            m_p, s_p, u_p, k_p, bad_p = (t[pl] for t in prev)
        vm = valid[..., None].expand_as(zn)
        m_new = torch.where(vm, torch.maximum(m_p, zn), m_p)
        fin_p, fin_n = torch.isfinite(m_p), torch.isfinite(m_new)
        zero = torch.zeros_like(zn)
        m_p_s = torch.where(fin_p, m_p, zero)
        m_n_s = torch.where(fin_n, m_new, zero)
        # exponent arguments are masked BEFORE exp: both are finite and at most 0 wherever selected
        alpha_arg = torch.where(fin_p & fin_n, m_p_s - m_n_s, zero)
        alpha = alpha_arg.exp() * fin_p.to(zn.dtype)
        beta_arg = torch.where(vm, zn - m_n_s, zero)
        beta = beta_arg.exp() * vm.to(zn.dtype)
        s_new = alpha * s_p + beta
        u_new = alpha[..., None] * u_p + beta[..., None] * Vn
        k_new = k_p + valid.to(k_p.dtype)
        bad_new = bad_p | ~valid
        prev = (m_new, s_new, u_new, k_new, bad_new)
        if lv["end_local"] is not None:
            el = torch.as_tensor(lv["end_local"], device=dev)
            outs.append(tuple(t[el] for t in prev))
    if not outs:
        raise ValueError("mc_walk: plan has no terminals")
    inv = torch.as_tensor(plan["inv"], device=dev)
    m, s, u, k, bad = (torch.cat([o[j] for o in outs])[inv] for j in range(5))
    usable = (~bad) & (k > 0)
    um = usable[..., None].expand_as(s)
    s_safe = torch.where(um, s, torch.ones_like(s))
    m_safe = torch.where(um, m, torch.zeros_like(m))
    pooled = u / s_safe[..., None]
    L = torch.where(um, m_safe + torch.log(s_safe), torch.zeros_like(m))
    return pooled, L, k, usable


# ----- pointer jumping over the same tries (alternative schedule; mc_walk remains the reference) -----
def mc_plan_jump(tries):
    """Pointer jumping plan over the same tries as mc_plan (same input contract and rejections). Returns pos,
    owner and parent per node (parent is a global node index, -1 at a root), term (terminal node of every subset,
    in mc_walk's output order: trie 0's subsets first), term_owner, counts, n_levels (longest path in nodes),
    rounds (smallest r with 2^r >= n_levels) and jumps [rounds, T + 1]: the ancestor pointer of every node in each
    round, where index T is the identity sentinel. Checked here: roots have depth 0, every other node has depth
    one more than its parent and the same tree as its parent, and every pointer reaches the sentinel after the
    last round."""
    if len(tries) == 0:
        raise ValueError("mc_plan_jump needs at least one trie")
    if any(tr is None or len(tr["terminals"]) == 0 for tr in tries):
        raise ValueError("mc_plan_jump: every trie needs at least one subset")
    pos_l, par_l, dep_l, term_l, own_l, counts = [], [], [], [], [], []
    base = 0
    for b, tr in enumerate(tries):
        T_b = len(tr["pos"])
        pos_l.append(np.asarray(tr["pos"], dtype=np.int64))
        p = np.asarray(tr["parent"], dtype=np.int64).copy()
        p[p >= 0] += base
        par_l.append(p)
        dep_l.append(np.asarray(tr["depth"], dtype=np.int64))
        term_l.append(np.asarray(tr["terminals"], dtype=np.int64) + base)
        own_l.append(np.full(T_b, b, dtype=np.int64))
        counts.append(len(tr["terminals"]))
        base += T_b
    pos, parent, depth = np.concatenate(pos_l), np.concatenate(par_l), np.concatenate(dep_l)
    term, owner = np.concatenate(term_l), np.concatenate(own_l)
    T = len(pos)
    root = parent < 0
    nr = ~root
    if (depth[root] != 0).any() or (depth[nr] != depth[parent[nr]] + 1).any() or (owner[nr] != owner[parent[nr]]).any():
        raise ValueError("mc_plan_jump: trie parent, depth and owner disagree")
    n_levels = int(depth.max()) + 1
    rounds = (n_levels - 1).bit_length()
    J = np.append(np.where(root, T, parent), T)
    jumps = np.empty((rounds, T + 1), dtype=np.int64)
    for r in range(rounds):
        jumps[r] = J
        J = J[J]
    if not (J == T).all():
        raise ValueError("mc_plan_jump: pointers did not reach the sentinel after the last round")
    return {"pos": pos, "owner": owner, "parent": parent, "term": term, "term_owner": owner[term],
            "counts": counts, "n_levels": n_levels, "rounds": rounds, "jumps": jumps}


def _jump_merge(a, b):
    """Combine segment a (earlier on the path) with segment b (later). States are (m, s, u, k, bad); an empty
    segment is (-inf, 0, 0, 0) with any flag. Exponent arguments are masked to finite values before exp, so
    merging empty or one sided segments never forms -inf minus -inf, in forward or backward."""
    ma, sa, ua, ka, bada = a
    mb, sb, ub, kb, badb = b
    m = torch.maximum(ma, mb)
    fa, fb, fm = torch.isfinite(ma), torch.isfinite(mb), torch.isfinite(m)
    zero = torch.zeros_like(m)
    m_s = torch.where(fm, m, zero)
    wa = torch.where(fa, torch.where(fa, ma, zero) - m_s, zero).exp() * fa.to(m.dtype)
    wb = torch.where(fb, torch.where(fb, mb, zero) - m_s, zero).exp() * fb.to(m.dtype)
    return m, wa * sa + wb * sb, wa[..., None] * ua + wb[..., None] * ub, ka + kb, bada | badb


def mc_walk_jump(z, V, plan, target_pos=None, owned=False, owner_target_pos=None):
    """Same contract and outputs as mc_walk, by pointer jumping over plan = mc_plan_jump(tries).
    Each node starts as its own segment: (z, 1, V, 1, not excluded) when its fact is usable for the channel, and
    the identity (-inf, 0, 0, 0) with the excluded flag set when it is the channel's own target, so the target
    adds nothing to pooling or count while every path through it is marked unusable. Round r merges each node's
    segment with the segment ending at its round r pointer (simultaneous update), doubling the covered path.
    Row T is the identity sentinel shared by every root; it merges with itself into itself, so no state crosses
    trees. After plan["rounds"] rounds every node covers its full path from the root. Work O(nodes x rounds)."""
    if plan is None or len(plan["term"]) == 0:
        raise ValueError("mc_walk_jump needs a plan with at least one subset")
    if not owned and target_pos is None:
        raise ValueError("shared mode needs target_pos")
    dev = z.device
    pos = torch.as_tensor(plan["pos"], device=dev)
    if owned:
        own = torch.as_tensor(plan["owner"], device=dev)
        zn = z[pos, own][:, None]                                                    # [T, 1, H]
        if owner_target_pos is None:
            valid = torch.ones(zn.shape[:2], dtype=torch.bool, device=dev)
        else:
            otp = torch.as_tensor(np.asarray(owner_target_pos), device=dev)
            valid = (pos != otp[own])[:, None]                                       # each owner excludes its own target
    else:
        tgt = torch.as_tensor(np.asarray(target_pos), device=dev)
        zn = z[pos]                                                                  # [T, C, H]
        valid = pos[:, None] != tgt[None, :]                                         # [T, C]
    dt, Cp, dh = zn.dtype, zn.shape[1], V.shape[-1]
    vm = valid[..., None].expand_as(zn)
    vmf = vm.to(dt)
    ninf = torch.full((1,) + tuple(zn.shape[1:]), float("-inf"), dtype=dt, device=dev)
    m = torch.cat([torch.where(vm, zn, ninf.expand_as(zn)), ninf])
    s = torch.cat([vmf, torch.zeros_like(ninf)])
    u = torch.cat([V[pos][:, None] * vmf[..., None], torch.zeros(1, Cp, zn.shape[2], dh, dtype=dt, device=dev)])
    k = torch.cat([valid.to(dt), torch.zeros(1, Cp, dtype=dt, device=dev)])
    bad = torch.cat([~valid, torch.zeros(1, Cp, dtype=torch.bool, device=dev)])
    state = (m, s, u, k, bad)
    if plan["rounds"]:
        jumps = torch.as_tensor(plan["jumps"], device=dev)
        for r in range(plan["rounds"]):
            J = jumps[r]
            state = _jump_merge(tuple(t[J] for t in state), state)
    tt = torch.as_tensor(plan["term"], device=dev)
    m, s, u, k, bad = (t[tt] for t in state)
    usable = (~bad) & (k > 0)
    um = usable[..., None].expand_as(s)
    s_safe = torch.where(um, s, torch.ones_like(s))
    m_safe = torch.where(um, m, torch.zeros_like(m))
    pooled = u / s_safe[..., None]
    L = torch.where(um, m_safe + torch.log(s_safe), torch.zeros_like(m))
    return pooled, L, k, usable


def mc_masked_attention(z, V, member, channel):
    """Direct (uncached) attention: row r attends over facts member[r] with channel[r]'s scores.
    member [R, n] bool, channel [R] long. Returns pooled [R, H, dh], L [R, H], k [R], usable [R].
    Rows with no member are unusable; their outputs are zero placeholders and must not be used."""
    zr = z[:, channel, :].permute(1, 2, 0)                                           # [R, H, n]
    usable = member.any(1)
    mask = (member & usable[:, None])[:, None, :].expand_as(zr)
    zero = torch.zeros_like(zr)
    z_in = torch.where(mask, zr, zero)
    m = torch.where(mask, z_in, torch.full_like(zr, float("-inf"))).amax(-1)        # -inf only on unusable rows
    m_s = torch.where(usable[:, None], m, torch.zeros_like(m)).detach()
    w_arg = torch.where(mask, z_in - m_s[..., None], zero)
    w_un = w_arg.exp() * mask.to(zr.dtype)
    s = w_un.sum(-1)
    s_safe = torch.where(usable[:, None], s, torch.ones_like(s))
    pooled = torch.einsum("rhn,nhd->rhd", w_un, V) / s_safe[..., None]
    L = torch.where(usable[:, None], m_s + torch.log(s_safe), torch.zeros_like(s))
    return pooled, L, member.sum(1).to(z.dtype), usable


def w_subset_denominator(n, P):
    """Horvitz Thompson denominator for W: pi x (2^(n-1) - 1) with pi = P / (2^n - 1), the sampler's exact
    inclusion probability over the 2^n - 1 nonempty subsets of n facts, and 2^(n-1) - 1 nonempty supporting
    contexts per target. Computed in float64 without forming 2^n."""
    if n < 2:
        return 0.0
    ratio = (1.0 - 2.0 ** -(n - 1)) / (2.0 - 2.0 ** -(n - 1))     # (2^(n-1) - 1) / (2^n - 1)
    return float(P) * ratio


def mc_target_loss(pred_full, y, full_usable, pred_sub=None, sub_channel=None, sub_k=None, sub_usable=None,
                   mode="W", w_denom=None, delta=1.0):
    """Mean over targets with a nonempty full context of: 0.5 x full context Huber + 0.5 x subset term.
    W: Horvitz Thompson estimate of the mean over all nonempty supporting contexts = (sum of usable subset
       losses for the target) / w_denom; zero survivors give a subset term of 0 (still 0.5 x full + 0.5 x 0),
       which keeps the estimate unbiased. w_denom from w_subset_denominator(n, P): a scalar, or one value per
       target when targets come from different filings.
    T: exhaustive; equal weight per effective size class (mean within class, then mean over classes).
       A target with no usable subset uses its full context loss alone.
    Unusable subsets and targets never enter."""
    if mode not in ("W", "T"):
        raise ValueError(f"mode must be 'W' or 'T', got {mode!r}")
    if not bool(full_usable.any()):
        raise ValueError("no target has a nonempty full context")
    l_full = F.huber_loss(torch.where(full_usable, pred_full, y), y, delta=delta, reduction="none")
    C, dt, dev = len(y), l_full.dtype, y.device
    if pred_sub is None:
        if mode == "W":
            raise ValueError("W needs subset predictions (pass an empty usable mask for zero survivors)")
        per = l_full
    else:
        use = sub_usable & full_usable[sub_channel]
        ch, kk = sub_channel[use], sub_k[use].long()
        l_sub = F.huber_loss(pred_sub[use], y[ch], delta=delta, reduction="none")
        if mode == "W":
            if w_denom is None:
                raise ValueError("W needs w_denom (a scalar, or one value per target)")
            wd = torch.as_tensor(w_denom, dtype=dt, device=dev)
            wd = wd.expand(C) if wd.ndim == 0 else wd
            if tuple(wd.shape) != (C,) or not bool((wd > 0).all()):
                raise ValueError("w_denom must be positive: a scalar or one value per target")
            tot = torch.zeros(C, dtype=dt, device=dev).index_add(0, ch, l_sub)
            per = 0.5 * l_full + 0.5 * tot / wd
        else:
            kmax = int(kk.max()) + 1 if len(kk) else 1
            key = ch * kmax + kk
            g_tot = torch.zeros(C * kmax, dtype=dt, device=dev).index_add(0, key, l_sub)
            g_cnt = torch.zeros(C * kmax, dtype=dt, device=dev).index_add(0, key, torch.ones_like(l_sub))
            present = g_cnt > 0
            g_mean = torch.where(present, g_tot / g_cnt.clamp(min=1), torch.zeros_like(g_tot)).view(C, kmax)
            ncls = present.view(C, kmax).sum(1).to(dt)
            has = ncls > 0
            sub = torch.where(has, g_mean.sum(1) / ncls.clamp(min=1), torch.zeros_like(ncls))
            per = torch.where(has, 0.5 * l_full + 0.5 * sub, l_full)
    return per[full_usable].mean()