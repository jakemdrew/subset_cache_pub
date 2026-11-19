"""
Benchmark implementations for Powerset Validation Cache Benchmarks (PVCB),
built to the frozen PVCB0 contract. Every implementation computes the same
result: for each sampled subset S of a set and each channel c, attention of
the channel's query over the subset's facts,

    out[S, c] = sum_{i in S} softmax_{i in S}(K[i, c] . q[c] / sqrt(d)) V[i, c].

Shared inputs (one construction for every implementation): q [C, d], K [N, C, d],
V [N, C, d], drawn once in float64 from a seeded generator and cast, so every
device and precision sees the same values. A workload is one or more sets
concatenated along the fact axis (N = sum of set sizes); its subsets are
global position tuples, set by set, each set's subsets in canonical
(lexicographic, prefix first) order.

Implementations:
  cache            subset_attention_levels_exact_bwd (trie forward, reverse
                   trie walk backward) on z = K . q / sqrt(d); not chunked.
  sdpa_masked      torch.nn.functional.scaled_dot_product_attention over ALL
                   N facts, one query row per subset per channel, a boolean
                   subset mask; PyTorch's default backend selection.
  sdpa_masked_math the same call forced to the math backend.
  direct_gathered  subset_attention_uncached on z: attention over each
                   subset's gathered facts, grouped by subset size.
Direct implementations follow PVCB0's chunked backward protocol: attention inputs
formed once and detached into leaf copies; each chunk's unscaled loss term
backpropagated immediately (no chunk graph retained); one backward carries the
accumulated gradients upstream. Chunk terms add up to the unchunked loss.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

from set_attention_cache import (sample_subsets_powerset, sample_subsets_runs,
                                 sample_subsets_powerset_runs, build_prefix_trie_canonical)
from set_attention_model import subset_attention_levels_exact_bwd, subset_attention_uncached
from set_attention_data import forest_level_plan_np, plan_to_device

SEED, KEY_TAG, RUN_LEN, RUN_SIZES = 367, "pvcb", 8, (4, 8, 16, 32, 64)
IMPLS = ("cache", "sdpa_masked", "sdpa_masked_math", "direct_gathered")


def sample_set(n, M, sampler, file_name, seed=SEED, epoch=0, key_tag=KEY_TAG, run_len=None):
    """One set's subsets (sorted position tuples, canonical order) per PVCB0 / V2C0.
    run_len is required for powerset_runs and must be None for the other samplers."""
    if sampler != "powerset_runs" and run_len is not None:
        raise ValueError(f"run_len applies only to powerset_runs, got {run_len} for {sampler}")
    if sampler == "uniform_powerset":
        s = sample_subsets_powerset(n, file_name, M, seed=seed, epoch=epoch, key_tag=key_tag)
    elif sampler == "run_based":
        s = sample_subsets_runs(n, file_name, M, RUN_LEN, k_sizes=RUN_SIZES, seed=seed,
                                epoch=epoch, key_tag=key_tag)
    elif sampler == "powerset_runs":
        if run_len is None:
            raise ValueError("powerset_runs requires run_len")
        s = sample_subsets_powerset_runs(n, file_name, M, run_len, seed=seed, epoch=epoch, key_tag=key_tag)
    else:
        raise ValueError(f"unknown sampler {sampler}")
    s = sorted(tuple(int(x) for x in S) for S in s)
    if len(set(s)) != len(s):
        raise AssertionError("duplicate subsets")
    return s


def make_workload(ns, M, sampler, tag="", run_len=None):
    """Sets of sizes ns (concatenated), M subsets requested per set; run_len for powerset_runs."""
    subs, tries, offsets, counts, off = [], [], [], [], 0
    for i, n in enumerate(ns):
        s = sample_set(int(n), M, sampler, f"pvcb_n{n}_set{i}{tag}", run_len=run_len)
        tries.append(build_prefix_trie_canonical(s) if s else None)
        offsets.append(off)
        counts.append(len(s))
        subs.extend(tuple(off + x for x in S) for S in s)
        off += int(n)
    nodes = sum(len(t["pos"]) for t in tries if t is not None)
    facts = sum(len(S) for S in subs)
    depth = max((int(t["depth"].max()) + 1 for t in tries if t is not None), default=0)
    return {"ns": list(map(int, ns)), "M": M, "sampler": sampler, "run_len": run_len, "N": off, "subs": subs,
            "tries": tries, "offsets": offsets, "counts": counts, "P": len(subs),
            "nodes": nodes, "facts": facts, "depth": depth}


def make_inputs(N, C, d, seed, dtype, device, q_scale=1.0):
    """Shared q [C, d], K [N, C, d], V [N, C, d]: drawn once in float64 on CPU, cast."""
    g = torch.Generator().manual_seed(int(seed))
    q = torch.randn(C, d, generator=g, dtype=torch.float64) * q_scale
    K = torch.randn(N, C, d, generator=g, dtype=torch.float64)
    V = torch.randn(N, C, d, generator=g, dtype=torch.float64)
    return [t.to(device=device, dtype=dtype).requires_grad_() for t in (q, K, V)]


def make_weights(P, C, d, seed, dtype, device):
    """Fixed loss weights W [P, C, d] for the attention only tier's scalar loss."""
    g = torch.Generator().manual_seed(int(seed) + 7919)
    return torch.randn(P, C, d, generator=g, dtype=torch.float64).to(device=device, dtype=dtype)


def scores(q, K):
    """z [N, C] = K . q / sqrt(d), the scores every implementation shares."""
    return torch.einsum("ncd,cd->nc", K, q) / math.sqrt(q.shape[-1])


def cache_plan(wl, device):
    """Preparation for `cache`: the forest plan of the workload's tries, on device."""
    return plan_to_device(forest_level_plan_np(wl["tries"], wl["offsets"]), device)


def subset_mask(subs, N, device):
    """Preparation for `sdpa_masked*`: boolean [len(subs), N], True = the subset's facts."""
    lens = np.fromiter((len(S) for S in subs), dtype=np.int64, count=len(subs))
    rows = np.repeat(np.arange(len(subs)), lens)
    cols = np.fromiter((x for S in subs for x in S), dtype=np.int64, count=int(lens.sum()))
    m = torch.zeros(len(subs), N, dtype=torch.bool)
    m[torch.from_numpy(rows), torch.from_numpy(cols)] = True
    return m.to(device)


def attn_cache(q, K, V, plan):
    return subset_attention_levels_exact_bwd(scores(q, K), V, plan)


def _sdpa(Qc, Kh, Vh, mask, math_only):
    """SDPA call: Qc [1, C, p, d], Kh and Vh [1, C, N, d], mask [p, N] (True = attend)."""
    am = mask[None, None]
    if math_only:
        with sdpa_kernel([SDPBackend.MATH]):
            return F.scaled_dot_product_attention(Qc, Kh, Vh, attn_mask=am)
    return F.scaled_dot_product_attention(Qc, Kh, Vh, attn_mask=am)


def sdpa_layout(q, K, V):
    """SDPA layout for the whole workload: Kh, Vh [1, C, N, d] (one copy each)."""
    return K.permute(1, 0, 2).unsqueeze(0).contiguous(), V.permute(1, 0, 2).unsqueeze(0).contiguous()


def sdpa_chunk(q, Kh, Vh, mask, math_only):
    """One chunk of subsets: one query row per subset per channel. Returns [p, C, d]."""
    C, d = q.shape
    Qc = q[None, :, None, :].expand(1, C, mask.shape[0], d).contiguous()
    return _sdpa(Qc, Kh, Vh, mask, math_only)[0].permute(1, 0, 2)


SDPA_OPS = {"_scaled_dot_product_flash_attention": "FLASH_ATTENTION",
            "_scaled_dot_product_flash_attention_for_cpu": "FLASH_ATTENTION (cpu)",
            "_scaled_dot_product_efficient_attention": "EFFICIENT_ATTENTION",
            "_scaled_dot_product_cudnn_attention": "CUDNN_ATTENTION",
            "_scaled_dot_product_attention_math": "MATH"}


def sdpa_executed_backend(q, K, V, mask, math_only):
    """The SDPA operator(s) actually executed, recorded with the profiler while running
    forward + backward on the SAME kind of inputs the benchmark uses (gradient enabled,
    same dtype, device, shapes and mask). Returns a sorted, comma separated label, or
    'not recorded' if the profiler saw no SDPA operator."""
    from torch.profiler import profile, ProfilerActivity
    qL = q.detach().clone().requires_grad_()
    KL = K.detach().clone().requires_grad_()
    VL = V.detach().clone().requires_grad_()
    Kh, Vh = sdpa_layout(qL, KL, VL)
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        sdpa_chunk(qL, Kh, Vh, mask, math_only).sum().backward()
    names = {e.key.replace("aten::", "") for e in prof.key_averages()}
    found = sorted({lab for op, lab in SDPA_OPS.items() if op in names})
    return ", ".join(found) if found else "not recorded"


def chunks_of(P, chunk):
    if chunk is None or chunk >= P:
        return [slice(0, P)]
    return [slice(i, min(i + chunk, P)) for i in range(0, P, chunk)]


def run_attention(impl, q, K, V, wl, W, chunk=None, plan=None, masks=None, collect=True):
    """Attention only tier: forward + backward of sum(W * out) for one implementation.
    Gradients accumulate into q, K, V (cleared first).

    cache: z = K . q / sqrt(d) once, one trie forward, one reverse walk backward.
    Direct implementations follow PVCB0's chunked backward protocol: the attention
    inputs are formed ONCE (direct_gathered: z and V; sdpa: q and the [1, C, N, d]
    layouts of K and V), detached into leaf copies, each chunk's UNSCALED loss term
    is backpropagated immediately into the leaf copies (no chunk graph retained),
    and one backward then carries the accumulated gradients upstream to q, K, V.

    `plan` (cache) and `masks` (one mask per chunk, sdpa) may be passed in
    precomputed (preparation tier); otherwise they are built here. With
    collect=True the detached outputs are returned (certification); with
    collect=False nothing is kept and None is returned (timing)."""
    for t in (q, K, V):
        t.grad = None
    P = wl["P"]
    if impl == "cache":
        if chunk is not None:
            raise ValueError("cache is not chunked")
        plan = cache_plan(wl, q.device) if plan is None else plan
        out = attn_cache(q, K, V, plan)
        (W * out).sum().backward()
        return out.detach() if collect else None
    if impl == "direct_gathered":
        ups = [scores(q, K), V]
    elif impl in ("sdpa_masked", "sdpa_masked_math"):
        ups = [q, *sdpa_layout(q, K, V)]
    else:
        raise ValueError(f"unknown implementation {impl}")
    leaves = [u.detach().requires_grad_() for u in ups]
    outs = [] if collect else None
    for j, sl in enumerate(chunks_of(P, chunk)):
        sub_c = wl["subs"][sl]
        if impl == "direct_gathered":
            o = subset_attention_uncached(leaves[0], leaves[1], sub_c)
        else:
            m = masks[j] if masks is not None else subset_mask(sub_c, wl["N"], q.device)
            o = sdpa_chunk(leaves[0], leaves[1], leaves[2], m, impl == "sdpa_masked_math")
        (W[sl] * o).sum().backward()
        if collect:
            outs.append(o.detach())
        del o
    torch.autograd.backward(ups, [lf.grad for lf in leaves])
    return torch.cat(outs) if collect else None