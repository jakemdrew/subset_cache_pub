"""
Subset indexed KV cache for set based causal attention.

Prefix states over a canonical token order (ascending ids) make every
prefix of that order a reusable unit. Two state embodiments: online
softmax triples (m, s, u) and additive pairs (sum a, sum b). Two cache
modes: RESIDENT (dict of every prefix state seen) and STREAMING
(sweep_lex branch stack, at most k states resident). Combinatorial
rank functions give random access by integer into the sorted
enumerations (fixed k layers and the full power set).
"""
import numpy as np
from math import comb
from typing import Dict, List, Optional, Sequence, Tuple

from set_attention import SetAttentionParams, keys_values


# ---------------------------------------------------------------------
# Combinatorial rank: random access into the sorted subset enumerations
# ---------------------------------------------------------------------
def rank_combination(subset: Sequence[int]) -> int:
    """Colex rank of a sorted k-combination."""
    return sum(comb(c, j + 1) for j, c in enumerate(subset))


def unrank_combination(rank: int, n: int, k: int) -> List[int]:
    """The sorted k-subset of range(n) at `rank` (colex order)."""
    if not 0 <= rank < comb(n, k):
        raise ValueError(f"rank {rank} outside [0, C({n},{k})) = [0, {comb(n, k)})")
    out, r = [], rank
    c = n - 1
    for j in range(k, 0, -1):
        while comb(c, j) > r:
            c -= 1
        out.append(c)
        r -= comb(c, j)
        c -= 1
    return out[::-1]


def strided_ranks(n: int, k: int, stride: int) -> range:
    """Every `stride`-th rank of the C(n, k) space: uniform coverage sampling.

    Returns a range of python ints (arbitrary precision), so spaces past
    2**63 do not overflow the way a fixed width integer array would.
    """
    return range(0, comb(n, k), stride)


def unrank_lex_combination(rank: int, n: int, k: int) -> List[int]:
    """
    The sorted k-subset of range(n) at `rank` in LEX order (first element most
    significant). Consecutive lex ranks share head prefixes, which is what the
    branch stack in sweep_lex() exploits.
    """
    if not 0 <= rank < comb(n, k):
        raise ValueError(f"rank {rank} outside [0, C({n},{k})) = [0, {comb(n, k)})")
    out, r, x = [], rank, 0
    for j in range(k):
        while True:
            cnt = comb(n - 1 - x, k - 1 - j)
            if r < cnt:
                out.append(x)
                x += 1
                break
            r -= cnt
            x += 1
    return out


def lex_successor(subset: List[int], n: int, k: int) -> bool:
    """
    Advance `subset` (ascending ids) IN PLACE to the next k-subset in LEX
    order: the odometer increment with strictly increasing digits. Bump the
    last id; on overflow carry left and reset the suffix to consecutive ids.
    Amortized constant work per call. Returns False when exhausted.
    """
    i = k - 1
    while i >= 0 and subset[i] == n - k + i:
        i -= 1
    if i < 0:
        return False
    subset[i] += 1
    for j in range(i + 1, k):
        subset[j] = subset[j - 1] + 1
    return True


# ---------------------------------------------------------------------
# The cache
# ---------------------------------------------------------------------
class SetAttentionCache:
    """
    Caches online-softmax prefix states for ONE token set (one filing).

    K scores and V rows are precomputed once per set; after that, extending a
    prefix by one token costs one exp, one multiply-accumulate on a [dim]
    vector, and one dict insert. attend() walks from the longest cached
    prefix of the requested subset.
    """

    def __init__(self, params: SetAttentionParams, ids: np.ndarray, values: np.ndarray,
                 query: Optional[np.ndarray] = None):
        self.n = len(ids)
        q = params.q_seed if query is None else query
        K, V = keys_values(params, ids, values)
        self.scores = (K @ q) / np.sqrt(params.dim)   # [n] per-token logits
        self.V = V                                     # [n, dim]
        # additive embodiment: exponentials against one common max, and
        # their weighted values. Subset output = b[S].sum() / a[S].sum().
        M = float(self.scores.max()) if self.n else 0.0
        self.a = np.exp(self.scores - M)               # [n]
        self.b = self.a[:, None] * V                   # [n, dim]
        # prefix tuple -> (running max, denominator, weighted V sum)
        self._states: Dict[Tuple[int, ...], Tuple[float, float, np.ndarray]] = {}
        # instrumentation
        self.requests = 0
        self.tokens_requested = 0
        self.extensions = 0

    def _extend(self, state, pos: int):
        """Online softmax update of (m, s, u) with token at local position `pos`."""
        m, s, u = state
        sc = self.scores[pos]
        if sc > m:
            corr = np.exp(m - sc)
            m, s, u = sc, s * corr, u * corr
        w = np.exp(sc - m)
        return m, s + w, u + w * self.V[pos]

    def _extend_additive(self, state, pos: int):
        """Additive embodiment update: running sums of a_i and a_i * V_i."""
        sa, sb = state
        return sa + self.a[pos], sb + self.b[pos]

    def attend(self, subset: Sequence[int]):
        """
        Attention output over `subset` (sorted local token positions), reusing
        the longest cached prefix. Returns out [dim]; caches every prefix
        state created on the way.
        """
        subset = tuple(subset)
        if not subset:
            raise ValueError("attend() requires a nonempty subset")
        if subset[0] < 0 or subset[-1] >= self.n or any(
                subset[i] >= subset[i + 1] for i in range(len(subset) - 1)):
            raise ValueError("subset must be strictly increasing positions in [0, n)")
        self.requests += 1
        self.tokens_requested += len(subset)

        # longest cached prefix
        j = len(subset)
        while j > 0 and subset[:j] not in self._states:
            j -= 1
        if j == 0:
            state = (-np.inf, 0.0, np.zeros(self.V.shape[1]))
        else:
            state = self._states[subset[:j]]

        for t in range(j, len(subset)):
            state = self._extend(state, subset[t])
            self._states[subset[: t + 1]] = state
            self.extensions += 1

        m, s, u = state
        return u / s

    def recompute_fraction(self) -> float:
        """Extensions actually performed over tokens requested (1.0 = no reuse)."""
        return self.extensions / max(self.tokens_requested, 1)

    def cached_prefixes(self) -> int:
        return len(self._states)

    def reset_counters(self):
        self.requests = self.tokens_requested = self.extensions = 0

    # -----------------------------------------------------------------
    # STREAMING MODE: the branch stack sweep (sorted enumeration holds
    # one lattice branch at a time; ~k states resident, zero reuse loss)
    # -----------------------------------------------------------------
    def sweep_lex(self, k: int, stride: int = 1, collect: bool = False,
                  mode: str = "softmax"):
        """
        Sweep every `stride`-th k-subset of this set in LEX order using a
        branch stack: states for the current prefix path only. A popped state
        is never needed again because the enumeration is sorted. Sets
        self.peak_stack (max states resident). EVERY visited subset's output
        is computed and consumed (the last is kept in self.last_output), so
        timings are comparable to baselines that also produce outputs;
        collect=True additionally returns them all [num_subsets, dim]
        (use only on small sweeps).

        Advancing: small strides count forward with the lex_successor
        odometer (amortized constant work per step); large strides jump by
        unranking (one combinatorial computation regardless of stride).

        mode="softmax": online softmax states (m, s, u), robust to any score
        range. mode="additive": running sums of precomputed exponentials
        (sum a, sum b); fewer ops per extension, but can underflow when a
        subset's scores all sit far below the set's maximum score.
        """
        if not 1 <= k <= self.n:
            raise ValueError(f"k must be in [1, {self.n}], got {k}")
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        if mode not in ("softmax", "additive"):
            raise ValueError(f"unknown mode: {mode}")
        softmax = mode == "softmax"
        extend = self._extend if softmax else self._extend_additive
        base = ((-np.inf, 0.0, np.zeros(self.V.shape[1])) if softmax
                else (0.0, np.zeros(self.V.shape[1])))

        total = comb(self.n, k)
        stack: List[tuple] = []
        prev: List[int] = []
        outputs = [] if collect else None
        self.peak_stack = 0
        count_forward = stride <= self.n   # count when cheap, jump when far

        subset = list(range(k))            # rank 0 in lex order
        r = 0
        while r < total:
            p = 0
            while p < len(prev) and prev[p] == subset[p]:
                p += 1
            del stack[p:]
            state = stack[p - 1] if p > 0 else base
            for t in range(p, k):
                state = extend(state, subset[t])
                stack.append(state)
                self.extensions += 1
            self.requests += 1
            self.tokens_requested += k
            self.peak_stack = max(self.peak_stack, len(stack))
            y = state[2] / state[1] if softmax else state[1] / state[0]
            self.last_output = y           # every subset pays normalization
            if collect:
                outputs.append(y)
            prev = subset

            r += stride
            if r >= total:
                break
            if count_forward:
                subset = subset.copy()     # prev keeps the processed ids
                for _ in range(stride):
                    lex_successor(subset, self.n, k)
            else:
                subset = unrank_lex_combination(r, self.n, k)

        return np.asarray(outputs) if collect else None


# ---------------------------------------------------------------------
# Full power set enumeration (any subset size): random access by rank
# ---------------------------------------------------------------------
def unrank_lex_subset(r, n):
    """Unrank the full power set (nonempty subsets of range(n), any size).

    Enumeration order is lexicographic with each prefix preceding its
    extensions: {0}, {0,1}, {0,1,2}, ... so consecutive ranks share long
    prefixes (branch stack friendly). Total subsets = 2**n - 1. Ranks are
    arbitrary precision ints, so n = 145 (2**145 - 1 ranks) works natively.
    """
    if not 0 <= r < (1 << n) - 1:
        raise ValueError(f"rank {r} outside [0, 2^{n} - 1)")
    subset = []
    e = 0
    while True:
        block = 1 << (n - 1 - e)      # subsets whose next element is e
        if r < block:
            subset.append(e)
            if r == 0:
                return tuple(subset)
            r -= 1                     # descend into extensions of ... e
            e += 1
        else:
            r -= block                 # skip e entirely, try e + 1
            e += 1


def powerset_total(n):
    """Number of nonempty subsets of an n element set."""
    return (1 << n) - 1


# ---------------------------------------------------------------------
# Sampling embodiments: the two layers of the systematic sampling story
# ---------------------------------------------------------------------
def systematic_sample(n, k, seed, key):
    """Deterministic systematic sample of element positions (the TRAINING
    INPUT layer: occurrences within one document, small integer n).

    Returns all positions when n <= k; otherwise exactly k evenly spaced
    positions idx = floor((arange(k) + u) * n / k), where the starting
    offset u in [0, 1) is a fixed function of (seed, crc32(key)). The same
    (n, k, seed, key) always yields the same sample, bit for bit; across
    keys the offsets form a hash family, decorrelating which structural
    positions are sampled across documents (the MinHash discipline: a
    deterministic function that is statistically unstructured). Matches the
    TT3 reference rule exactly.

    Embodiment: systematic element sampling with a seeded per key starting
    offset; deterministic given the seed, unbiased for means over the
    element sequence in expectation across the offset family.
    """
    import zlib as _z
    if k < 1 or n < 0:
        raise ValueError("require n >= 0 and k >= 1")
    if n <= k:
        return np.arange(n, dtype=np.int64)
    u = np.random.default_rng((seed, _z.crc32(str(key).encode()))).uniform()
    return np.minimum(np.floor((np.arange(k) + u) * n / k),
                      n - 1).astype(np.int64)


def strided_ranks_random_start(total_ranks, stride, seed, key):
    """Strided combinatorial ranks with a seeded random starting rank (the
    INTERROGATION layer: subset ranks over an enumeration of total_ranks
    entries, arbitrary precision integers supported, e.g. 2**145).

    Returns the Python int iterator range(offset, total_ranks, stride),
    where offset is drawn uniformly from [0, stride) as a fixed function of
    (seed, key): classical systematic sampling with random start, retaining
    every prefix reuse property of deterministic striding. Every rank has
    inclusion probability exactly 1/stride, so stride x (sample sum) is an
    unbiased estimator of the enumeration total. The plain sample mean is
    unbiased only when stride divides total_ranks; otherwise offsets yield
    different sample sizes and it can be biased (values [0,0,0,0,1],
    stride 2: expected sample mean 1/6, population mean 1/5). The offset is derived from seeded random
    bytes reduced modulo stride with 64 guard bits, so the modulo bias is
    below 2**-64 at any stride size. Deterministic given (seed, key); a
    family of keys yields independent systematic samples (hash family
    coverage schedules). Regular rank spacing covers RANK SPACE; it is not
    automatically balanced coverage of feature identities, subset sizes, or
    interaction orders, which the caller reports per the plan of record.
    """
    import zlib as _z
    if stride < 1 or total_ranks < 1:
        raise ValueError("require total_ranks >= 1 and stride >= 1")
    if stride >= total_ranks:
        stride = total_ranks  # degenerates to a single sampled rank
    rng = np.random.default_rng((seed, _z.crc32(str(key).encode())))
    nbytes = (int(stride).bit_length() + 64 + 7) // 8
    offset = int.from_bytes(rng.bytes(nbytes), "big") % int(stride)
    return range(offset, int(total_ranks), int(stride))
    
# ---------------------------------------------------------------------
# Trie embodiment: many sampled subsets of one set evaluated in ONE call,
# each shared canonical prefix computed once (Thesis Validation Medium)
# ---------------------------------------------------------------------
def sample_subsets_runs(n, file_name, M, run_len, k_sizes=(4, 8, 16, 32, 64),
                        seed=367, epoch=0, key_tag=None):
    """Subsets of range(n) for one filing, as sorted position tuples.

    Budget M is split evenly over the valid sizes (k < n), remainder to the
    largest sizes, each size capped at C(n, k); a shortfall is NOT
    redistributed. Per size, r = ceil(a / run_len) anchor ranks come from
    strided_ranks_random_start keyed (seed, f"{epoch}|{k}|{file_name}"
    [+ f"|{key_tag}"]); each anchor is followed by consecutive LEX ranks,
    which share long canonical prefixes, so run_len is the coverage versus
    reuse dial (run_len = 1 is the plain strided sampler). Duplicates are
    removed and the actual count is what the caller normalizes by.
    """
    from itertools import islice
    if M < 0 or run_len < 1:
        raise ValueError("require M >= 0 and run_len >= 1")
    valid = [k for k in k_sizes if k < n]
    if M == 0 or not valid:
        return []
    base, rem = divmod(M, len(valid))
    alloc = {k: base for k in valid}
    for k in sorted(valid, reverse=True)[:rem]:
        alloc[k] += 1
    out = []
    for k in valid:
        total = comb(n, k)
        a = min(alloc[k], total)
        if a == 0:
            continue
        r = -(-a // run_len)
        key = f"{epoch}|{k}|{file_name}" + (f"|{key_tag}" if key_tag else "")
        anchors = list(islice(iter(strided_ranks_random_start(
            total, max(total // r, 1), seed, key)), r))
        ranks, seen = [], set()
        for a0 in anchors:
            for t in range(run_len):
                rk = a0 + t
                if rk >= total or len(ranks) >= a:
                    break
                if rk not in seen:
                    seen.add(rk)
                    ranks.append(rk)
        out.extend(tuple(unrank_lex_combination(int(x), n, k)) for x in ranks)
    if len(set(out)) != len(out):
        raise AssertionError("duplicate subsets")
    return out


def build_prefix_trie(subsets, with_mask=True):
    """Prefix trie over sorted position tuples. Nodes keyed by canonical
    prefix and inserted parents first, so node indices strictly increase
    along every path. Returns dict: pos [T] (position in the set), parent
    [T] (-1 for roots), depth [T], mult [T] (evaluated paths containing the
    node), allowed [T, T] bool (node attends to its ancestors and itself;
    None when with_mask=False, which avoids T x T memory at large M),
    paths (node indices per subset, in input order)."""
    index, pos, parent, mult = {}, [], [], []
    for S in subsets:
        for d in range(1, len(S) + 1):
            pre = tuple(S[:d])
            if pre not in index:
                index[pre] = len(pos)
                pos.append(S[d - 1])
                parent.append(index[pre[:-1]] if d > 1 else -1)
                mult.append(0)
            mult[index[pre]] += 1
    T = len(pos)
    parent = np.array(parent, dtype=np.int64)
    depth = np.zeros(T, dtype=np.int64)
    for i in range(T):
        depth[i] = 0 if parent[i] == -1 else depth[parent[i]] + 1
    allowed = None                        # with_mask=False: callers that never use
    if with_mask:                         # the T x T ancestor mask skip its memory
        allowed = np.zeros((T, T), dtype=bool)
        for i in range(T):
            j = i
            while j != -1:
                allowed[i, j] = True
                j = parent[j]
    paths = [[index[tuple(S[:d])] for d in range(1, len(S) + 1)] for S in subsets]
    return {"pos": np.array(pos, dtype=np.int64), "parent": parent,
            "depth": depth, "mult": np.array(mult, dtype=np.int64),
            "allowed": allowed, "paths": paths}


def subset_coverage(sampled_subsets, n):
    """Coverage of SAMPLED subsets only (callers exclude the full set, which
    trivially covers everything). Returns (feature fraction appearing in at
    least one subset, pair fraction co-occurring in at least one subset,
    mean subset size). Pair coverage is not exhaustive interaction
    coverage; higher order interactions are not measured. Pair coverage is
    NaN when n < 2 (no pairs exist)."""
    if not sampled_subsets:
        return float("nan"), float("nan"), float("nan")
    feat = np.zeros(n, dtype=bool)
    pairs = np.zeros((n, n), dtype=bool)
    for S in sampled_subsets:
        idx = np.asarray(S)
        feat[idx] = True
        pairs[np.ix_(idx, idx)] = True
    n_pairs = comb(n, 2)
    pair_cov = float(np.triu(pairs, k=1).sum() / n_pairs) if n_pairs else float("nan")
    return (float(feat.mean()), pair_cov,                 # NaN: no pairs exist when n < 2
            float(np.mean([len(S) for S in sampled_subsets])))


def cache_from_scores_values(scores, V):
    """A SetAttentionCache over externally supplied per token scores and
    values (e.g., a trained readout's scores over trie node states), with
    the same fields __init__ sets. Used to cross check trie readouts
    against the reference cache: attend(path) over node indices that
    strictly increase along each trie path."""
    c = SetAttentionCache.__new__(SetAttentionCache)
    c.scores = np.asarray(scores, dtype=np.float64)
    c.V = np.asarray(V, dtype=np.float64)
    c.n = len(c.scores)
    M = float(c.scores.max()) if c.n else 0.0
    c.a = np.exp(c.scores - M)
    c.b = c.a[:, None] * c.V
    c._states = {}
    c.requests = c.tokens_requested = c.extensions = 0
    return c
    
def pack_tries(tries):
    """Pad several filings' tries to one length for one batched call.
    Returns node_pos [B, Tmax] (-1 = pad), node_pad [B, Tmax] bool, and
    forbid [B, 1, Tmax, Tmax] bool (True = forbidden). Real nodes attend only
    to their own ancestors and themselves; pad rows attend only to
    themselves (no all forbidden rows); no node attends across filings."""
    if len(tries) == 0:
        raise ValueError("pack_tries needs at least one trie")
    Ts = [len(t["pos"]) for t in tries]
    if min(Ts) < 1:
        raise ValueError("empty trie")
    B, Tm = len(tries), max(Ts)
    pos = np.full((B, Tm), -1, dtype=np.int64)
    pad = np.ones((B, Tm), dtype=bool)
    forbid = np.ones((B, 1, Tm, Tm), dtype=bool)
    for b, (t, T) in enumerate(zip(tries, Ts)):
        al = np.asarray(t["allowed"], dtype=bool)
        if al.shape != (T, T):
            raise ValueError(f"trie {b}: allowed is {al.shape}, expected {(T, T)}")
        if not al.diagonal().all():
            raise ValueError(f"trie {b}: every node must attend to itself")
        pos[b, :T] = t["pos"]
        pad[b, :T] = False
        forbid[b, 0, :T, :T] = ~al
        ar = np.arange(T, Tm)
        forbid[b, 0, ar, ar] = False
    if forbid[:, 0].all(axis=-1).any():
        raise ValueError("an all forbidden attention row would produce NaN")
    return {"node_pos": pos, "node_pad": pad, "forbid": forbid, "Tmax": Tm}
    
# ---------------------------------------------------------------------
# Thesis sampler: uniform stride over the FULL power set (every subset
# size) in the canonical enumeration, and its one pass trie builder
# ---------------------------------------------------------------------
def sample_subsets_powerset(n, file_name, M, seed=367, epoch=0, key_tag=None):
    """The thesis sampler: a uniform stride over the ENTIRE power set of
    range(n) (every subset size, 2**n - 1 nonempty subsets), in the library's
    canonical enumeration (unrank_lex_subset: lexicographic, each prefix
    before its extensions).

    Ranks are range(offset, 2**n - 1, stride) from strided_ranks_random_start
    with stride = ceil((2**n - 1) / M) and the random start keyed (seed,
    f"{epoch}|powerset|{file_name}" [+ f"|{key_tag}"]): each subset is
    reached by exactly one start in [0, stride), so its inclusion
    probability is 1 / stride up to the start's modulo bias (below 2**-64,
    see strided_ranks_random_start), and the count is at most M
    (all 2**n - 1 subsets when they number M or fewer). The full set itself
    is one rank of the enumeration and is sampled like any other. Returns
    sorted position tuples in canonical (ascending rank) order; the caller
    normalizes by the actual count."""
    if M < 0 or n < 0:
        raise ValueError("require M >= 0 and n >= 0")
    if M == 0 or n == 0:
        return []
    total = powerset_total(n)
    stride = -(-total // M)
    key = f"{epoch}|powerset|{file_name}" + (f"|{key_tag}" if key_tag else "")
    return [unrank_lex_subset(r, n) for r in strided_ranks_random_start(total, stride, seed, key)]


def build_prefix_trie_canonical(subsets):
    """Prefix trie over subsets given in strictly increasing canonical order
    (lexicographic on sorted position tuples, a prefix before its
    extensions), built in one pass: in that order the longest prefix a subset
    shares with ANY earlier subset is the one it shares with its predecessor,
    so the trie path is a stack. Linear in the nodes created (no T x T mask,
    no per prefix tuples), same node set, numbering, parents and depths as
    build_prefix_trie(subsets, with_mask=False). Raises if the order is not
    strictly increasing. Returns dict: pos, parent (-1 for roots), depth,
    terminals (each subset's last node, in input order)."""
    pos, parent, depth, terminals, stack = [], [], [], [], []
    prev = ()
    for S in subsets:
        L = len(S)
        if L == 0:
            raise ValueError("empty subset")
        m = min(L, len(prev))
        l = 0
        while l < m and S[l] == prev[l]:
            l += 1
        if not ((l < m and S[l] > prev[l]) or (l == len(prev) < L)):
            raise ValueError("subsets are not in strictly increasing canonical order")
        del stack[l:]
        for d in range(l, L):
            stack.append(len(pos))
            pos.append(S[d])
            parent.append(stack[-2] if d > 0 else -1)
            depth.append(d)
        terminals.append(stack[-1])
        prev = S
    return {"pos": np.asarray(pos, dtype=np.int64), "parent": np.asarray(parent, dtype=np.int64),
            "depth": np.asarray(depth, dtype=np.int64),
            "terminals": np.asarray(terminals, dtype=np.int64)}
            
# ---------------------------------------------------------------------
# Uniform full power set RUNS: rotated pattern of consecutive canonical ranks
# ---------------------------------------------------------------------
def canonical_successor(S, n):
    """Next subset after S in the canonical enumeration (unrank_lex_subset's
    order: lexicographic, each prefix before its extensions), or None after
    the last subset (n - 1,). S: sorted position tuple. Extend by max + 1
    when possible (descend the branch); otherwise drop the last element and
    increment the new last (move to the next branch)."""
    if S[-1] < n - 1:
        return S + (S[-1] + 1,)
    T_ = S[:-1]
    if not T_:
        return None
    return T_[:-1] + (T_[-1] + 1,)


def powerset_run_base(n, M, run_len):
    """The fixed pattern of runs over the canonical enumeration of the
    2**n - 1 nonempty subsets: R = M // L runs of L = min(run_len, M)
    consecutive ranks starting at j * s (s = T // R >= L, so runs never
    overlap). Returns (T, R, L, s); P = R * L. Requires 1 <= M < T."""
    T = powerset_total(n)
    if not 1 <= M < T or run_len < 1:
        raise ValueError("require 1 <= M < 2**n - 1 and run_len >= 1")
    L = min(int(run_len), int(M))
    R = int(M) // L
    s = T // R
    assert s >= L
    return T, R, L, s


def powerset_run_ranks(n, M, run_len, offset):
    """Ranks of the rotated pattern for a given offset in [0, T): the run
    starts (j * s + offset) mod T, each followed by L - 1 consecutive ranks
    mod T. For a uniform offset every rank is included with probability
    exactly P / T. Exposed for exhaustive certification on small n."""
    T, R, L, s = powerset_run_base(n, M, run_len)
    if not 0 <= offset < T:
        raise ValueError("offset outside [0, T)")
    return [((j * s + offset) % T, L) for j in range(R)]


def sample_subsets_powerset_runs(n, file_name, M, run_len, seed=367, epoch=0, key_tag=None):
    """Uniform entry points with traversal locality over the ENTIRE power set:
    a fixed pattern of R = M // L runs of L = min(run_len, M) consecutive
    canonical ranks (unrank_lex_subset's order), rotated by an offset drawn
    uniformly from [0, 2**n - 1) keyed (seed, f"{epoch}|powerset_runs|
    {file_name}" [+ f"|{key_tag}"]), ranks taken modulo 2**n - 1. Every
    nonempty subset has inclusion probability exactly P / (2**n - 1),
    P = R * L, up to the offset's modulo bias (below 2**-64, 64 guard bits as
    in strided_ranks_random_start). A run is L consecutive ranks; it follows
    the depth first enumeration and may stop before its branch ends. M >= 2**n - 1
    returns every subset. Returns sorted position tuples in canonical order
    (distinct by construction); the caller normalizes by the actual count."""
    import zlib as _z
    if M < 0 or n < 0:
        raise ValueError("require M >= 0 and n >= 0")
    if M == 0 or n == 0:
        return []
    T = powerset_total(n)
    if M >= T:
        return [unrank_lex_subset(r, n) for r in range(T)]
    key = f"{epoch}|powerset_runs|{file_name}" + (f"|{key_tag}" if key_tag else "")
    rng = np.random.default_rng((seed, _z.crc32(key.encode())))
    nbytes = (int(T).bit_length() + 64 + 7) // 8
    offset = int.from_bytes(rng.bytes(nbytes), "big") % int(T)
    out = []
    for start, L in powerset_run_ranks(n, M, run_len, offset):
        S = unrank_lex_subset(start, n)
        for _ in range(L):
            out.append(S)
            S = canonical_successor(S, n)
            if S is None:                     # past rank T - 1: continue at rank 0
                S = (0,)
    out.sort()
    return out