"""
set_attention.py - REFERENCE implementation: full, uncached set attention.

This is the ground truth for set_attention_cache.py: the cache's completed
pass over a set must equal full_set_attention() to within tolerance. The
canonical order is a computation order, not a mask; this reference has no
order at all (softmax over the whole set), which is exactly the point.

Tokens arrive as (ids, values): registry column ids (ascending = canonical
order) and pct-change values, i.e. the output of MLSparseFeatures.tokens().
Token vector = embed[id] + signed_log(value) * w_val + b_val.
All parameters are random but seeded: correctness needs no training.
numpy float64 only.
"""
from dataclasses import dataclass, field

import numpy as np


def signed_log(values: np.ndarray) -> np.ndarray:
    """Tail taming: sign(v) * log1p(|v|)."""
    return np.sign(values) * np.log1p(np.abs(values))


@dataclass
class SetAttentionParams:
    """Seeded random parameters shared by the reference and the cache."""
    vocab_size: int
    dim: int = 64
    seed: int = 367

    embed: np.ndarray = field(init=False)   # [vocab, dim] token identity embeddings
    w_val: np.ndarray = field(init=False)   # [dim] value displacement direction
    b_val: np.ndarray = field(init=False)   # [dim] value projection bias
    W_k: np.ndarray = field(init=False)     # [dim, dim] key projection
    W_v: np.ndarray = field(init=False)     # [dim, dim] value projection
    q_seed: np.ndarray = field(init=False)  # [dim] readout query vector

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        scale = 1.0 / np.sqrt(self.dim)
        self.embed = rng.normal(0.0, scale, size=(self.vocab_size, self.dim))
        self.w_val = rng.normal(0.0, scale, size=self.dim)
        self.b_val = rng.normal(0.0, scale, size=self.dim)
        self.W_k = rng.normal(0.0, scale, size=(self.dim, self.dim))
        self.W_v = rng.normal(0.0, scale, size=(self.dim, self.dim))
        self.q_seed = rng.normal(0.0, scale, size=self.dim)


def token_vectors(params: SetAttentionParams, ids: np.ndarray, values: np.ndarray) -> np.ndarray:
    """[T, dim] token vectors: identity embedding + value displacement."""
    return (params.embed[ids]
            + signed_log(values)[:, None] * params.w_val[None, :]
            + params.b_val[None, :])


def keys_values(params: SetAttentionParams, ids: np.ndarray, values: np.ndarray):
    """K [T, dim] and V [T, dim] for a token set."""
    x = token_vectors(params, ids, values)
    return x @ params.W_k, x @ params.W_v


def full_set_attention(params: SetAttentionParams, ids: np.ndarray, values: np.ndarray,
                       query: np.ndarray = None):
    """
    Full (unrestricted) softmax attention of one query over the whole set.

    Returns (out [dim], weights [T]): the attended output and the attention
    weights over the set's tokens. This is the quantity the cache's completed
    canonical pass must reproduce exactly.
    """
    if len(ids) == 0:
        raise ValueError("empty token set")
    q = params.q_seed if query is None else query
    K, V = keys_values(params, ids, values)
    scores = (K @ q) / np.sqrt(params.dim)
    scores = scores - scores.max()          # stabilized, same as the cache's running max
    w = np.exp(scores)
    w = w / w.sum()
    return w @ V, w