"""
Track B model: set attention encoder with masked value regression.

Token representation (matches the measured form and the provisional's
Section 11): x_i = E[id_i] + g(v_i) * w + b, with g the signed log
compression. On masked positions the value term g(v) * w is replaced by
a trainable mask vector, the identity stays visible, and the model
regresses g(v) from the rest of the set. No positional encoding: a
filing is a set, order carries no information, and attention is fully
symmetric across positions. Padding is handled by key padding masks and
excluded from the loss.

Id conventions (the dataset module enforces these):
  real identities      0 .. vocab_size - 2
  OOV shared identity  vocab_size - 1
  padding              vocab_size (embedding row exists, never trained)
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def signed_log(v: torch.Tensor) -> torch.Tensor:
    """g(v) = sign(v) * log(1 + |v|), elementwise."""
    return torch.sign(v) * torch.log1p(torch.abs(v))


@dataclass
class SetModelConfig:
    vocab_size: int          # real identities + 1 OOV slot (pad row added internally)
    dim: int = 256
    n_layers: int = 4
    n_heads: int = 4
    ffn_mult: int = 4
    dropout: float = 0.1
    huber_delta: float = 1.0

    @property
    def pad_id(self) -> int:
        return self.vocab_size


class TokenEmbed(nn.Module):
    """E[id] + g(v) * w + b, with a trainable mask vector on masked slots."""

    def __init__(self, cfg: SetModelConfig):
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab_size + 1, cfg.dim, padding_idx=cfg.pad_id)
        self.w_val = nn.Parameter(torch.zeros(cfg.dim))
        self.b_val = nn.Parameter(torch.zeros(cfg.dim))
        self.mask_vec = nn.Parameter(torch.zeros(cfg.dim))
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.normal_(self.w_val, std=0.02)
        nn.init.normal_(self.mask_vec, std=0.02)

    def forward(self, ids, values, mask_flags):
        # ids [B, L] long | values [B, L] float (raw) | mask_flags [B, L] bool
        g = signed_log(values).unsqueeze(-1)                    # [B, L, 1]
        value_term = g * self.w_val                             # [B, L, dim]
        value_term = torch.where(mask_flags.unsqueeze(-1),
                                 self.mask_vec.expand_as(value_term),
                                 value_term)
        return self.embed(ids) + value_term + self.b_val


class SetAttentionBlock(nn.Module):
    """Pre norm transformer block, batch first, no positional information."""

    def __init__(self, cfg: SetModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.dim)
        self.attn = nn.MultiheadAttention(cfg.dim, cfg.n_heads,
                                          dropout=cfg.dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(cfg.dim)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.dim, cfg.ffn_mult * cfg.dim),
            nn.GELU(),
            nn.Linear(cfg.ffn_mult * cfg.dim, cfg.dim),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x, pad_mask):
        # pad_mask [B, L] bool, True = padding (ignored by attention)
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, key_padding_mask=pad_mask, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class SetAttentionEncoder(nn.Module):
    """Masked value pretraining model: encoder plus value regression head."""

    def __init__(self, cfg: SetModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embed = TokenEmbed(cfg)
        self.blocks = nn.ModuleList(SetAttentionBlock(cfg) for _ in range(cfg.n_layers))
        self.ln_out = nn.LayerNorm(cfg.dim)
        self.value_head = nn.Linear(cfg.dim, 1)

    def forward(self, ids, values, mask_flags, pad_mask):
        """
        ids [B, L] long | values [B, L] float raw | mask_flags [B, L] bool
        (True = value hidden, predict it) | pad_mask [B, L] bool (True = pad).
        Returns predicted g(v) per position, [B, L].
        """
        x = self.token_embed(ids, values, mask_flags)
        for block in self.blocks:
            x = block(x, pad_mask)
        return self.value_head(self.ln_out(x)).squeeze(-1)

    def encode(self, ids, values, pad_mask):
        """Token representations with nothing masked (for Track C and sweeps)."""
        flags = torch.zeros_like(ids, dtype=torch.bool)
        x = self.token_embed(ids, values, flags)
        for block in self.blocks:
            x = block(x, pad_mask)
        return self.ln_out(x)

    def masked_value_loss(self, pred, values, mask_flags, pad_mask):
        """Huber loss in signed log space over masked, nonpad positions only."""
        target = signed_log(values)
        active = mask_flags & ~pad_mask
        if not active.any():
            return pred.sum() * 0.0
        return F.huber_loss(pred[active], target[active],
                            delta=self.cfg.huber_delta)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
        

class AttentionReadout(nn.Module):
    """Fixed query attention readout over frozen token representations.

    The operator of the provisional's Section 2 applied to the encoder's
    outputs: per token score z_i = (W_k h_i) dot q / sqrt(d_attn), softmax
    weights over the set, output = weighted sum of W_v h_i. At inference
    the scores and values are fixed once per filing, so subset sweeps run
    the cache's running softmax (m, D, U) states exactly as benchmarked
    in Track A; token_scores_values() exposes z_i and V_i for the sweeps.
    """

    def __init__(self, dim: int, d_attn: int | None = None):
        super().__init__()
        d_attn = d_attn or dim
        self.scale = d_attn ** 0.5
        self.q = nn.Parameter(torch.zeros(d_attn))
        self.W_k = nn.Linear(dim, d_attn, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False)
        nn.init.normal_(self.q, std=0.02)

    def scores(self, h: torch.Tensor) -> torch.Tensor:
        """Per token scores z_i, [..., L]."""
        return (self.W_k(h) @ self.q) / self.scale

    def forward(self, h: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        # h [B, L, dim] | pad_mask [B, L] bool, True = padding
        z = self.scores(h).masked_fill(pad_mask, float("-inf"))
        w = torch.softmax(z.float(), dim=-1).to(h.dtype)      # [B, L]
        return (w.unsqueeze(-1) * self.W_v(h)).sum(dim=1)     # [B, dim]

    def token_scores_values(self, h: torch.Tensor):
        """For cache sweeps over one filing: h [L, dim] (no padding) ->
        (z [L], V [L, dim]), the fixed per token inputs to the running
        softmax subset states."""
        return self.scores(h), self.W_v(h)
        
class HistoricalSetEncoder(nn.Module):
    """Candidate 2 encoder: the pretrained SetAttentionEncoder applied to a
    HISTORICAL WINDOW token set (anchor filing plus up to max_prior prior
    filings, composed by FilingHistoryCube), with three additive temporal
    coordinates per token:

        x_i = TokenEmbed(id_i, v_i) + G[group_i] + F[form_i] + log1p(age_i) * w_age

    group_i = availability group (0 = anchor, 1 = most recent prior DATE, ...).
              Within day order carries NO coordinate, matching the cube's
              availability policy.
    form_i  = form type id under a SAVED caller owned mapping with an explicit
              unknown id. Save form_map, n_groups, n_forms with any checkpoint.
    age_i   = calendar days from the anchor's FileDate to the token's filing
              date (0 on anchor tokens), continuous, log compressed.

    Zero initialization of the temporal parameters preserves the BASE
    COMPUTATION: an anchor only window in evaluation mode reproduces the base
    encoder's representations within numerical tolerance. A window WITH
    history changes self attention and the readout from the FIRST forward
    pass; what the gradients decide is only the learned content of the
    temporal coordinates. No bit identical claim holds in training mode
    (dropout). MATCHED CONTROL RULE: the current only control uses this same
    class and initialization, receiving anchor tokens with group 0 and age 0
    and its REAL form id, so form information is common to both arms and any
    difference is attributable to history, not to the added coordinates.

    Coordinates are VALIDATED, never clamped: groups in [0, n_groups), forms
    in [0, n_forms), ages finite and nonnegative, or encode raises.

    Cache note: the readout consumes fixed per token representations, so the
    Track A machinery applies unchanged; a token occurrence's cache identity
    is (namespace, source_filing, feature_id), with the anchor, window, and
    model version defining the cache scope. Padding positions should receive
    group 0, form 0, age 0; they are excluded by the key padding mask and the
    readout regardless.
    """

    def __init__(self, base: SetAttentionEncoder, n_groups: int = 8,
                 n_forms: int = 16):
        super().__init__()
        self.base = base
        self.n_groups = n_groups
        self.n_forms = n_forms
        dim = base.cfg.dim
        self.group_embed = nn.Embedding(n_groups, dim)
        self.form_embed = nn.Embedding(n_forms, dim)
        self.w_age = nn.Parameter(torch.zeros(dim))
        nn.init.zeros_(self.group_embed.weight)
        nn.init.zeros_(self.form_embed.weight)

    def temporal_parameters(self):
        """The new parameters (full LR group in fine tunes, vs 0.1x on base)."""
        return [self.group_embed.weight, self.form_embed.weight, self.w_age]

    def encode(self, ids, values, pad_mask, groups, ages_days, forms):
        """
        ids [B, L] long | values [B, L] float raw | pad_mask [B, L] bool
        (True = pad) over the FLATTENED window token set | groups [B, L] long |
        ages_days [B, L] float | forms [B, L] long.
        Returns token representations [B, L, dim].
        """
        if bool((groups < 0).any()) or bool((groups >= self.n_groups).any()):
            raise ValueError(f"groups must lie in [0, {self.n_groups})")
        if bool((forms < 0).any()) or bool((forms >= self.n_forms).any()):
            raise ValueError(f"forms must lie in [0, {self.n_forms}); map unknown "
                             f"forms to the explicit unknown id, never clamp")
        if not bool(torch.isfinite(ages_days).all()) or bool((ages_days < 0).any()):
            raise ValueError("ages_days must be finite and nonnegative")
        flags = torch.zeros_like(ids, dtype=torch.bool)
        x = self.base.token_embed(ids, values, flags)
        x = x + self.group_embed(groups)
        x = x + self.form_embed(forms)
        x = x + torch.log1p(ages_days.float()).unsqueeze(-1) * self.w_age
        for block in self.base.blocks:
            x = block(x, pad_mask)
        return self.base.ln_out(x)
        
class PhraseAugmentedEncoder(nn.Module):
    """Full Feature Space arms A/B/C in one class: the pretrained
    SetAttentionEncoder over GAAP tokens, optionally joined by a phrase COUNT
    token and by SAMPLED SEMANTIC PHRASE tokens carrying precomputed MiniLM
    vectors through a learned adapter.

    Arm semantics (one class, three configurations, so shared initialization
    across arms is trivial):
      A: phrases=None, counts=None   (the incumbent current filing model)
      C: phrases=None, counts given  (count token only)
      B: phrases given, counts given (count token + sampled phrase tokens)
    Phrase vectors WITHOUT counts are rejected: no declared arm uses them.

    Token construction:
      GAAP token    x_i = TokenEmbed(id_i, v_i)                (pretrained)
      count token   x_c = c_vec + g(n_total) * w_count         (zero init;
        g = signed_log, which is log1p for nonnegative counts)
      phrase token  x_p = W_adapt(m_p) + t_phrase
        m_p is the phrase occurrence's MiniLM vector (384), W_adapt a learned
        Linear(384 -> dim) at std 0.02, t_phrase a zero initialized offset.

    Execution contract: intended for torch.amp autocast. Outside autocast,
    phrase vectors are cast to the adapter weight dtype so float16 storage
    never meets float32 weights uncast.

    Honesty notes, binding on any write up: arms share INITIALIZATION, not
    initial computation; count and phrase tokens participate in attention
    normalization and the readout from the first forward pass. counts must be
    the TOTAL occurrence count (n_total from the TT3 manifest), the sampled
    count recorded separately; TT3 incomplete anchors are excluded from all
    arms. Arm A equivalence to base.encode() is verified empirically in TT4,
    not assumed. Cache note: after encoding, every token has a fixed
    representation, so the Track A readout and sweep machinery apply; a
    phrase token's cache identity is (phrase namespace, source filing, block
    index, position in block), scoped by anchor, sampled token set, and model
    version.
    """

    def __init__(self, base: SetAttentionEncoder, phrase_dim: int = 384):
        super().__init__()
        self.base = base
        self.phrase_dim = phrase_dim
        dim = base.cfg.dim
        self.adapter = nn.Linear(phrase_dim, dim, bias=False)
        nn.init.normal_(self.adapter.weight, std=0.02)
        self.t_phrase = nn.Parameter(torch.zeros(dim))
        self.c_vec = nn.Parameter(torch.zeros(dim))
        self.w_count = nn.Parameter(torch.zeros(dim))

    def new_parameters(self):
        """Parameters absent from the pretrained base (full LR group)."""
        return [self.adapter.weight, self.t_phrase, self.c_vec, self.w_count]

    def encode(self, ids, values, pad_mask, phrase_vecs=None,
               phrase_pad=None, counts=None):
        """
        ids [B, Lg] long | values [B, Lg] float raw | pad_mask [B, Lg] bool
        (True = pad) | phrase_vecs [B, Lp, 384] float or None | phrase_pad
        [B, Lp] bool or None | counts [B] float (TOTAL occurrences) or None.
        Returns (token representations [B, L, dim], combined pad mask [B, L]),
        tokens ordered [gaap | count | phrases].
        """
        if ids.ndim != 2 or ids.shape != values.shape \
                or ids.shape != pad_mask.shape:
            raise ValueError("ids, values, pad_mask must share one 2D shape")
        if not (ids.dtype in (torch.int64, torch.int32)):
            raise ValueError("ids must be integer indices")
        if pad_mask.dtype != torch.bool:
            raise ValueError("pad_mask must be boolean")
        dev = ids.device
        if values.device != dev or pad_mask.device != dev:
            raise ValueError("ids, values, pad_mask must share one device")
        if phrase_pad is not None and phrase_vecs is None:
            raise ValueError("phrase_pad supplied without phrase_vecs")
        if phrase_vecs is not None and counts is None:
            raise ValueError("phrase_vecs without counts matches no declared arm")

        B = ids.shape[0]
        flags = torch.zeros_like(ids, dtype=torch.bool)
        parts = [self.base.token_embed(ids, values, flags)]
        masks = [pad_mask]
        if counts is not None:
            if counts.shape != (B,) or counts.device != dev:
                raise ValueError("counts must be shape [B] on the same device")
            if not bool(torch.isfinite(counts).all()) \
                    or bool((counts < 0).any()):
                raise ValueError("counts must be finite and nonnegative")
            g = signed_log(counts).view(B, 1, 1)
            parts.append(self.c_vec + g * self.w_count)
            masks.append(torch.zeros(B, 1, dtype=torch.bool, device=dev))
        if phrase_vecs is not None:
            if phrase_pad is None or phrase_pad.dtype != torch.bool:
                raise ValueError("phrase_pad must be a boolean mask")
            if (phrase_vecs.ndim != 3
                    or phrase_vecs.shape[-1] != self.phrase_dim
                    or phrase_vecs.shape[:2] != phrase_pad.shape
                    or phrase_vecs.shape[0] != B):
                raise ValueError("phrase_vecs/phrase_pad shapes inconsistent")
            if not phrase_vecs.is_floating_point():
                raise ValueError("phrase_vecs must be floating point")
            if phrase_vecs.device != dev or phrase_pad.device != dev:
                raise ValueError("phrase inputs must share the batch device")
            if not bool(torch.isfinite(phrase_vecs).all()):
                raise ValueError("phrase_vecs must be finite")
            if not torch.is_autocast_enabled() \
                    and phrase_vecs.dtype != self.adapter.weight.dtype:
                phrase_vecs = phrase_vecs.to(self.adapter.weight.dtype)
            parts.append(self.adapter(phrase_vecs) + self.t_phrase)
            masks.append(phrase_pad)
        x = torch.cat(parts, dim=1)
        full_pad = torch.cat(masks, dim=1)
        if bool(full_pad.all(dim=1).any()):
            raise ValueError("a sample has no unmasked tokens")
        for block in self.base.blocks:
            x = block(x, full_pad)
        return self.base.ln_out(x), full_pad
        
class SemanticGaapEncoder(nn.Module):
    """Arm F pathway: GAAP tokens constructed from concept NAME meaning vectors
    instead of the learned id table, so no vocabulary limits identity coverage.

    Token construction mirrors TokenEmbed exactly except for the identity term:
        token = adapter(name_vec) + signed_log(value) * w_val + b_val
    where TokenEmbed computes E[id] + signed_log(value) * w + b. The transformer
    blocks and final layer norm are the wrapped base encoder's own modules.

    Shared initialization honesty: arm F does NOT start computationally equal to
    the base model (the identity pathway differs by construction). What is shared
    with the other arms: the pretrained blocks and ln_out via the base, and the
    VALUE channel, whose w_val and b_val are COPIED from the base token_embed at
    construction, so values enter arm F exactly as the pretrained model reads
    them. The adapter is new (std 0.02). State this in any findings.

    Arm S (shuffled names control) is a DATA level construction: the same class
    fed a fixed seed permutation of the name vector table, so meaning is
    scrambled while token counts, values, and architecture stay identical.

    Inputs to encode(): name_vecs [B, L, name_dim] floating (float16 accepted,
    cast outside autocast), values [B, L] float, pad [B, L] bool (True = padded).
    Returns (representations [B, L, dim], pad) after the base blocks and ln_out.
    """

    def __init__(self, base: SetAttentionEncoder, name_dim: int = 384):
        super().__init__()
        self.base = base
        self.name_dim = name_dim
        dim = base.cfg.dim
        self.adapter = nn.Linear(name_dim, dim, bias=False)
        nn.init.normal_(self.adapter.weight, std=0.02)
        self.w_val = nn.Parameter(base.token_embed.w_val.detach().clone())
        self.b_val = nn.Parameter(base.token_embed.b_val.detach().clone())

    def new_parameters(self):
        return [self.adapter.weight, self.w_val, self.b_val]

    def encode(self, name_vecs, values, pad):
        if name_vecs.dim() != 3 or name_vecs.shape[-1] != self.name_dim:
            raise ValueError(f"name_vecs must be [B, L, {self.name_dim}], "
                             f"got {tuple(name_vecs.shape)}")
        if not torch.is_floating_point(name_vecs):
            raise ValueError("name_vecs must be floating point")
        B, L, _ = name_vecs.shape
        if values.shape != (B, L) or pad.shape != (B, L):
            raise ValueError("values and pad must both be [B, L]")
        if pad.dtype != torch.bool:
            raise ValueError("pad must be bool")
        if not (values.device == name_vecs.device == pad.device):
            raise ValueError("inputs on different devices")
        if bool(pad.all(dim=1).any()):
            raise ValueError("every sample needs at least one unmasked token")
        nv = name_vecs.to(self.adapter.weight.dtype)   # cast outside autocast
        if not torch.isfinite(nv).all():
            raise ValueError("non finite name vectors")
        x = self.adapter(nv) + signed_log(values).unsqueeze(-1) * self.w_val \
            + self.b_val
        for block in self.base.blocks:
            x = block(x, pad)
        return self.base.ln_out(x), pad
        
class SemanticMaskedSetEncoder(nn.Module):
    """Thesis Validation pretrain arm P_sem: masked value pretraining FROM
    SCRATCH with token identity carried by concept NAME meaning vectors
    through a learned adapter, no vocabulary and no frequency based identity
    exclusion.

    Token construction mirrors TokenEmbed with the identity term swapped:
        x_i = adapter(name_vec_i) + g(v_i) * w_val + b_val
    and on masked positions the value term g(v) * w_val is replaced by a
    trainable mask vector, identity stays visible, and the model regresses
    g(v) from the rest of the set: the same objective, substitution rule,
    head, and loss as SetAttentionEncoder, verified against that class at
    source. Initialization follows TokenEmbed's scheme (adapter, w_val and
    mask_vec at std 0.02, b_val zeros); blocks, ln_out and value_head are
    constructed fresh. cfg.vocab_size is UNUSED by this class (no identity
    table exists); pass the matched control's cfg for dim and layer parity
    and state that in the run record.

    Matched arm honesty: the two pretrain arms share corpus, sampled token
    positions, masked positions, schedule, architecture dimensions, AND the
    initialization of every common shaped parameter: the training cell
    initializes the vocabulary arm first and copies its blocks, ln_out,
    value_head, w_val, b_val, and mask_vec into this class. Only the identity
    pathway differs at initialization (embedding table vs adapter), which is
    the thing under test. State this in any findings.
    """

    def __init__(self, cfg: SetModelConfig, name_dim: int = 384):
        super().__init__()
        self.cfg = cfg
        self.name_dim = name_dim
        dim = cfg.dim
        self.adapter = nn.Linear(name_dim, dim, bias=False)
        self.w_val = nn.Parameter(torch.zeros(dim))
        self.b_val = nn.Parameter(torch.zeros(dim))
        self.mask_vec = nn.Parameter(torch.zeros(dim))
        nn.init.normal_(self.adapter.weight, std=0.02)
        nn.init.normal_(self.w_val, std=0.02)
        nn.init.normal_(self.mask_vec, std=0.02)
        self.blocks = nn.ModuleList(SetAttentionBlock(cfg)
                                    for _ in range(cfg.n_layers))
        self.ln_out = nn.LayerNorm(dim)
        self.value_head = nn.Linear(dim, 1)

    def _embed(self, name_vecs, values, mask_flags):
        if name_vecs.dim() != 3 or name_vecs.shape[-1] != self.name_dim:
            raise ValueError(f"name_vecs must be [B, L, {self.name_dim}], "
                             f"got {tuple(name_vecs.shape)}")
        if not torch.is_floating_point(name_vecs):
            raise ValueError("name_vecs must be floating point")
        B, L, _ = name_vecs.shape
        if values.shape != (B, L) or mask_flags.shape != (B, L):
            raise ValueError("values and mask_flags must both be [B, L]")
        if not torch.is_floating_point(values):
            raise ValueError("values must be floating point")
        if mask_flags.dtype != torch.bool:
            raise ValueError("mask_flags must be bool")
        if not (values.device == name_vecs.device == mask_flags.device):
            raise ValueError("inputs on different devices")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("non finite values")
        nv = name_vecs
        if not torch.is_autocast_enabled() \
                and nv.dtype != self.adapter.weight.dtype:
            nv = nv.to(self.adapter.weight.dtype)
        if not bool(torch.isfinite(nv).all()):
            raise ValueError("non finite name vectors")
        g = signed_log(values).unsqueeze(-1)
        value_term = g * self.w_val
        value_term = torch.where(mask_flags.unsqueeze(-1),
                                 self.mask_vec.expand_as(value_term),
                                 value_term)
        return self.adapter(nv) + value_term + self.b_val

    def _check_pad(self, values, pad_mask):
        if pad_mask.dtype != torch.bool or pad_mask.shape != values.shape:
            raise ValueError("pad_mask must be bool with shape [B, L]")
        if pad_mask.device != values.device:
            raise ValueError("pad_mask on a different device")
        if bool(pad_mask.all(dim=1).any()):
            raise ValueError("every sample needs at least one unpadded token")

    def forward(self, name_vecs, values, mask_flags, pad_mask):
        """Returns predicted g(v) per position, [B, L], exactly as
        SetAttentionEncoder.forward does for the vocabulary arm."""
        self._check_pad(values, pad_mask)
        x = self._embed(name_vecs, values, mask_flags)
        for block in self.blocks:
            x = block(x, pad_mask)
        return self.value_head(self.ln_out(x)).squeeze(-1)

    def encode(self, name_vecs, values, pad_mask):
        """Token representations with nothing masked (for fine tunes and
        sweeps), mirroring SetAttentionEncoder.encode."""
        self._check_pad(values, pad_mask)
        flags = torch.zeros(values.shape, dtype=torch.bool,
                            device=values.device)
        x = self._embed(name_vecs, values, flags)
        for block in self.blocks:
            x = block(x, pad_mask)
        return self.ln_out(x)

    def masked_value_loss(self, pred, values, mask_flags, pad_mask):
        """Huber loss in signed log space over masked, nonpad positions only;
        identical to SetAttentionEncoder.masked_value_loss. Padded positions
        are excluded from the active target set by construction."""
        target = signed_log(values)
        active = mask_flags & ~pad_mask
        if not active.any():
            return pred.sum() * 0.0
        return F.huber_loss(pred[active], target[active],
                            delta=self.cfg.huber_delta)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
        
class MaskedSetAttentionBlock(SetAttentionBlock):
    """SetAttentionBlock's parameters with attention computed EXPLICITLY under
    an arbitrary boolean mask (True = forbidden), shape [T, T] or
    [B, 1, T, T] (broadcast over heads). nn.MultiheadAttention's own forward
    was measured nonreproducible and inexact for arbitrary boolean masks with
    gradients on (TVM1e: torch 2.5.1, CPU, float64), so it is bypassed.
    Attention weight dropout mirrors MHA's placement, training mode only."""

    def forward(self, x, attn_mask):
        if attn_mask.dtype != torch.bool:
            raise ValueError("attn_mask must be bool (True = forbidden)")
        h = self.ln1(x)
        B, T, d = h.shape
        nh = self.attn.num_heads
        dh = d // nh
        qkv = F.linear(h, self.attn.in_proj_weight, self.attn.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, nh, dh).transpose(1, 2)
        k = k.view(B, T, nh, dh).transpose(1, 2)
        v = v.view(B, T, nh, dh).transpose(1, 2)
        sc = (q @ k.transpose(-2, -1)) / dh ** 0.5
        sc = sc.masked_fill(attn_mask, float("-inf"))
        w = torch.softmax(sc, dim=-1)
        if self.training and self.attn.dropout > 0:
            w = F.dropout(w, p=self.attn.dropout)
        o = (w @ v).transpose(1, 2).reshape(B, T, d)
        x = x + self.attn.out_proj(o)
        return x + self.mlp(self.ln2(x))


def causal_pad_forbid(pad_mask):
    """[B, 1, L, L] forbidden mask for left aligned padded batches: causal in
    canonical order plus key padding. Position 0 must be real in every row."""
    if bool(pad_mask[:, 0].any()):
        raise ValueError("position 0 must be a real token in every row")
    L = pad_mask.shape[1]
    causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=pad_mask.device),
                        diagonal=1)
    return causal[None, None] | pad_mask[:, None, None, :]


class CausalSemanticSetEncoder(SemanticMaskedSetEncoder):
    """Thesis Validation Medium encoder: SemanticMaskedSetEncoder's semantic
    embedding (inherited, not copied) with CAUSAL attention in canonical
    order (ascending registry column id): a token attends to itself and the
    tokens before it, so its state at every layer depends only on its
    canonical prefix. Subsets sharing a canonical prefix therefore share
    every layer's states for that prefix, which the trie pass computes once.

    Two execution paths, verified equivalent in TVM1R:
      forward / encode: padded batches, one causal sequence per row
        (the uncached reference).
      trie_layers / trie_forward: one call over a prefix trie (from
        set_attention_cache.build_prefix_trie) with an ancestor mask.
    Recorded change from SEM: SEM's encoder is bidirectional."""

    def __init__(self, cfg: SetModelConfig, name_dim: int = 384):
        super().__init__(cfg, name_dim)
        self.blocks = nn.ModuleList(MaskedSetAttentionBlock(cfg)
                                    for _ in range(cfg.n_layers))

    def forward(self, name_vecs, values, mask_flags, pad_mask):
        self._check_pad(values, pad_mask)
        x = self._embed(name_vecs, values, mask_flags)
        forbid = causal_pad_forbid(pad_mask)
        for block in self.blocks:
            x = block(x, forbid)
        return self.value_head(self.ln_out(x)).squeeze(-1)

    def encode(self, name_vecs, values, pad_mask):
        self._check_pad(values, pad_mask)
        flags = torch.zeros(values.shape, dtype=torch.bool, device=values.device)
        x = self._embed(name_vecs, values, flags)
        forbid = causal_pad_forbid(pad_mask)
        for block in self.blocks:
            x = block(x, forbid)
        return self.ln_out(x)

    def trie_layers(self, name_vecs, values, mask_flags, attn_forbid):
        """name_vecs [T, name_dim], values [T], mask_flags [T] bool over trie
        nodes; attn_forbid [T, T] bool (True = not an ancestor or self).
        Returns [embedding, block 1, ..., block L, ln_out], each [T, dim]."""
        x = self._embed(name_vecs.unsqueeze(0), values.unsqueeze(0),
                        mask_flags.unsqueeze(0))
        outs = [x.squeeze(0)]
        for block in self.blocks:
            x = block(x, attn_forbid)
            outs.append(x.squeeze(0))
        outs.append(self.ln_out(x).squeeze(0))
        return outs

    def trie_forward(self, name_vecs, values, mask_flags, attn_forbid):
        """Value predictions per trie node, [T]."""
        H = self.trie_layers(name_vecs, values, mask_flags, attn_forbid)[-1]
        return self.value_head(H).squeeze(-1)


def trie_readout(readout, H, parent, terminals):
    """Fixed query readout over each requested trie path, accumulated with
    running prefix states (m, s, u) along the trie, processed level by level
    (all nodes at one depth together). Every operation is out of place, so
    the computation graph and gradients flow through shared prefixes.
    H [T, dim] final node states; parent [T] numpy (-1 = root); terminals =
    terminal node per path. Returns pooled outputs [len(terminals), dim]."""
    import numpy as _np
    parent = _np.asarray(parent)
    T = len(parent)
    depth = _np.zeros(T, dtype=_np.int64)
    for i in range(T):
        depth[i] = 0 if parent[i] == -1 else depth[parent[i]] + 1
    z = readout.scores(H)
    V = readout.W_v(H)
    m, s, u = z, torch.ones_like(z), V
    for d in range(1, int(depth.max()) + 1 if T else 1):
        idx = _np.flatnonzero(depth == d)
        it = torch.as_tensor(idx, device=H.device)
        pt = torch.as_tensor(parent[idx], device=H.device)
        mp, sp, up = m[pt], s[pt], u[pt]
        zi = z[it]
        mn = torch.maximum(mp, zi)
        a_old, a_new = torch.exp(mp - mn), torch.exp(zi - mn)
        m = m.index_put((it,), mn)
        s = s.index_put((it,), sp * a_old + a_new)
        u = u.index_put((it,), up * a_old[:, None] + a_new[:, None] * V[it])
    tt = torch.as_tensor(_np.asarray(terminals), device=H.device)
    return u[tt] / s[tt][:, None]
    
def trie_encode_batched(enc, name_vecs, values, mask_flags, attn_forbid):
    """Batched trie pass for CausalSemanticSetEncoder: name_vecs [B, T, name_dim],
    values [B, T], mask_flags [B, T] bool, attn_forbid [B, 1, T, T] bool from
    set_attention_cache.pack_tries. Pad nodes must carry zero name vectors,
    zero values, and False flags; callers exclude them from every loss.
    Returns final node states [B, T, dim]."""
    if name_vecs.dim() != 3:
        raise ValueError(f"name_vecs must be [B, T, name_dim], got {tuple(name_vecs.shape)}")
    B, T, _ = name_vecs.shape
    if attn_forbid.dtype != torch.bool or tuple(attn_forbid.shape) != (B, 1, T, T):
        raise ValueError(f"attn_forbid must be bool [{B}, 1, {T}, {T}], got "
                         f"{attn_forbid.dtype} {tuple(attn_forbid.shape)}")
    x = enc._embed(name_vecs, values, mask_flags)
    for block in enc.blocks:
        x = block(x, attn_forbid)
    return enc.ln_out(x)
    
def trie_prefix_states(z, V, parent):
    """Prefix softmax states for every trie node by pointer jumping.
    z [T] scores, V [T, d] values, parent [T] numpy (-1 = root). Node v's final
    state (m, s, u) covers its whole root path: m = max score,
    s = sum exp(z - m), u = sum exp(z - m) * V.
    Invariant: after each round, v's state covers the segment (J[v], v]; merging
    it with J[v]'s state (online softmax combine, associative) doubles the
    segment, then J[v] = J[J[v]]. Roots point at an identity row
    (m = -inf, s = 0, u = 0), so every node updates every round with no
    index_put and no special cases. Sequential rounds = ceil(log2(depth)).
    Every node's state is computed once and shared by all paths through it.
    Fully out of place, so gradients flow through the sharing."""
    import numpy as _np
    T = len(parent)
    dev = z.device
    J = _np.asarray(parent, dtype=_np.int64).copy()
    J[J < 0] = T                                        # identity row index
    pad_m = torch.full((1,), float("-inf"), dtype=z.dtype, device=dev)
    pad_s = torch.zeros(1, dtype=z.dtype, device=dev)
    pad_u = torch.zeros((1, V.shape[1]), dtype=V.dtype, device=dev)
    m, s, u = z, torch.ones_like(z), V
    while (J < T).any():
        Jt = torch.as_tensor(J, device=dev)
        mj = torch.cat([m, pad_m])[Jt]
        sj = torch.cat([s, pad_s])[Jt]
        uj = torch.cat([u, pad_u])[Jt]
        mn = torch.maximum(m, mj)
        av, aj = torch.exp(m - mn), torch.exp(mj - mn)
        m, s, u = mn, s * av + sj * aj, u * av[:, None] + uj * aj[:, None]
        J = _np.append(J, T)[J]
    return m, s, u


def trie_readout_jump(readout, H, parent, terminals):
    """trie_readout computed with trie_prefix_states (log depth rounds).
    Same inputs and outputs as trie_readout: pooled outputs [len(terminals), dim]."""
    import numpy as _np
    z = readout.scores(H)
    V = readout.W_v(H)
    m, s, u = trie_prefix_states(z, V, parent)
    tt = torch.as_tensor(_np.asarray(terminals), device=H.device)
    return u[tt] / s[tt][:, None]
    
def encoder_layer_kv(enc, name_vecs, values, pad_mask):
    """Full set KV caches from SemanticMaskedSetEncoder (bidirectional, padding
    mask only, nothing masked). Runs the encoder's blocks unchanged and records,
    for every block, that block's OWN attention keys and values for every fact,
    computed exactly as the block computes them (ln1, then in_proj). The final
    block's output is never used, so its forward is skipped.
    Blocks advance through MaskedSetAttentionBlock's explicit attention with a
    key padding mask, NOT the block's nn.MultiheadAttention forward: with
    gradients on, that forward was measured inexact (differences up to 1e6)
    and nonreproducible (TVMF3f, torch 2.5.1, CPU, float64), as in TVM1e. The
    explicit path equals the block's own forward with gradients off (TVMF3e).
    Returns a list over layers of (K, V), each [B, L, n_heads, d_head]."""
    enc._check_pad(values, pad_mask)
    flags = torch.zeros(values.shape, dtype=torch.bool, device=values.device)
    x = enc._embed(name_vecs, values, flags)
    key_forbid = pad_mask[:, None, None, :]   # [B, 1, 1, L]: padded keys forbidden
    kv = []
    last = len(enc.blocks) - 1
    for i, block in enumerate(enc.blocks):
        h = block.ln1(x)
        B, L, d = h.shape
        nh = block.attn.num_heads
        dh = d // nh
        _, k, v = F.linear(h, block.attn.in_proj_weight, block.attn.in_proj_bias).chunk(3, dim=-1)
        kv.append((k.view(B, L, nh, dh), v.view(B, L, nh, dh)))
        if i < last:                      # the final block's output is never used
            x = MaskedSetAttentionBlock.forward(block, x, key_forbid)
    return kv


class SubsetInteractionStack(nn.Module):
    """Interaction stack over the encoder's per layer, per head KV caches.
    Learned queries q[l, h, j] (independent of the subset) attend with full
    attention over a subset's facts using layer l, head h keys and values, so
    every score and value is fixed per fact and subset results can be built
    from shared prefix states. Channels C = n_layers * n_heads * n_queries,
    ordered (layer, head, query). A subset's representation is the
    concatenation of its C attention results, then LayerNorm + MLP -> scalar."""

    def __init__(self, n_layers, n_heads, d_head, n_queries=1, hidden=256):
        super().__init__()
        self.n_layers, self.n_heads, self.d_head, self.n_queries = n_layers, n_heads, d_head, n_queries
        self.q = nn.Parameter(torch.randn(n_layers, n_heads, n_queries, d_head) * 0.02)
        width = n_layers * n_heads * n_queries * d_head
        self.mlp = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, hidden), nn.GELU(),
                                 nn.Linear(hidden, 1))

    def scores_values(self, kv):
        """kv: list over layers of (K, V), each [n, n_heads, d_head], for ONE
        filing. Returns per fact scores z [n, C] and values V [n, C, d_head]."""
        if len(kv) != self.n_layers:
            raise ValueError(f"expected {self.n_layers} layers of keys and values, got {len(kv)}")
        zs, vs = [], []
        for l, (K, V) in enumerate(kv):
            n = K.shape[0]
            zl = torch.einsum("nhd,hqd->nhq", K, self.q[l].to(K.dtype)) / self.d_head ** 0.5
            zs.append(zl.reshape(n, -1))
            vs.append(V.unsqueeze(2).expand(-1, -1, self.n_queries, -1).reshape(n, -1, self.d_head))
        return torch.cat(zs, dim=1), torch.cat(vs, dim=1)

    def predict(self, pooled):
        """pooled [P, C, d_head] attention results per subset -> predictions [P]."""
        return self.mlp(pooled.reshape(pooled.shape[0], -1)).squeeze(-1)


def trie_prefix_states_mc(z, V, parent):
    """trie_prefix_states for C channels at once. z [T, C] scores, V [T, C, d]
    values, parent [T] numpy (-1 = root). Same pointer jumping invariant as the
    certified single channel trie_prefix_states: after each round v's state
    covers (J[v], v]; merging with J[v]'s state doubles it; roots point at an
    identity row (m = -inf, s = 0, u = 0). Rounds = ceil(log2(depth)). Every
    node's state is computed once and shared by all paths through it. Fully
    out of place. Returns m [T, C], s [T, C], u [T, C, d]."""
    import numpy as _np
    T = len(parent)
    dev = z.device
    J = _np.asarray(parent, dtype=_np.int64).copy()
    J[J < 0] = T
    pad_m = torch.full((1,) + tuple(z.shape[1:]), float("-inf"), dtype=z.dtype, device=dev)
    pad_s = torch.zeros((1,) + tuple(z.shape[1:]), dtype=z.dtype, device=dev)
    pad_u = torch.zeros((1,) + tuple(V.shape[1:]), dtype=V.dtype, device=dev)
    m, s, u = z, torch.ones_like(z), V
    while (J < T).any():
        Jt = torch.as_tensor(J, device=dev)
        mj = torch.cat([m, pad_m])[Jt]
        sj = torch.cat([s, pad_s])[Jt]
        uj = torch.cat([u, pad_u])[Jt]
        mn = torch.maximum(m, mj)
        av, aj = torch.exp(m - mn), torch.exp(mj - mn)
        m, s, u = mn, s * av + sj * aj, u * av[..., None] + uj * aj[..., None]
        J = _np.append(J, T)[J]
    return m, s, u


def subset_attention_cached(z_pos, V_pos, trie):
    """Full attention of every channel over every trie path, from shared prefix
    states (the invention). z_pos [n, C], V_pos [n, C, d] per fact; trie from
    build_prefix_trie. Returns [P, C, d] in trie path order."""
    import numpy as _np
    pos = torch.as_tensor(trie["pos"], device=z_pos.device)
    m, s, u = trie_prefix_states_mc(z_pos[pos], V_pos[pos], trie["parent"])
    tt = torch.as_tensor(_np.asarray([p[-1] for p in trie["paths"]]), device=z_pos.device)
    return u[tt] / s[tt][..., None]


def subset_attention_uncached(z_pos, V_pos, subsets):
    """The same full attention computed independently for every subset, no
    sharing: subsets of equal length form one dense batch (no padding), and
    each is a direct softmax over its own facts. z_pos [n, C], V_pos [n, C, d];
    subsets = position tuples. Returns [P, C, d] in input order."""
    import numpy as _np
    groups = {}
    for i, S in enumerate(subsets):
        groups.setdefault(len(S), []).append(i)
    outs, order = [], []
    for ids in groups.values():
        idx = torch.as_tensor(_np.asarray([subsets[i] for i in ids], dtype=_np.int64),
                              device=z_pos.device)
        w = torch.softmax(z_pos[idx], dim=1)
        outs.append(torch.einsum("glc,glcd->gcd", w, V_pos[idx]))
        order += ids
    inv = torch.as_tensor(_np.argsort(_np.asarray(order)), device=z_pos.device)
    return torch.cat(outs)[inv]
    
def level_plan(trie, device):
    """Precomputed level schedule for subset_attention_levels (built once per
    filing like the trie itself; includes moving the index tensors to device).
    trie from build_prefix_trie over the SUBSETS ONLY (no full set). Per depth
    level d: the facts at that level, each node's parent as a local index into
    level d - 1, and the paths that end at that level with their terminal's
    local index. Returns a dict with 'levels' (list of dicts of device tensors),
    'inv' (restores trie path order), and 'n_paths'."""
    import numpy as _np
    pos = _np.asarray(trie["pos"], dtype=_np.int64)
    parent = _np.asarray(trie["parent"], dtype=_np.int64)
    depth = _np.asarray(trie["depth"], dtype=_np.int64)
    T = len(pos)
    D = int(depth.max()) + 1
    local = _np.empty(T, dtype=_np.int64)
    by_level = []
    for d in range(D):
        idx = _np.flatnonzero(depth == d)
        local[idx] = _np.arange(len(idx))
        by_level.append(idx)
    term = _np.asarray([p[-1] for p in trie["paths"]], dtype=_np.int64)
    term_depth = depth[term]
    levels, order = [], []
    for d, idx in enumerate(by_level):
        ends = _np.flatnonzero(term_depth == d)
        levels.append({
            "pos": torch.as_tensor(pos[idx], device=device),
            "parent_local": (torch.as_tensor(local[parent[idx]], device=device) if d > 0 else None),
            "end_local": (torch.as_tensor(local[term[ends]], device=device) if len(ends) else None)})
        order.append(ends)
    order = _np.concatenate(order)
    return {"levels": levels, "inv": torch.as_tensor(_np.argsort(order), device=device),
            "n_paths": len(term)}


def subset_attention_levels(z_pos, V_pos, plan):
    """Work efficient cached subset attention (the invention): walks the subset
    trie level by level, computing each node's prefix state (m, s, u) exactly
    once from its parent's state (online softmax, out of place), using compact
    per level states without full trie copies (autograd still retains each
    level's intermediates for backward). Sequential steps = trie depth (at most
    the largest subset size), state updates = trie nodes. z_pos [n, C],
    V_pos [n, C, d] per fact; plan from level_plan. Returns [P, C, d] in trie
    path order."""
    outs = []
    m = s = u = None
    for d, lv in enumerate(plan["levels"]):
        zi, Vi = z_pos[lv["pos"]], V_pos[lv["pos"]]
        if d == 0:
            m, s, u = zi, torch.ones_like(zi), Vi
        else:
            pl = lv["parent_local"]
            mp, sp, up = m[pl], s[pl], u[pl]
            mn = torch.maximum(mp, zi)
            a_old, a_new = torch.exp(mp - mn), torch.exp(zi - mn)
            m, s, u = mn, sp * a_old + a_new, up * a_old[..., None] + a_new[..., None] * Vi
        if lv["end_local"] is not None:
            el = lv["end_local"]
            outs.append(u[el] / s[el][..., None])
    return torch.cat(outs)[plan["inv"]]


def full_and_subsets_attention(z_pos, V_pos, plan):
    """Attention results for [full set] + subsets, full set FIRST (the order
    ret_loss expects). The full set is one direct softmax over every fact; the
    subsets use subset_attention_levels. plan = level_plan of the subset trie,
    or None when there are no subsets (Arm A). Returns [1 + P, C, d]."""
    w = torch.softmax(z_pos, dim=0)
    full = torch.einsum("nc,ncd->cd", w, V_pos)[None]
    if plan is None:
        return full
    return torch.cat([full, subset_attention_levels(z_pos, V_pos, plan)])
    
# ---------------------------------------------------------------------
# Level walk with an exact custom backward: nothing saved per trie node
# ---------------------------------------------------------------------
class _SubsetAttentionLevelsFn(torch.autograd.Function):
    """Exact subset attention over a trie plan with a reverse trie walk backward
    (revision 2: same mathematics as revision 1, fewer output sized buffers and
    fewer operations per level).

    Forward: the online softmax level walk without autograd. Per level, z and V
    of the level's facts are gathered together from ZV = [V | z] (one gather) and
    the parent states (u | m | s) from one state tensor X (one gather; the copy is
    owned, so the update is done in place on it and parent states are never
    modified). Each level's terminal results are written directly into the output
    in final (forest path) order, with M_P and S_P beside them in MS [P, C, 2].
    Saved for backward: ZV [N, C, d+1], the output itself, MS. Nothing is saved
    per trie node.

    Backward, exact (as revision 1): with w_{P,i} = exp(z_i - M_P) / S_P and
    incoming gradient g_P,
        dV_i = sum_{P containing i} w_{P,i} g_P
        dz_i = sum_{P containing i} w_{P,i} (g_P . V_i - g_P . o_P)
    One reverse level walk carries, per node t, mu_t = min of M_P over its
    subtree's terminals and GH_t = [G_t | H_t] with
        G_t = sum exp(mu_t - M_P) g_P / S_P,   H_t = sum exp(mu_t - M_P) (g_P . o_P) / S_P
    (every exponent <= 0); node t holding fact i contributes e = exp(z_i - mu_t),
    dV_i += e G_t, dz_i += e (V_i . G_t - H_t), accumulated into one [N, C, d+1]
    buffer. g_P, o_P, M_P and S_P are read per level for that level's terminals
    only; no output sized scratch tensor is formed. A child's GH is scaled in
    place only in its parent's step, after its own contribution has been used."""

    @staticmethod
    def forward(ctx, z_pos, V_pos, plan):
        levels, inv = plan["levels"], plan["inv"]
        P = inv.numel()
        N, C, D = V_pos.shape
        ZV = torch.cat((V_pos, z_pos.unsqueeze(-1)), dim=-1)                    # [N, C, D+1]: V | z
        out = V_pos.new_empty((P, C, D))
        MS = z_pos.new_empty((P, C, 2))                                          # (M_P, S_P), final order
        dest = torch.empty_like(inv)
        dest[inv] = torch.arange(P, device=inv.device)                           # level order row -> final row
        X, k = None, 0                                                           # X [T, C, D+2]: u | m | s
        for d, lv in enumerate(levels):
            gz = ZV.index_select(0, lv["pos"])                                   # owned [T, C, D+1]
            Vi, zi = gz[..., :D], gz[..., D]
            if d == 0:
                X = torch.cat((gz, torch.ones_like(gz[..., D:D + 1])), dim=-1)   # u = V, m = z, s = 1
            else:
                Xp = X.index_select(0, lv["parent_local"])                       # owned copy of parent states
                up, mp, sp = Xp[..., :D], Xp[..., D], Xp[..., D + 1]
                mn = torch.maximum(mp, zi)
                a = torch.stack((mp, zi)).sub_(mn).exp_()                        # exp(mp - mn), exp(zi - mn)
                sp.mul_(a[0]).add_(a[1])
                up.mul_(a[0].unsqueeze(-1)).addcmul_(a[1].unsqueeze(-1), Vi)
                mp.copy_(mn)
                X = Xp
            el = lv["end_local"]
            if el is not None:
                nt = el.numel()
                rows = dest[k:k + nt]
                k += nt
                Xt = X.index_select(0, el)
                out[rows] = Xt[..., :D] / Xt[..., D + 1:D + 2]
                MS[rows] = Xt[..., D:]
        ctx.save_for_backward(ZV, out, MS)
        ctx.plan, ctx.dest = plan, dest
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, gO):
        ZV, out, MS = ctx.saved_tensors
        plan, dest = ctx.plan, ctx.dest
        levels = plan["levels"]
        N, C, D1 = ZV.shape
        D = D1 - 1
        gZV = torch.zeros_like(ZV)                                               # [N, C, D+1]: dV | dz
        starts, k = [], 0
        for lv in levels:
            starts.append(k)
            k += 0 if lv["end_local"] is None else lv["end_local"].numel()
        child = None                                                             # (parent_local, mu, GH) of level d + 1
        for d in range(len(levels) - 1, -1, -1):
            lv = levels[d]
            pos = lv["pos"]
            T = pos.numel()
            mu = ZV.new_full((T, C), float("inf"))
            el = lv["end_local"]
            if el is not None:
                rows = dest[starts[d]:starts[d] + el.numel()]
                MSt = MS.index_select(0, rows)                                   # [nt, C, 2]
                mu[el] = MSt[..., 0]                                             # at most one terminal per node
            GH = ZV.new_zeros((T, C, D1))                                        # G | H
            if child is not None:
                pl, mu_c, GH_c = child
                mu.scatter_reduce_(0, pl[:, None].expand(-1, C), mu_c, reduce="amin", include_self=True)
                f = (mu.index_select(0, pl) - mu_c).exp_()
                GH.index_add_(0, pl, GH_c.mul_(f.unsqueeze(-1)))                 # child's GH no longer needed
            if el is not None:
                gP = gO.index_select(0, rows)
                oP = out.index_select(0, rows)
                ft = (mu.index_select(0, el) - MSt[..., 0]).exp_().div_(MSt[..., 1])   # exp(mu - M_P) / S_P
                term = torch.cat((gP, (gP * oP).sum(-1, keepdim=True)), dim=-1).mul_(ft.unsqueeze(-1))
                GH.index_add_(0, el, term)
            gz = ZV.index_select(0, pos)                                         # V_i | z_i
            e = (gz[..., D] - mu).exp_()                                         # exp(z_i - mu_t)
            dz_i = ((gz[..., :D] * GH[..., :D]).sum(-1) - GH[..., D]).mul_(e)
            gZV.index_add_(0, pos, torch.cat((GH[..., :D] * e.unsqueeze(-1), dz_i.unsqueeze(-1)), dim=-1))
            child = (lv["parent_local"], mu, GH) if d > 0 else None
        return (gZV[..., D].contiguous() if ctx.needs_input_grad[0] else None,
                gZV[..., :D].contiguous() if ctx.needs_input_grad[1] else None, None)


def subset_attention_levels_exact_bwd(z_pos, V_pos, plan):
    """Same results as subset_attention_levels (same plan, same output order),
    with the exact reverse trie walk backward of _SubsetAttentionLevelsFn:
    memory saved for backward is per fact and per subset, not per trie node."""
    return _SubsetAttentionLevelsFn.apply(z_pos, V_pos, plan)