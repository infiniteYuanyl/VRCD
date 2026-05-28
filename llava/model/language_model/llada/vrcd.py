from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


IMAGE_PLACEHOLDER_ID = -200
IGNORE_INDEX = -100


def normalize_decoding_method(method: Optional[str] = None, remasking: Optional[str] = None) -> str:
    name = str(method or remasking or "confidence").lower()
    aliases = {
        "low_confidence": "confidence",
        "confidence": "confidence",
        "entropy": "entropy",
        "entrophy": "entropy",
        "margin": "margin",
        "vrcd": "vrcd",
        "random": "random",
    }
    if name not in aliases:
        raise NotImplementedError(f"Unsupported decoding method: {method or remasking}")
    return aliases[name]


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if float(temperature) == 0.0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** float(temperature)
    return logits.exp() / gumbel_noise


def compute_candidate_scores(
    logits: torch.Tensor,
    *,
    x_current: torch.Tensor,
    mask_index: torch.Tensor,
    method: str,
    temperature: float,
    block_end: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return predicted tokens and descending-order scores for masked positions."""
    method = normalize_decoding_method(method)
    score_method = "confidence" if method == "vrcd" else method

    logits_with_noise = add_gumbel_noise(logits, temperature=float(temperature))
    x0 = torch.argmax(logits_with_noise, dim=-1)

    if score_method == "confidence":
        probs = F.softmax(logits.to(torch.float64), dim=-1)
        scores = torch.squeeze(torch.gather(probs, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
    elif score_method == "random":
        scores = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    elif score_method == "entropy":
        epsilon = 1e-10
        probs = F.softmax(logits.to(torch.float64), dim=-1)
        log_probs = torch.log(probs + epsilon)
        scores = torch.sum(probs * log_probs, dim=-1)
    elif score_method == "margin":
        probs = F.softmax(logits.to(torch.float64), dim=-1)
        top2_probs = torch.topk(probs, k=2, dim=-1).values
        scores = top2_probs[:, :, 0] - top2_probs[:, :, 1]
    else:
        raise NotImplementedError(score_method)

    scores[:, int(block_end) :] = -float("inf")
    x0 = torch.where(mask_index, x0, x_current)
    scores = torch.where(mask_index, scores, -float("inf"))
    return x0, scores


def positions_topk(score_row: torch.Tensor, candidate_positions: torch.Tensor, k: int) -> torch.Tensor:
    if int(candidate_positions.numel()) == 0 or int(k) <= 0:
        return candidate_positions.new_empty((0,), dtype=torch.long)
    keep = min(int(k), int(candidate_positions.numel()))
    scores = score_row.index_select(0, candidate_positions.long())
    order = torch.argsort(scores, descending=True)
    return candidate_positions.index_select(0, order[:keep])


def percentile_rank(values: torch.Tensor) -> torch.Tensor:
    values = values.float()
    if int(values.numel()) <= 1:
        return torch.zeros_like(values, dtype=torch.float32)
    if float((values.max() - values.min()).abs().item()) <= 1e-8:
        return torch.zeros_like(values, dtype=torch.float32)
    sorted_values, order = torch.sort(values)
    sorted_ranks = torch.empty_like(sorted_values, dtype=torch.float32)
    length = int(sorted_values.numel())
    i = 0
    while i < length:
        j = i + 1
        cur = float(sorted_values[i].item())
        while j < length and float(sorted_values[j].item()) == cur:
            j += 1
        sorted_ranks[i:j] = (0.5 * float(i + j - 1)) / float(max(length - 1, 1))
        i = j
    ranks = torch.empty_like(sorted_ranks, dtype=torch.float32)
    ranks[order] = sorted_ranks
    return ranks


def build_visual_support_distribution(attn_ci: torch.Tensor) -> torch.Tensor:
    size = int(attn_ci.shape[0])
    if size <= 0 or int(attn_ci.shape[-1]) <= 0:
        return torch.zeros((size, 0), dtype=torch.float32, device=attn_ci.device)

    probs = attn_ci.float().clamp_min(0.0)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    uniform = 1.0 / float(max(int(probs.shape[-1]), 1))
    support = torch.relu(probs - uniform)
    mass = support.sum(dim=-1, keepdim=True)
    return torch.where(mass > 1e-8, support / mass.clamp_min(1e-8), torch.zeros_like(support)).float()


def build_visual_redundancy_matrix(attn_ci: torch.Tensor, *, return_visual_mask: bool = False):
    support = build_visual_support_distribution(attn_ci)
    size = int(support.shape[0])
    redundancy = torch.zeros((size, size), dtype=torch.float32, device=support.device)
    visual_mask = support.sum(dim=-1) > 1e-8
    if size <= 1 or int(support.shape[-1]) <= 0:
        return (redundancy, visual_mask) if return_visual_mask else redundancy

    overlap = torch.matmul(torch.sqrt(support.clamp_min(0.0)), torch.sqrt(support.clamp_min(0.0)).transpose(0, 1))
    overlap = torch.clamp(overlap, min=0.0, max=1.0)
    overlap.fill_diagonal_(0.0)

    tri = torch.triu_indices(size, size, offset=1, device=support.device)
    valid_pair = visual_mask.index_select(0, tri[0]) & visual_mask.index_select(0, tri[1])
    valid_pair_ids = torch.nonzero(valid_pair, as_tuple=False).squeeze(-1)
    valid_i = tri[0].index_select(0, valid_pair_ids)
    valid_j = tri[1].index_select(0, valid_pair_ids)
    values = overlap[valid_i, valid_j]
    if int(values.numel()) == 0:
        return (redundancy, visual_mask) if return_visual_mask else redundancy
    ranked = percentile_rank(values)
    redundancy[valid_i, valid_j] = ranked
    redundancy[valid_j, valid_i] = ranked
    return (redundancy, visual_mask) if return_visual_mask else redundancy


def vrcd_confweighted_avg_redundancy(
    confidence: torch.Tensor,
    visual_redundancy: torch.Tensor,
    visual_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    size = int(confidence.numel())
    if size == 0:
        return confidence.new_empty((0,), dtype=torch.float32)
    if size == 1:
        return confidence.new_zeros((1,), dtype=torch.float32)

    conf = confidence.float()
    rel = visual_redundancy.float()
    diag_mask = torch.eye(size, device=rel.device, dtype=torch.bool)
    if visual_mask is None:
        pair_mask = ~diag_mask
    else:
        valid = visual_mask.to(device=rel.device, dtype=torch.bool)
        pair_mask = valid.unsqueeze(0) & valid.unsqueeze(1) & ~diag_mask
    peer_conf = conf.unsqueeze(0).expand(size, -1).clone()
    peer_conf.masked_fill_(~pair_mask, 0.0)
    peer_conf_sum = peer_conf.sum(dim=1, keepdim=True)
    weights = torch.where(peer_conf_sum > 1e-8, peer_conf / peer_conf_sum.clamp_min(1e-8), torch.zeros_like(peer_conf))
    return (weights * rel.masked_fill(~pair_mask, 0.0)).sum(dim=1).float()


def vrcd_scores(
    confidence: torch.Tensor,
    visual_redundancy: torch.Tensor,
    alpha: float,
    visual_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if int(confidence.numel()) <= 1:
        return confidence.float()
    avg_redundancy = vrcd_confweighted_avg_redundancy(confidence, visual_redundancy, visual_mask=visual_mask)
    return confidence.float() * torch.pow(1.0 + avg_redundancy.float(), -float(alpha))


def repeat_kv_to_query_heads(k: torch.Tensor, q_heads: int) -> torch.Tensor:
    if int(k.shape[1]) == int(q_heads):
        return k
    repeat_factor = max(int(q_heads // max(int(k.shape[1]), 1)), 1)
    return k.repeat_interleave(repeat_factor, dim=1, output_size=q_heads)


def attention_probs_for_queries(
    *,
    base_model,
    hidden_states: Sequence[torch.Tensor],
    past_key_values: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]],
    layer_idx: int,
    query_positions: torch.Tensor,
) -> torch.Tensor:
    x = hidden_states[int(layer_idx)]
    block = base_model.transformer.blocks[int(layer_idx)]
    x_norm = block.attn_norm(x)
    if hasattr(block, "att_proj"):
        q, k, _ = block.att_proj(x_norm).split(block.fused_dims, dim=-1)
    else:
        q = block.q_proj(x_norm)
        k = block.k_proj(x_norm)

    if getattr(block, "q_norm", None) is not None and getattr(block, "k_norm", None) is not None:
        q = block.q_norm(q).to(dtype=q.dtype)
        k = block.k_norm(k).to(dtype=k.dtype)

    bsz, seq_len, width = q.shape
    n_heads = int(block.config.n_heads)
    head_dim = width // max(n_heads, 1)
    q = q.view(bsz, seq_len, n_heads, head_dim).transpose(1, 2)
    k = k.view(bsz, seq_len, int(block.config.effective_n_kv_heads), head_dim).transpose(1, 2)

    if past_key_values is not None:
        past_key = past_key_values[int(layer_idx)][0]
        if past_key is not None:
            k = torch.cat((past_key.to(dtype=k.dtype), k), dim=-2)

    k = repeat_kv_to_query_heads(k, int(q.shape[1]))
    if getattr(base_model.config, "rope", False):
        q, k = block.rotary_emb(q, k)
    q = q.index_select(2, query_positions.long())
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) / math.sqrt(max(int(q.shape[-1]), 1))
    return torch.softmax(scores, dim=-1)


def image_positions_for_row(
    prefix_input_ids: Optional[torch.Tensor],
    *,
    row_idx: int,
    image_token_id: int = IMAGE_PLACEHOLDER_ID,
) -> Optional[torch.Tensor]:
    if prefix_input_ids is None:
        return None
    row = prefix_input_ids[int(row_idx)]
    valid_mask = row.ne(IGNORE_INDEX)
    positions = torch.nonzero(valid_mask & row.eq(int(image_token_id)), as_tuple=False).squeeze(-1)
    if int(positions.numel()) == 0:
        return None
    return positions.long()


def build_visual_redundancy_from_model(
    *,
    base_model,
    hidden_states: Optional[Sequence[torch.Tensor]],
    past_key_values: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]],
    window_positions: torch.Tensor,
    image_positions: Optional[torch.Tensor],
    row_idx: int = 0,
    return_visual_mask: bool = False,
):
    if hidden_states is None or image_positions is None:
        return None
    size = int(window_positions.numel())
    if size <= 1 or int(image_positions.numel()) <= 0:
        redundancy = torch.zeros((size, size), dtype=torch.float32, device=window_positions.device)
        visual_mask = torch.zeros((size,), dtype=torch.bool, device=window_positions.device)
        return (redundancy, visual_mask) if return_visual_mask else redundancy

    last_idx = len(base_model.transformer.blocks) - 1
    probs = attention_probs_for_queries(
        base_model=base_model,
        hidden_states=hidden_states,
        past_key_values=past_key_values,
        layer_idx=last_idx,
        query_positions=window_positions,
    )
    attn_ci = probs[int(row_idx)].index_select(-1, image_positions.long()).mean(dim=0)
    return build_visual_redundancy_matrix(attn_ci, return_visual_mask=return_visual_mask)


def select_transfer_positions(
    *,
    score_row: torch.Tensor,
    select_k: int,
    method: str,
    base_model=None,
    hidden_states: Optional[Sequence[torch.Tensor]] = None,
    past_key_values: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]] = None,
    prefix_input_ids: Optional[torch.Tensor] = None,
    row_idx: int = 0,
    image_token_id: int = IMAGE_PLACEHOLDER_ID,
    vrcd_alpha: float = 1.5,
    vrcd_lambda: float = 2.0,
) -> torch.Tensor:
    candidate_positions = torch.nonzero(torch.isfinite(score_row), as_tuple=False).squeeze(-1)
    if int(candidate_positions.numel()) == 0 or int(select_k) <= 0:
        return candidate_positions.new_empty((0,), dtype=torch.long)

    keep = min(int(select_k), int(candidate_positions.numel()))
    method = normalize_decoding_method(method)
    if method != "vrcd" or base_model is None or hidden_states is None:
        return positions_topk(score_row, candidate_positions, keep)

    window_k = max(keep, int(math.ceil(float(vrcd_lambda) * float(max(keep, 1)))))
    window_positions = positions_topk(score_row, candidate_positions, window_k)
    if int(window_positions.numel()) <= keep:
        return window_positions

    image_positions = image_positions_for_row(prefix_input_ids, row_idx=row_idx, image_token_id=image_token_id)
    visual_result = build_visual_redundancy_from_model(
        base_model=base_model,
        hidden_states=hidden_states,
        past_key_values=past_key_values,
        window_positions=window_positions,
        image_positions=image_positions,
        row_idx=int(row_idx),
        return_visual_mask=True,
    )
    if visual_result is None:
        return positions_topk(score_row, candidate_positions, keep)
    visual_redundancy, visual_mask = visual_result

    window_confidence = score_row.index_select(0, window_positions.long()).float()
    reranked_scores = vrcd_scores(window_confidence, visual_redundancy, alpha=float(vrcd_alpha), visual_mask=visual_mask)
    local_order = torch.argsort(reranked_scores, descending=True)[:keep]
    return window_positions.index_select(0, local_order.long())
