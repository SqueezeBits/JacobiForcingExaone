from __future__ import annotations

from typing import Optional

import torch
from transformers.cache_utils import Cache, DynamicCache


def _crop_cache(cache: DynamicCache, new_length: int) -> None:
    """Safely crop a DynamicCache to a committed prefix length."""
    if cache is None:
        return
    cache.crop(int(new_length))


def _pad_state_to_block_len(
    state: torch.Tensor,
    block_len: int,
    tokenizer=None,
    eos_token_id: Optional[int] = None,
) -> torch.Tensor:
    """
    Ensure trajectory states always have fixed length = n_token_seq_len.
    This matches the assumption in the existing packing scripts.
    """
    cur_len = int(state.shape[1])
    if cur_len == block_len:
        return state.clone()
    if cur_len > block_len:
        return state[:, :block_len].clone()

    pad_id = None
    if tokenizer is not None:
        pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = eos_token_id if eos_token_id is not None else 0

    pad = torch.full(
        (state.shape[0], block_len - cur_len),
        fill_value=int(pad_id),
        dtype=state.dtype,
        device=state.device,
    )
    return torch.cat((state, pad), dim=-1)


@torch.inference_mode()
def jacobi_forward_greedy(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    use_cache: Optional[bool] = None,
    prefill_phase: Optional[bool] = False,
    n_token_seq_len: int = 64,
    tokenizer=None,
    eos_token_id: Optional[int] = None,
):
    """
    Minimal EXAONE4 port of greedy Jacobi decoding for one block.

    This intentionally uses the model's public HF forward path rather than
    reimplementing decoder-layer internals. It is slower than a fully fused
    custom path, but much safer for an architecture bring-up.
    """
    if input_ids is None:
        raise ValueError("input_ids must be provided")

    eos_enabled = eos_token_id is not None

    if prefill_phase:
        outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=None,
            use_cache=True,
        )
        logits = outputs.logits.float()
        past_key_values = outputs.past_key_values
        first_correct_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        return past_key_values, first_correct_token, None, 0

    if past_key_values is None:
        raise ValueError("past_key_values must be provided for generation phase")

    out = input_ids.clone()
    accepted_n_gram = torch.empty_like(input_ids)
    total_accepted = 0
    iterations = 0
    next_token = input_ids[:, :1].clone()

    while total_accepted < n_token_seq_len:
        iterations += 1
        cached_prefix_len = past_key_values.get_seq_length()

        outputs = self(
            input_ids=out,
            attention_mask=None,
            past_key_values=past_key_values,
            use_cache=True,
        )
        logits = outputs.logits.float()
        past_key_values = outputs.past_key_values

        if out.shape[1] == 1:
            num_accepted_raw = 1
        else:
            greedy_tokens = torch.argmax(logits[:, :-1, :], dim=-1)
            mismatch = out[:, 1:] != greedy_tokens
            accepted = (mismatch.cumsum(dim=-1) == 0).sum(dim=-1) + 1
            num_accepted_raw = int(accepted.item())

        num_accepted = num_accepted_raw

        if eos_enabled:
            eos_in_prefix = out[0, :num_accepted_raw] == eos_token_id
            if eos_in_prefix.any():
                first_eos_idx = int(torch.nonzero(eos_in_prefix, as_tuple=False)[0].item())
                num_accepted = first_eos_idx + 1

        if num_accepted > 0:
            accepted_n_gram[:, total_accepted : total_accepted + num_accepted] = out[:, :num_accepted]
            total_accepted += num_accepted

        committed_cache_len = cached_prefix_len + num_accepted_raw
        _crop_cache(past_key_values, committed_cache_len)

        if eos_enabled and (out[0, :num_accepted] == eos_token_id).any():
            return past_key_values, out[:, num_accepted - 1 : num_accepted], accepted_n_gram[:, :total_accepted], iterations

        has_rejected = num_accepted_raw < out.shape[1]
        if has_rejected:
            next_token = torch.argmax(logits[:, num_accepted_raw - 1, :], dim=-1, keepdim=True)
            rebuilt = next_token

            q_probs_rem = logits[:, num_accepted_raw:-1, :]
            if q_probs_rem.shape[1] > 0:
                greedy_tail = torch.argmax(q_probs_rem, dim=-1)
                rebuilt = torch.cat((rebuilt, greedy_tail), dim=-1)
            out = rebuilt
            continue

        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        accepted_n_gram[:, total_accepted : total_accepted + 1] = next_token
        total_accepted += 1

        if eos_enabled and int(next_token.item()) == eos_token_id:
            return past_key_values, next_token, accepted_n_gram[:, :total_accepted], iterations

        out = next_token

    return past_key_values, next_token, accepted_n_gram[:, :total_accepted], iterations


@torch.inference_mode()
def get_jacobi_forward_trajectory_greedy(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    use_cache: Optional[bool] = None,
    prefill_phase: Optional[bool] = False,
    n_token_seq_len: int = 64,
    tokenizer=None,
    eos_token_id: Optional[int] = None,
):
    """
    Greedy Jacobi trajectory collector for EXAONE4.

    Returns:
      - prefill_phase=True:
            past_key_values, first_correct_token
      - prefill_phase=False:
            past_key_values, next_token, answer_trajectory_ids
    """
    if input_ids is None:
        raise ValueError("input_ids must be provided")

    eos_enabled = eos_token_id is not None
    answer_trajectory_ids: list[torch.Tensor] = []

    if prefill_phase:
        outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=None,
            use_cache=True,
        )
        logits = outputs.logits.float()
        past_key_values = outputs.past_key_values
        first_correct_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        return past_key_values, first_correct_token

    if past_key_values is None:
        raise ValueError("past_key_values must be provided for generation phase")

    out = input_ids.clone()
    accepted_n_gram = torch.empty_like(input_ids)
    answer_trajectory_ids.append(
        _pad_state_to_block_len(out, n_token_seq_len, tokenizer=tokenizer, eos_token_id=eos_token_id)
    )

    total_accepted = 0
    next_token = input_ids[:, :1].clone()

    while total_accepted < n_token_seq_len:
        cached_prefix_len = past_key_values.get_seq_length()

        outputs = self(
            input_ids=out,
            attention_mask=None,
            past_key_values=past_key_values,
            use_cache=True,
        )
        logits = outputs.logits.float()
        past_key_values = outputs.past_key_values

        if out.shape[1] == 1:
            num_accepted_raw = 1
        else:
            greedy_tokens = torch.argmax(logits[:, :-1, :], dim=-1)
            mismatch = out[:, 1:] != greedy_tokens
            accepted = (mismatch.cumsum(dim=-1) == 0).sum(dim=-1) + 1
            num_accepted_raw = int(accepted.item())

        num_accepted = num_accepted_raw
        if eos_enabled:
            eos_in_prefix = out[0, :num_accepted_raw] == eos_token_id
            if eos_in_prefix.any():
                first_eos_idx = int(torch.nonzero(eos_in_prefix, as_tuple=False)[0].item())
                num_accepted = first_eos_idx + 1

        if num_accepted > 0:
            accepted_n_gram[:, total_accepted : total_accepted + num_accepted] = out[:, :num_accepted]
            total_accepted += num_accepted

        committed_cache_len = cached_prefix_len + num_accepted_raw
        _crop_cache(past_key_values, committed_cache_len)

        if eos_enabled and (out[0, :num_accepted] == eos_token_id).any():
            answer_trajectory_ids.append(
                _pad_state_to_block_len(
                    accepted_n_gram[:, :total_accepted],
                    n_token_seq_len,
                    tokenizer=tokenizer,
                    eos_token_id=eos_token_id,
                )
            )
            return past_key_values, out[:, num_accepted - 1 : num_accepted], answer_trajectory_ids

        has_rejected = num_accepted_raw < out.shape[1]
        if has_rejected:
            next_token = torch.argmax(logits[:, num_accepted_raw - 1, :], dim=-1, keepdim=True)
            rebuilt = next_token
            q_logits_rem = logits[:, num_accepted_raw:-1, :]
            if q_logits_rem.shape[1] > 0:
                greedy_tail = torch.argmax(q_logits_rem, dim=-1)
                rebuilt = torch.cat((rebuilt, greedy_tail), dim=-1)

            visible_state = torch.cat((accepted_n_gram[:, :total_accepted], rebuilt), dim=-1)
            answer_trajectory_ids.append(
                _pad_state_to_block_len(
                    visible_state,
                    n_token_seq_len,
                    tokenizer=tokenizer,
                    eos_token_id=eos_token_id,
                )
            )
            out = rebuilt
            continue

        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        accepted_n_gram[:, total_accepted : total_accepted + 1] = next_token
        total_accepted += 1
        answer_trajectory_ids.append(
            _pad_state_to_block_len(
                accepted_n_gram[:, :total_accepted],
                n_token_seq_len,
                tokenizer=tokenizer,
                eos_token_id=eos_token_id,
            )
        )

        if eos_enabled and int(next_token.item()) == eos_token_id:
            return past_key_values, next_token, answer_trajectory_ids

        out = next_token

    if not answer_trajectory_ids or answer_trajectory_ids[-1].shape[1] != total_accepted:
        answer_trajectory_ids.append(
            _pad_state_to_block_len(
                accepted_n_gram[:, :total_accepted],
                n_token_seq_len,
                tokenizer=tokenizer,
                eos_token_id=eos_token_id,
            )
        )

    return past_key_values, next_token, answer_trajectory_ids
