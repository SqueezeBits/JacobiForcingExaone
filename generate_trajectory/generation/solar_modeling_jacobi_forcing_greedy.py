from typing import Optional

import torch
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation.logits_process import LogitsProcessorList


def trim_dynamic_cache(cache: DynamicCache, num_of_false_tokens: int) -> None:
    if num_of_false_tokens <= 0:
        return
    current_len = cache.get_seq_length()
    target_len = max(0, current_len - num_of_false_tokens)
    cache.crop(target_len)


def ensure_attention_types(model) -> None:
    layer_types = getattr(model.config, "layer_types", None)
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        return
    for idx, layer in enumerate(layers):
        if hasattr(layer, "attention_type"):
            continue
        if isinstance(layer_types, list) and idx < len(layer_types):
            layer.attention_type = layer_types[idx]
        else:
            layer.attention_type = "full_attention"


@torch.inference_mode()
def get_jacobi_forward_trajectory_greedy(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    use_cache: Optional[bool] = None,
    prefill_phase: Optional[bool] = False,
    n_token_seq_len=64,
    temperature=1.0,
    top_p=0.9,
    top_k=None,
    repetition_penalty=None,
    lenience=1.0,
    accept_threshold=0.99,
    tokenizer=None,
    eos_token_id: Optional[int] = None,
):

    if input_ids is None:
        raise ValueError("You must specify exactly input_ids")

    ensure_attention_types(self)
    eos_id = eos_token_id
    eos_enabled = eos_id is not None
    logits_processors = LogitsProcessorList()

    if prefill_phase:
        # Prefill consumes the prompt once and initializes the KV cache. The next-token greedy
        # prediction becomes the fixed first token for the upcoming Jacobi draft block.
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        past_key_values = outputs.past_key_values
        logits = self.lm_head(outputs.last_hidden_state).float()
        scores = logits_processors(input_ids, logits.squeeze(0)).unsqueeze(0)
        # Compute all greedy choices once and reuse them instead of repeatedly taking argmax
        # from different score slices.
        greedy_all_tokens = torch.argmax(scores, dim=-1)
        first_correct_token = greedy_all_tokens[:, -1:].clone()
        return past_key_values, first_correct_token

    assert past_key_values is not None

    batch, out, device = input_ids.shape[0], input_ids, input_ids.device
    accepted_n_gram = out.clone()
    answer_trajectory_ids = [out.clone()]
    total_accepted = 0
    itr = 0

    while total_accepted < n_token_seq_len:
        itr += 1
        # Each loop refines the current draft block `out` and accepts the longest greedy-matching prefix.
        outputs = self.model(
            input_ids=out,
            attention_mask=torch.ones_like(out, device=device),
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        logits = self.lm_head(outputs.last_hidden_state).float()
        scores = logits_processors(out, logits.squeeze(0)).unsqueeze(0)
        greedy_all_tokens = torch.argmax(scores, dim=-1)
        greedy_tokens = greedy_all_tokens[:, :-1]
        mismatch = out[:, 1:] != greedy_tokens
        accepted = (mismatch.cumsum(dim=-1) == 0).sum(dim=-1) + 1
        num_accepted_raw = int(accepted[0])
        seq_len = out.shape[1]

        num_accepted = num_accepted_raw
        if eos_enabled:
            eos_in_prefix = out[0, :num_accepted_raw] == eos_id
            if eos_in_prefix.any():
                first_eos_idx = torch.nonzero(eos_in_prefix, as_tuple=False)[0].item()
                num_accepted = first_eos_idx + 1

        if num_accepted > 0:
            accepted_n_gram[:, total_accepted : total_accepted + num_accepted] = out[:, :num_accepted].clone()
        total_accepted += num_accepted

        if eos_enabled and (out[0, :num_accepted] == eos_id).any():
            current_len = past_key_values.get_seq_length()
            desired_len = total_accepted
            to_delete = max(0, current_len - desired_len)
            if to_delete > 0:
                trim_dynamic_cache(past_key_values, to_delete)
            return (
                past_key_values,
                torch.full((batch, 1), eos_id, device=device, dtype=out.dtype),
                answer_trajectory_ids,
            )

        has_rejected = num_accepted_raw < seq_len
        if has_rejected:
            trim_dynamic_cache(past_key_values, out.shape[1] - num_accepted_raw)
            next_token = greedy_tokens[:, num_accepted_raw - 1].unsqueeze(-1)

            if eos_enabled and next_token.item() == eos_id:
                accepted_n_gram[:, total_accepted : total_accepted + 1] = next_token
                total_accepted += 1
                current_len = past_key_values.get_seq_length()
                desired_len = total_accepted
                to_delete = max(0, current_len - desired_len)
                if to_delete > 0:
                    trim_dynamic_cache(past_key_values, to_delete)
                answer_trajectory_ids.append(accepted_n_gram[:, :total_accepted].clone())
                return past_key_values, next_token, answer_trajectory_ids

            out = next_token
            # The remaining greedy positions have already been computed, so we reuse them directly.
            q_tokens_rem = greedy_tokens[:, num_accepted_raw:]
            if q_tokens_rem.shape[1] > 0:
                q_sampled = q_tokens_rem
                out = torch.cat((out, q_sampled), dim=-1)
            answer_trajectory_ids.append(torch.cat((accepted_n_gram[:, :total_accepted], out), dim=-1))
            continue

        next_token = greedy_all_tokens[:, -1:].clone()
        accepted_n_gram[:, total_accepted : total_accepted + 1] = next_token

        if eos_enabled and next_token.item() == eos_id:
            total_accepted += 1
            current_len = past_key_values.get_seq_length()
            desired_len = total_accepted
            to_delete = max(0, current_len - desired_len)
            if to_delete > 0:
                trim_dynamic_cache(past_key_values, to_delete)
            answer_trajectory_ids.append(accepted_n_gram[:, :total_accepted].clone())
            return past_key_values, next_token, answer_trajectory_ids

        total_accepted += 1
        answer_trajectory_ids.append(accepted_n_gram[:, :total_accepted].clone())

    return past_key_values, next_token, answer_trajectory_ids
