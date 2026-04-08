#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate one Jacobi Forcing training step on EXAONE4.")
    parser.add_argument(
        "--model-id",
        default="LGAI-EXAONE/EXAONE-4.0-1.2B",
        help="HF repo id or local model path.",
    )
    parser.add_argument(
        "--packed-jsonl",
        default="/workspace/exaone_workspace/JacobiForcing/tmp/exaone4_test/packed.jsonl",
        help="Packed training sample JSONL path.",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    parser.add_argument(
        "--run-backward",
        action="store_true",
        help="Run backward() to verify gradients are produced.",
    )
    return parser.parse_args()


def _to_device_dtype(model_id: str, dtype_flag: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[dtype_flag]
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        attn_implementation="flex_attention",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return device, model, tokenizer


def load_one_sample(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        line = f.readline()
    return json.loads(line)


def index_layout(prompt_len: int, T: int, N: int):
    k_starts = [prompt_len + 2 * j * N for j in range(T)]
    l_starts = [prompt_len + (2 * j + 1) * N for j in range(T)]
    return k_starts, l_starts


def build_shared_position_ids(device: torch.device, L: int, prompt_len: int, T: int, N: int):
    pos = torch.empty(L, dtype=torch.long, device=device)
    pos[:prompt_len] = torch.arange(prompt_len, device=device)
    k_starts, l_starts = index_layout(prompt_len, T, N)
    rel = torch.arange(N, device=device)
    for j in range(T):
        base = prompt_len + j * N
        ks = k_starts[j]
        ls = l_starts[j]
        pos[ks : ks + N] = base + rel
        pos[ls : ls + N] = base + rel
    return pos


def duplicate_prefix_mask(input_ids: torch.Tensor, prompt_len: int, T: int, N: int) -> torch.Tensor:
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    k_starts, l_starts = index_layout(prompt_len, T, N)
    for j in range(T):
        ks = k_starts[j]
        ls = l_starts[j]
        k_block = input_ids[ks : ks + N]
        l_block = input_ids[ls : ls + N]
        eq = k_block == l_block
        if torch.any(~eq):
            first_diff = int(torch.nonzero(~eq, as_tuple=False)[0])
        else:
            first_diff = N
        if first_diff > 0:
            mask[ks : ks + first_diff] = True
    return mask


def build_padding_mask_for_loss(input_ids: torch.Tensor, prompt_len: int, T: int, N: int, pad_id: int | None):
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    if pad_id is not None:
        mask |= input_ids == pad_id
    mask |= duplicate_prefix_mask(input_ids, prompt_len, T, N)
    return mask


def block_keep_mask_divergence(input_ids: torch.Tensor, k_start: int, l_start: int, N: int) -> torch.Tensor:
    offs = torch.arange(N, device=input_ids.device)
    k_block = input_ids[k_start : k_start + N]
    l_block = input_ids[l_start : l_start + N]
    diff = k_block != l_block
    if diff.any():
        first_diff = int(torch.nonzero(diff, as_tuple=False)[0])
        return offs >= first_diff
    return torch.zeros(N, dtype=torch.bool, device=input_ids.device)


def soft_cross_entropy(predicts: torch.Tensor, targets: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
    if (~padding_mask).sum() == 0:
        return predicts.sum() * 0.0
    predict_log_prob = torch.nn.functional.log_softmax(predicts, dim=-1)
    targets_prob = torch.nn.functional.softmax(targets, dim=-1)
    entropy = -targets_prob * predict_log_prob
    expand_mask = padding_mask.unsqueeze(-1).expand_as(entropy)
    entropy = entropy.masked_fill(expand_mask, 0)
    return entropy.sum() / (~padding_mask).sum()


def build_block_mask(device: torch.device, prompt_len: int, T: int, N: int, heads: int):
    k_starts, l_starts = index_layout(prompt_len, T, N)
    ks = torch.tensor(k_starts, device=device)
    ls = torch.tensor(l_starts, device=device)

    def mask_mod(b, h, q, k):
        rel_q = q - prompt_len
        rel_k = k - prompt_len
        block_idx_q = torch.div(rel_q, N, rounding_mode="floor")
        block_idx_k = torch.div(rel_k, N, rounding_mode="floor")

        is_prompt_q = q < prompt_len
        is_prompt_k = k < prompt_len

        is_kj_q = (q >= prompt_len) & (block_idx_q % 2 == 0)
        is_lastj_q = (q >= prompt_len) & (block_idx_q % 2 == 1)
        is_kj_k = (k >= prompt_len) & (block_idx_k % 2 == 0)
        is_lastj_k = (k >= prompt_len) & (block_idx_k % 2 == 1)

        j_q = torch.clamp(block_idx_q // 2, min=0, max=T - 1)
        ks_per_q = ks[j_q]
        ls_per_q = ls[j_q]

        k_in_prev_k = is_kj_k & (block_idx_k < 2 * j_q)
        last_in_prev_last = is_lastj_k & (block_idx_k < 2 * j_q)
        mask_prompt = is_prompt_q & (k <= q)

        same_kj_block = is_kj_q & is_kj_k & (block_idx_q == block_idx_k)
        mask_kj = is_kj_q & (
            is_prompt_k
            | k_in_prev_k
            | (same_kj_block & (k >= ks_per_q) & (k <= q))
        )

        same_lastj_block = is_lastj_q & is_lastj_k & (block_idx_q == block_idx_k)
        mask_lastj = is_lastj_q & (
            is_prompt_k
            | last_in_prev_last
            | (same_lastj_block & (k >= ls_per_q) & (k <= q))
        )

        return mask_prompt | mask_kj | mask_lastj

    return create_block_mask(
        mask_mod,
        B=1,
        H=heads,
        Q_LEN=prompt_len + 2 * T * N,
        KV_LEN=prompt_len + 2 * T * N,
        device=device,
        _compile=True,
    )


def main() -> int:
    args = parse_args()
    device, model, tokenizer = _to_device_dtype(args.model_id, args.dtype)
    sample = load_one_sample(args.packed_jsonl)

    prompt_len = int(sample["prompt_ids_len"])
    input_ids = torch.tensor(sample["complete_training_sequence_ids"], dtype=torch.long, device=device)
    traj_position_indices = sample["traj_position_indices"]
    T = len(traj_position_indices)

    if T <= 0:
        raise ValueError("Packed sample must contain at least one trajectory pair.")

    N = (len(sample["complete_training_sequence_ids"]) - prompt_len) // (2 * T)
    L = input_ids.shape[0]
    expected_len = prompt_len + 2 * T * N
    if L != expected_len:
        raise ValueError(f"Length mismatch: L={L}, expected={expected_len}")

    model.train(args.run_backward)

    blk_mask = build_block_mask(device, prompt_len, T, N, model.config.num_attention_heads)
    position_ids = build_shared_position_ids(device, L, prompt_len, T, N)

    attn_mask_mapping = {"full_attention": blk_mask}
    outputs = model(
        input_ids=input_ids.unsqueeze(0),
        attention_mask=attn_mask_mapping,
        position_ids=position_ids.unsqueeze(0),
        use_cache=False,
    )
    logits = outputs.logits

    k_starts, l_starts = index_layout(prompt_len, T, N)

    pair_logit_positions = []
    pair_target_positions = []

    def add_forward_pairs(seg_start: int, seg_end: int):
        if seg_end - seg_start <= 1:
            return
        p = torch.arange(seg_start, seg_end - 1, device=device, dtype=torch.long)
        t = p + 1
        pair_logit_positions.append(p)
        pair_target_positions.append(t)

    add_forward_pairs(0, prompt_len)
    for j in range(T):
        ls = l_starts[j]
        if j == 0:
            logit_pos = prompt_len - 1
            target_pos = ls
        else:
            prev_ls = l_starts[j - 1]
            logit_pos = prev_ls + (N - 1)
            target_pos = ls
        pair_logit_positions.append(torch.tensor([logit_pos], device=device))
        pair_target_positions.append(torch.tensor([target_pos], device=device))
        add_forward_pairs(ls, ls + N)

    p_all = torch.cat(pair_logit_positions, dim=0)
    t_all = torch.cat(pair_target_positions, dim=0)
    ar_logits = logits[0, p_all, :]
    ar_targets = input_ids.index_select(0, t_all).clone().detach()
    if tokenizer.pad_token_id is not None:
        ar_targets[ar_targets == tokenizer.pad_token_id] = -100
    loss_ar = F.cross_entropy(
        ar_logits.float(),
        ar_targets,
        reduction="mean",
        label_smoothing=0.0,
        ignore_index=-100,
    ) * 10

    offs = torch.arange(N, device=device)
    student_positions, teacher_positions = [], []
    for j in range(T):
        ks, ls = k_starts[j], l_starts[j]
        pair_keep = block_keep_mask_divergence(input_ids, ks, ls, N)
        if pair_keep.any():
            student_positions.append(ks + offs[pair_keep])
            teacher_positions.append(ls + offs[pair_keep])

    if student_positions:
        sp = torch.cat(student_positions, dim=0)
        tp = torch.cat(teacher_positions, dim=0)
        global_pad_and_dup_mask = build_padding_mask_for_loss(
            input_ids, prompt_len, T, N, tokenizer.pad_token_id
        )
        padding_mask = global_pad_and_dup_mask.index_select(0, sp)
        student_logits_all = logits[0, sp, :]
        teacher_logits_all = logits[0, tp, :].detach()
        loss_consistency = soft_cross_entropy(
            student_logits_all.float(),
            teacher_logits_all.float(),
            padding_mask,
        ) / T
    else:
        loss_consistency = torch.zeros((), device=device)

    total_loss = loss_ar + loss_consistency

    grad_norm = None
    if args.run_backward:
        model.zero_grad(set_to_none=True)
        total_loss.backward()
        for param in model.parameters():
            if param.grad is not None:
                grad_norm = float(param.grad.detach().float().norm().item())
                break

    print(
        json.dumps(
            {
                "prompt_len": prompt_len,
                "T": T,
                "block_size": N,
                "seq_len": L,
                "num_heads": model.config.num_attention_heads,
                "logits_shape": list(logits.shape),
                "loss_ar": float(loss_ar.detach().float().item()),
                "loss_consistency": float(loss_consistency.detach().float().item()),
                "total_loss": float(total_loss.detach().float().item()),
                "ran_backward": bool(args.run_backward),
                "first_grad_norm": grad_norm,
                "target_preview": tokenizer.decode(input_ids[l_starts[0] : l_starts[0] + N], skip_special_tokens=False),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
