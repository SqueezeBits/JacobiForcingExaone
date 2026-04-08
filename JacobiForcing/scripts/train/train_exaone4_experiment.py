#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment-grade EXAONE4 Jacobi Forcing trainer.")
    parser.add_argument("--model-id", default="LGAI-EXAONE/EXAONE-4.0-1.2B")
    parser.add_argument("--data-path", required=True, help="Packed training JSONL")
    parser.add_argument("--output-dir", required=True, help="Directory for checkpoints and logs")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lr-scheduler", default="cosine", choices=["cosine", "constant"])
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--sample-every", type=int, default=0, help="Generate a sample every N optimizer steps")
    parser.add_argument("--sample-max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--sample-prompt",
        default="Write a short Python function named fib(n) that returns the nth Fibonacci number. Return code only.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--limit-rows", type=int, default=0, help="Optional cap on number of rows loaded")
    parser.add_argument("--save-optimizer", action="store_true")
    parser.add_argument(
        "--resume-from",
        default="",
        help="Optional checkpoint directory such as .../step_00010",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_rows(path: str, limit_rows: int = 0) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit_rows > 0 and len(rows) >= limit_rows:
                break
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


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


def build_padding_mask_for_loss(
    input_ids: torch.Tensor,
    prompt_len: int,
    T: int,
    N: int,
    pad_id: int | None,
) -> torch.Tensor:
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
        mask_kj = is_kj_q & (is_prompt_k | k_in_prev_k | (same_kj_block & (k >= ks_per_q) & (k <= q)))

        same_lastj_block = is_lastj_q & is_lastj_k & (block_idx_q == block_idx_k)
        mask_lastj = is_lastj_q & (
            is_prompt_k | last_in_prev_last | (same_lastj_block & (k >= ls_per_q) & (k <= q))
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


def compute_loss(model, tokenizer, row: dict, device: torch.device, autocast_ctx) -> tuple[torch.Tensor, dict]:
    prompt_len = int(row["prompt_ids_len"])
    input_ids = torch.tensor(row["complete_training_sequence_ids"], dtype=torch.long, device=device)
    traj_position_indices = row["traj_position_indices"]
    T = len(traj_position_indices)
    if T <= 0:
        raise ValueError("Packed sample must contain at least one trajectory pair.")

    N = (len(row["complete_training_sequence_ids"]) - prompt_len) // (2 * T)
    L = input_ids.shape[0]

    blk_mask = build_block_mask(device, prompt_len, T, N, model.config.num_attention_heads)
    position_ids = build_shared_position_ids(device, L, prompt_len, T, N)

    with autocast_ctx:
        outputs = model(
            input_ids=input_ids.unsqueeze(0),
            attention_mask={"full_attention": blk_mask},
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
    stats = {
        "prompt_len": prompt_len,
        "T": T,
        "block_size": N,
        "seq_len": L,
        "loss_ar": float(loss_ar.detach().float().item()),
        "loss_consistency": float(loss_consistency.detach().float().item()),
        "total_loss": float(total_loss.detach().float().item()),
    }
    return total_loss, stats


def create_scheduler(optimizer, total_steps: int, warmup_ratio: float, scheduler_name: str):
    warmup_steps = int(total_steps * warmup_ratio)
    warmup_steps = max(0, warmup_steps)

    if scheduler_name == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)

    def lr_lambda(step: int):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(
    model,
    tokenizer,
    optimizer,
    scheduler,
    output_dir: str,
    step: int,
    save_optimizer: bool,
    extra_state: dict,
) -> None:
    ckpt_dir = Path(output_dir) / f"step_{step:05d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt_dir, safe_serialization=False)
    tokenizer.save_pretrained(ckpt_dir)
    state = {"step": step, **extra_state}
    with open(ckpt_dir / "trainer_state.json", "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    if save_optimizer:
        torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
        torch.save(scheduler.state_dict(), ckpt_dir / "scheduler.pt")


def maybe_resume(
    args,
    model,
    tokenizer,
    optimizer,
    scheduler,
    output_dir: Path,
):
    if not args.resume_from:
        return 0, 0

    ckpt_dir = Path(args.resume_from)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {ckpt_dir}")

    trainer_state_path = ckpt_dir / "trainer_state.json"
    if not trainer_state_path.exists():
        raise FileNotFoundError(f"Missing trainer_state.json in {ckpt_dir}")

    state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    optimizer_step = int(state.get("step", 0))
    global_step = int(state.get("global_step", optimizer_step))

    if (ckpt_dir / "optimizer.pt").exists():
        optimizer.load_state_dict(torch.load(ckpt_dir / "optimizer.pt", map_location="cpu"))
    if (ckpt_dir / "scheduler.pt").exists():
        scheduler.load_state_dict(torch.load(ckpt_dir / "scheduler.pt", map_location="cpu"))

    # The caller already loaded the model from args.model_id or a checkpoint path.
    # We only verify tokenizer consistency and return counters.
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                "resume_from": str(ckpt_dir),
                "optimizer_step": optimizer_step,
                "global_step": global_step,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return global_step, optimizer_step


@torch.inference_mode()
def generate_sample(model, tokenizer, prompt: str, device: torch.device, max_new_tokens: int) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": prompt},
    ]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer(rendered, return_tensors="pt").to(device)
    output_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_ids = output_ids[0, model_inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_ids, skip_special_tokens=False)


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "train_metrics.jsonl"
    samples_path = output_dir / "sample_generations.jsonl"

    rows = load_rows(args.data_path, limit_rows=args.limit_rows)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    target_dtype = dtype_map[args.dtype]

    model_load_path = args.resume_from if args.resume_from else args.model_id
    load_kwargs = {
        "attn_implementation": "flex_attention",
        "torch_dtype": target_dtype,
        "low_cpu_mem_usage": True,
    }
    tokenizer_load_kwargs = {}
    if args.resume_from:
        load_kwargs["local_files_only"] = True
        tokenizer_load_kwargs["local_files_only"] = True

    model = AutoModelForCausalLM.from_pretrained(
        model_load_path,
        **load_kwargs,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_load_path, **tokenizer_load_kwargs)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model.train()
    if args.shuffle:
        random.shuffle(rows)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_optimizer_steps = max(1, args.max_steps)
    scheduler = create_scheduler(optimizer, total_optimizer_steps, args.warmup_ratio, args.lr_scheduler)

    if device.type == "cuda" and target_dtype in (torch.bfloat16, torch.float16):
        autocast_ctx = torch.autocast(device_type="cuda", dtype=target_dtype)
    else:
        autocast_ctx = nullcontext()

    global_step = 0
    optimizer_step = 0
    running_loss = 0.0

    metrics_f = metrics_path.open("a", encoding="utf-8")
    samples_f = samples_path.open("a", encoding="utf-8")

    try:
        global_step, optimizer_step = maybe_resume(
            args,
            model,
            tokenizer,
            optimizer,
            scheduler,
            output_dir,
        )
        optimizer.zero_grad(set_to_none=True)

        for epoch in range(args.epochs):
            if args.shuffle:
                random.shuffle(rows)

            for row_idx, row in enumerate(rows):
                if optimizer_step >= args.max_steps:
                    break

                loss, stats = compute_loss(model, tokenizer, row, device, autocast_ctx)
                loss_to_backprop = loss / args.gradient_accumulation_steps
                loss_to_backprop.backward()
                running_loss += float(loss.detach().float().item())
                global_step += 1

                should_step = (global_step % args.gradient_accumulation_steps == 0)
                if not should_step:
                    continue

                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm).item()
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1

                log_record = {
                    "epoch": epoch,
                    "row_idx": row_idx,
                    "global_step": global_step,
                    "optimizer_step": optimizer_step,
                    "lr": optimizer.param_groups[0]["lr"],
                    "grad_norm": grad_norm,
                    "avg_loss_since_last_step": running_loss / args.gradient_accumulation_steps,
                    **stats,
                }
                running_loss = 0.0

                if optimizer_step % args.log_every == 0:
                    print(json.dumps(log_record, ensure_ascii=False), flush=True)
                metrics_f.write(json.dumps(log_record, ensure_ascii=False) + "\n")
                metrics_f.flush()

                if args.sample_every > 0 and optimizer_step % args.sample_every == 0:
                    model.eval()
                    sample_text = generate_sample(
                        model,
                        tokenizer,
                        prompt=args.sample_prompt,
                        device=device,
                        max_new_tokens=args.sample_max_new_tokens,
                    )
                    model.train()
                    sample_record = {
                        "optimizer_step": optimizer_step,
                        "prompt": args.sample_prompt,
                        "generation": sample_text,
                    }
                    print(json.dumps({"sample": sample_record}, ensure_ascii=False), flush=True)
                    samples_f.write(json.dumps(sample_record, ensure_ascii=False) + "\n")
                    samples_f.flush()

                if args.save_every > 0 and optimizer_step % args.save_every == 0:
                    save_checkpoint(
                        model,
                        tokenizer,
                        optimizer,
                        scheduler,
                        args.output_dir,
                        optimizer_step,
                        args.save_optimizer,
                        extra_state={"epoch": epoch, "global_step": global_step},
                    )

            if optimizer_step >= args.max_steps:
                break

        if optimizer_step > 0:
            save_checkpoint(
                model,
                tokenizer,
                optimizer,
                scheduler,
                args.output_dir,
                optimizer_step,
                args.save_optimizer,
                extra_state={"epochs_completed": epoch + 1, "global_step": global_step},
            )
    finally:
        metrics_f.close()
        samples_f.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
