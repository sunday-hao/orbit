"""Check that opd_jsd_loss_function under tensor parallelism matches the TP=1 reference.

Spawns `--tp-size` gloo processes on CPU, runs the loss with the vocabulary sharded across
them, and compares the loss, the metrics and the gradient w.r.t. the student logits against a
single-process run over the unsharded vocabulary. No GPU and no Megatron checkpoint needed --
the teacher LM head is injected straight into the module cache, and sliced here rather than by
`load_teacher_lm_head`, so this checks the loss and not the loader (loss.py cross-checks the
loader's shard width against the student's logits at runtime anyway).

    python tools/verify_opd_jsd_tp.py --tp-size 4 --beta 0.5
    python tools/verify_opd_jsd_tp.py --tp-size 2 --beta 0.0 --topk-overlap
"""

import argparse
import os
from argparse import Namespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from orbit.backends.training_utils import teacher_lm_head as teacher_lm_head_module
from orbit.backends.training_utils.loss import opd_jsd_loss_function
from orbit.backends.training_utils.parallel import GroupInfo, ParallelState, set_parallel_state

CHECKPOINT_KEY = "<verify-opd-jsd-tp>"

HIDDEN_SIZE = 16
PADDED_VOCAB_SIZE = 512
RESPONSE_LENGTHS = [5, 1, 4]
PROMPT_LENGTHS = [3, 2, 6]


def build_args(beta: float, topk_overlap: bool) -> Namespace:
    return Namespace(
        opd_jsd_beta=beta,
        rollout_temperature=0.8,
        opd_log_prob_min_clamp=-20.0,
        opd_loss_max_clamp=100.0,
        opd_jsd_pointwise_clip=0.5,
        opd_log_topk_overlap=topk_overlap,
        opd_topk_overlap_ks=[1, 5, 20],
        use_kl_loss=False,
        teacher_hf_checkpoint=CHECKPOINT_KEY,
        qkv_format="thd",
        allgather_cp=False,
        log_probs_chunk_size=-1,
        true_on_policy_mode=False,
    )


def build_inputs(teacher_vocab_size: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Deterministic logits, teacher head and batch -- identical on every rank."""
    generator = torch.Generator().manual_seed(1234)
    total_lengths = [p + r for p, r in zip(PROMPT_LENGTHS, RESPONSE_LENGTHS, strict=True)]

    logits = torch.randn(
        1, sum(total_lengths), PADDED_VOCAB_SIZE, generator=generator, dtype=torch.float32
    )
    teacher_head = torch.randn(teacher_vocab_size, HIDDEN_SIZE, generator=generator, dtype=torch.float32)

    batch = {
        "unconcat_tokens": [
            torch.randint(0, teacher_vocab_size, (total,), generator=generator) for total in total_lengths
        ],
        "response_lengths": RESPONSE_LENGTHS,
        "total_lengths": total_lengths,
        "teacher_hidden_states": [
            torch.randn(response, HIDDEN_SIZE, generator=generator, dtype=torch.float32).numpy()
            for response in RESPONSE_LENGTHS
        ],
    }
    return logits, teacher_head, batch


def set_tp_state(rank: int, tp_size: int, group: dist.ProcessGroup | None) -> None:
    single = GroupInfo(rank=0, size=1, group=None)
    set_parallel_state(
        ParallelState(
            intra_dp=single,
            intra_dp_cp=single,
            cp=single,
            tp=GroupInfo(rank=rank, size=tp_size, group=group),
        )
    )


def run_loss(args: Namespace, logits: torch.Tensor, head: torch.Tensor, batch: dict):
    """Run the loss on `logits`, returning (loss, metrics, grad-w.r.t.-logits)."""
    teacher_lm_head_module._TEACHER_LM_HEAD_CACHE[CHECKPOINT_KEY] = head
    # Already sliced by the caller, so stop load_teacher_lm_head() from sharding it again.
    teacher_lm_head_module._SHARDED.add(CHECKPOINT_KEY)
    logits = logits.detach().clone().requires_grad_(True)
    loss, metrics = opd_jsd_loss_function(args, batch, logits, lambda x: x.sum())
    loss.backward()
    return loss.detach(), {k: v.clone() for k, v in metrics.items()}, logits.grad.detach()


def worker(rank: int, tp_size: int, port: int, beta: float, topk_overlap: bool, teacher_vocab: int) -> None:
    dist.init_process_group(
        "gloo", rank=rank, world_size=tp_size, init_method=f"tcp://127.0.0.1:{port}"
    )
    args = build_args(beta, topk_overlap)
    logits, teacher_head, batch = build_inputs(teacher_vocab)

    # Reference first, while the loss still sees a TP size of 1 and takes no collectives --
    # safe to run on rank 0 alone.
    if rank == 0:
        set_tp_state(0, 1, None)
        ref_loss, ref_metrics, ref_grad = run_loss(args, logits, teacher_head, batch)

    shard = PADDED_VOCAB_SIZE // tp_size
    start, end = rank * shard, (rank + 1) * shard
    set_tp_state(rank, tp_size, dist.group.WORLD)
    tp_loss, tp_metrics, tp_grad = run_loss(
        args,
        logits[:, :, start:end],
        teacher_head[start : min(end, teacher_vocab)],
        batch,
    )

    gathered = [torch.empty_like(tp_grad) for _ in range(tp_size)]
    dist.all_gather(gathered, tp_grad.contiguous())
    if rank != 0:
        dist.destroy_process_group()
        return

    tp_grad_full = torch.cat(gathered, dim=-1)
    print(
        f"\n=== tp_size={tp_size} beta={beta} topk_overlap={topk_overlap} "
        f"teacher_vocab={teacher_vocab}/{PADDED_VOCAB_SIZE} ==="
    )
    failures = []

    def check(name: str, ref: torch.Tensor, got: torch.Tensor, atol: float, rtol: float) -> None:
        diff = (ref - got).abs().max().item()
        scale = ref.abs().max().item()
        ok = torch.allclose(ref, got, atol=atol, rtol=rtol)
        print(f"  {'ok  ' if ok else 'FAIL'} {name:<22} max|diff|={diff:.3e}  max|ref|={scale:.3e}")
        if not ok:
            failures.append(name)

    check("loss", ref_loss, tp_loss, atol=1e-4, rtol=1e-5)
    check("grad(logits)", ref_grad, tp_grad_full, atol=1e-6, rtol=1e-4)
    for key in sorted(ref_metrics):
        check(f"metrics[{key}]", ref_metrics[key], tp_metrics[key], atol=1e-4, rtol=1e-5)

    print("  ->", "ALL MATCH" if not failures else f"MISMATCH in {failures}")
    dist.destroy_process_group()
    if failures:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--beta", type=float, default=0.5, help="--opd-jsd-beta to exercise")
    parser.add_argument("--topk-overlap", action="store_true", help="also check the top-k overlap metric")
    parser.add_argument(
        "--teacher-vocab-size",
        type=int,
        default=500,
        help=(
            f"Teacher vocab, against a padded student vocab of {PADDED_VOCAB_SIZE}. The default "
            "leaves the last rank partly on padding; drop it well below PADDED_VOCAB_SIZE/tp_size "
            "to give the trailing ranks a zero-width teacher shard."
        ),
    )
    parser.add_argument("--port", type=int, default=int(os.environ.get("VERIFY_OPD_PORT", 29517)))
    cli = parser.parse_args()

    assert PADDED_VOCAB_SIZE % cli.tp_size == 0, "--tp-size must divide the padded vocab size"
    assert 0 < cli.teacher_vocab_size <= PADDED_VOCAB_SIZE
    mp.spawn(
        worker,
        args=(cli.tp_size, cli.port, cli.beta, cli.topk_overlap, cli.teacher_vocab_size),
        nprocs=cli.tp_size,
        join=True,
    )


if __name__ == "__main__":
    main()
