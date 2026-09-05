"""
P2.0 -- real activation-statistics collection for AWQ calibration.

This is the "real" counterpart to scripts/p2_awq_stats_verify.py: instead of
a tiny hand-computed nn.Linear, it loads the actual TARGET model (the one P2
will eventually produce a 4-bit AWQ version of -- see project_plan_v9.md
line 336: "目标模型本身是 P2 阶段自己校准出来的 4-bit AWQ 版本"), runs it
forward over a real calibration set, and saves the per-layer per-channel
activation statistics that P2.1's scaling-factor calibration will consume.

Calibration dataset: C4 (project_plan_v9.md line 232/286 -- "标准实践C4数据
集, group size 128"; group_size is a P2.1 quantization-granularity parameter,
unrelated to how many calibration SAMPLES we collect here, but the paper's
own convention of ~128 calibration samples is what --n-samples defaults to).

Three ways to run this, cheapest first:

  1. --dry-run --n-samples 8
     No network download beyond the model itself (already cached from prior
     milestones). Uses a handful of hardcoded sentences just to prove the
     wiring works: model loads, hooks fire, stats accumulate, files get
     written. Takes seconds. ALWAYS run this first.

  2. --n-samples 16
     A small real run against streamed C4 data, to see actual per-sample
     timing on this machine before committing to the full run.

  3. --n-samples 128  (the default -- the actual calibration run)
     Produces the real activation statistics P2.1 will use.

Output:
  results/p2_awq_activation_stats.pt          -- full stats (all channels,
                                                  as tensors) via torch.save.
                                                  This is what P2.1 loads.
  results/p2_awq_activation_stats_summary.json -- human-readable per-layer
                                                  scalar summary (min/mean/max
                                                  across channels), so the run
                                                  can be sanity-checked and
                                                  committed to git without
                                                  shipping a binary tensor
                                                  blob into every commit diff.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch

from src.awq_calibration import register_activation_hooks, remove_hooks
from src.model_loader import TARGET_MODEL_NAME, get_device, load_model_and_tokenizer

DRY_RUN_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "In 2023, researchers published a new method for compressing large neural networks.",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)",
    "人工智能技术在过去十年中取得了显著进展，尤其是在自然语言处理领域。",
    "The stock market fluctuated wildly after the central bank's announcement.",
    "SELECT customer_id, SUM(amount) FROM orders GROUP BY customer_id ORDER BY 2 DESC;",
    "Photosynthesis is the process by which green plants convert sunlight into chemical energy.",
    "Please summarize the attached document in three bullet points.",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-samples", type=int, default=128,
                    help="number of calibration texts to run through the model (default: 128)")
    p.add_argument("--max-length", type=int, default=512,
                    help="truncate each calibration text to this many tokens (default: 512)")
    p.add_argument("--seed", type=int, default=0, help="shuffle seed for streaming C4 (ignored in --dry-run)")
    p.add_argument("--dry-run", action="store_true",
                    help="skip the C4 download; use a handful of hardcoded sentences to sanity-check the pipeline")
    p.add_argument("--out", type=str, default="results/p2_awq_activation_stats.pt",
                    help="where to save the full stats tensor file")
    return p.parse_args()


def load_calibration_texts(n_samples, seed, dry_run):
    if dry_run:
        reps = (n_samples // len(DRY_RUN_TEXTS)) + 1
        return (DRY_RUN_TEXTS * reps)[:n_samples]

    from datasets import load_dataset

    print("Streaming allenai/c4 (en split)... (first samples may take a moment to start)")
    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10000)

    texts = []
    for row in ds:
        text = (row.get("text") or "").strip()
        if text:
            texts.append(text)
        if len(texts) >= n_samples:
            break
    return texts


@torch.no_grad()
def collect_stats(model, tokenizer, texts, max_length, device):
    handles, stats = register_activation_hooks(model)
    t0 = time.perf_counter()
    n_processed = 0
    try:
        for i, text in enumerate(texts):
            enc = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            if input_ids.shape[1] == 0:
                continue
            model(input_ids=input_ids, use_cache=False)
            n_processed += 1
            if n_processed % 10 == 0 or (i + 1) == len(texts):
                elapsed = time.perf_counter() - t0
                print(f"  processed {n_processed}/{len(texts)} samples "
                      f"({elapsed:.1f}s elapsed, {elapsed / n_processed:.2f}s/sample)")
    finally:
        # Always remove hooks, even if a forward pass raises partway through --
        # a dangling hook on a model object that outlives this function would
        # silently keep accumulating into a stats dict nothing else looks at.
        remove_hooks(handles)
    return stats, n_processed


def save_full_stats(stats, out_path, meta):
    payload = {
        "meta": meta,
        "layers": {
            name: {
                "in_features": s.in_features,
                "sum_abs": s.sum_abs,
                "max_abs": s.max_abs,
                "n_samples": s.n_samples,
            }
            for name, s in stats.items()
        },
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)


def build_json_summary(stats, meta):
    summary = {"meta": meta, "n_layers": len(stats), "layers": {}}
    for name, s in stats.items():
        mean_abs = s.mean_abs()
        summary["layers"][name] = {
            "in_features": s.in_features,
            "n_samples": s.n_samples,
            "mean_abs_min": round(mean_abs.min().item(), 6),
            "mean_abs_avg": round(mean_abs.mean().item(), 6),
            "mean_abs_max": round(mean_abs.max().item(), 6),
            "max_abs_min": round(s.max_abs.min().item(), 6),
            "max_abs_max": round(s.max_abs.max().item(), 6),
        }
    return summary


def main():
    args = parse_args()
    device = get_device()

    print("=" * 70)
    print("Specter -- P2.0 AWQ activation-statistics collection")
    print(f"Target model: {TARGET_MODEL_NAME}")
    print(f"Mode: {'DRY RUN (hardcoded sentences)' if args.dry_run else 'real C4 calibration'}")
    print(f"n_samples={args.n_samples}  max_length={args.max_length}  seed={args.seed}")
    print("=" * 70)

    model, tokenizer = load_model_and_tokenizer(TARGET_MODEL_NAME)

    print("\nLoading calibration texts...")
    texts = load_calibration_texts(args.n_samples, args.seed, args.dry_run)
    print(f"Got {len(texts)} calibration texts.")

    print("\nRunning forward passes and collecting activation stats...")
    stats, n_processed = collect_stats(model, tokenizer, texts, args.max_length, device)

    meta = {
        "target_model": TARGET_MODEL_NAME,
        "calibration_dataset": "dry_run_hardcoded_sentences" if args.dry_run else "allenai/c4 (en, streaming, shuffled)",
        "n_calibration_samples": n_processed,
        "max_length": args.max_length,
        "seed": args.seed,
    }

    save_full_stats(stats, args.out, meta)
    print(f"\nSaved full stats (tensors, {len(stats)} layers) to {args.out}")

    json_path = str(Path(args.out).with_suffix("")) + "_summary.json"
    summary = build_json_summary(stats, meta)
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved human-readable summary to {json_path}")

    print(f"\nDone. Collected activation statistics for {len(stats)} Linear layers "
          f"across {n_processed} calibration samples.")


if __name__ == "__main__":
    main()
