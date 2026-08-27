"""P4.0/P4.1 (B track) -- quantization throughput curve + memory footprint, per batch size.

project_plan_v9.md SS7 "支柱4：Batch Size 交叉点实验", P4.0 asks for one throughput
curve over batch in {1,4,8,16,32,64} from speculative decoding (A's track) and one
from quantization (this script, B's track); P4.1 asks each curve to also record its
own memory footprint. This script only produces B's curve. It does not attempt to
merge with A's curve or locate the cross-over point -- that is explicitly the
"[A+B 联合]" TASKS.md line, blocked on A's P1 work, out of scope here.

Field-naming note (departure from Appendix C1, project_plan_v9.md line ~484):
the plan's own batch-scan schema is `batch_size, throughput_tokens_per_sec,
draft_model_memory_mb, kv_cache_memory_mb, speedup_ratio_vs_no_spec` -- written
for A's speculative-decoding curve (draft model + KV cache are spec-decoding
concepts; "vs_no_spec" is a spec-decoding baseline). None of those fields have a
quantization equivalent. This script instead reports, per batch size:
`batch_size, baseline_tps, awq_tps, baseline_peak_memory_gb, awq_peak_memory_gb,
speedup_ratio_vs_baseline` -- baseline here is the unquantized bf16 model at the
same batch size (the quantization-relevant "no-compression" comparison), and
memory is mlx's own peak allocator figure (GB, not MB, since that's the unit
`mlx.core.get_peak_memory()` / `GenerationResponse.peak_memory` already reports).

Model source: reuses the AWQ model already produced by P2.4
(p2_4_mlx_awq_crosscheck.py's run_quantize(), src/results/p2_4_awq_mlx_model/) --
not awq_quantize.py (P2.1's hand-written fake-quantize, PyTorch/MPS float
arithmetic that never produces a real quantized tensor or a real speed number,
per SS9.1 Risk B / SS3 non-goal 5). mlx_lm's own AWQ path is the only one on this
Mac that runs through real Metal int4 kernels, so it is the only one that can
produce a real throughput number to build a curve out of.

Batch mechanics: unlike P2.4's bench (batch=1 only, real text prompts, one
`stream_generate` call per prompt), batch>1 here uses `mlx_lm.batch_generate`
over synthetic uniform-length random-token prompts -- the same approach
`mlx_lm.benchmark` (mlx-lm's own bundled benchmarking CLI, .venv's
mlx_lm/benchmark.py) uses, and for the same reason: real text prompts have
different token lengths, which would force ragged padding and dirty the
per-batch-size throughput number with padding overhead rather than measuring
the quantization effect alone. EOS is disabled (tokenizer._eos_token_ids = {})
so every sample in a batch runs the full max_tokens and batch shape stays
constant for the whole call, again matching mlx_lm.benchmark's approach.

batch=1 point: reused as-is from src/results/p2_4_mlx_awq_result.json
(baseline_bf16 / awq_int4_g128 aggregate.generation_tps mean, and
process_actual_rss_gb) rather than re-run, per the batch=1 already having a
real measured number (mlx_lm.benchmark itself also special-cases batch=1 to a
plain stream_generate path, so this isn't a methodology detour). Caveat this
carries: that point used real short prompts (11-13 tokens) and max_tokens=128,
this script's batch>=4 points use synthetic 128-token prompts and max_tokens=128
-- prompt length source differs, generation length matches. Noted in the output
JSON, not hidden.

OOM/thrashing handling (SS9.6 risk 2 -- research integrity, don't silently
retry-until-nice; and this Mac's own prior incident, see P2.4's AWQ_NUM_SAMPLES
comment, where a real run thrashed 24GB of swap): each batch size's bench runs
in its own subprocess with a wall-clock timeout. A timeout is treated as
suspected swap thrashing (large batches degrade to swap-speed, not just "a bit
slower") and recorded with status="timeout_suspected_thrashing" rather than
silently retried or dropped from the batch list; a non-zero exit (e.g. mlx
Metal allocation failure) is recorded with status="oom". Batch sizes are run in
increasing order and the sweep stops at the first failure -- larger batch sizes
are strictly worse for memory, so continuing past a failure would not add
information, only wasted wall time.
"""

import argparse
import json
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"
QUANT_MODEL_DIR = RESULTS_DIR / "p2_4_awq_mlx_model"
P2_4_RESULT_PATH = RESULTS_DIR / "p2_4_mlx_awq_result.json"
RESULT_PATH = RESULTS_DIR / "p4_quant_throughput_result.json"

BASELINE_MLX_MODEL = "mlx-community/Qwen2.5-3B-Instruct-bf16"

BATCH_SIZES = [1, 4, 8, 16, 32, 64]
PROMPT_TOKENS = 128
GENERATION_TOKENS = 128  # matches P2.4's MAX_TOKENS, for the batch=1 reuse to be comparable
N_TRIALS = 3  # SS9.6 risk 2: mean +/- std over >=3 runs, not best-of-N
SEED = 123  # same seed as P2.4's AWQ_SEED, reused here for the synthetic prompt RNG too

# Generous but bounded: a healthy run (batch<=64 comfortably fits 24GB unified
# memory) finishes a warmup + N_TRIALS trials in well under a minute per model.
# Swapping degrades wall time by orders of magnitude, so a timeout this size
# only trips under genuine thrashing, not ordinary variance.
SUBPROCESS_TIMEOUT_S = 300


def run_bench(model_path: str, label: str, batch_size: int):
    """Runs in its own subprocess (see main()). Prints one JSON line to stdout."""
    import mlx.core as mx
    from mlx_lm import batch_generate
    from mlx_lm.generate import stream_generate
    from mlx_lm.utils import load

    model, tokenizer = load(model_path)
    tokenizer._eos_token_ids = {}  # keep batch shape constant for the whole call, matches mlx_lm.benchmark

    mx.random.seed(SEED)
    vocab_size = tokenizer.vocab_size
    prompts = mx.random.randint(0, vocab_size, (batch_size, PROMPT_TOKENS)).tolist()

    def one_call():
        if batch_size == 1:
            last = None
            for resp in stream_generate(model, tokenizer, prompts[0], max_tokens=GENERATION_TOKENS):
                last = resp
            return {
                "prompt_tps": last.prompt_tps,
                "generation_tps": last.generation_tps,
                "peak_memory_gb": last.peak_memory,
            }
        else:
            stats = batch_generate(model, tokenizer, prompts, max_tokens=GENERATION_TOKENS).stats
            return {
                "prompt_tps": stats.prompt_tps,
                "generation_tps": stats.generation_tps,
                "peak_memory_gb": stats.peak_memory,
            }

    one_call()  # untimed warmup, absorbs first-call / Metal-compile overhead

    runs = []
    for _ in range(N_TRIALS):
        mx.reset_peak_memory()
        t0 = time.perf_counter()
        r = one_call()
        r["wall_s"] = time.perf_counter() - t0
        runs.append(r)

    def agg(key):
        vals = [r[key] for r in runs]
        return {"mean": statistics.mean(vals), "std": statistics.stdev(vals) if len(vals) > 1 else 0.0, "n": len(vals)}

    ru_maxrss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() != "Darwin":
        ru_maxrss_bytes *= 1024  # ru_maxrss is KB on Linux, bytes on macOS

    result = {
        "label": label,
        "model_path": model_path,
        "batch_size": batch_size,
        "runs": runs,
        "aggregate": {
            "generation_tps": agg("generation_tps"),
            "prompt_tps": agg("prompt_tps"),
            "mlx_peak_memory_gb": agg("peak_memory_gb"),
        },
        "process_actual_rss_gb": ru_maxrss_bytes / 1e9,
    }
    print(json.dumps(result))


def bench_subprocess(model_path: str, label: str, batch_size: int):
    """Fresh subprocess per (model, batch_size): no Metal cache / peak-memory
    carryover between runs, and a hung/thrashing run can be timed out and
    killed without taking this driver process down with it."""
    print(f"[bench] {label} batch={batch_size}: {model_path}", file=sys.stderr)
    try:
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()),
             "--phase", "bench", "--model", model_path, "--label", label, "--batch-size", str(batch_size)],
            check=False, capture_output=True, text=True, timeout=SUBPROCESS_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as e:
        print(f"  [{label} b={batch_size}] TIMEOUT after {SUBPROCESS_TIMEOUT_S}s "
              f"(suspected swap thrashing, see module docstring)", file=sys.stderr)
        # e.stderr isn't guaranteed to be decoded to str on the timeout path
        # (unlike a normal completed proc, where text=True always applies) --
        # decode defensively so a timeout can't also crash the JSON write.
        stderr_tail = e.stderr.decode(errors="replace") if isinstance(e.stderr, bytes) else e.stderr
        return {"status": "timeout_suspected_thrashing", "label": label, "batch_size": batch_size,
                "timeout_s": SUBPROCESS_TIMEOUT_S, "stderr_tail": (stderr_tail or "")[-2000:] if stderr_tail else None}

    json_line = None
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            json_line = line
        elif line:
            print(f"  [{label} b={batch_size}] {line}", file=sys.stderr)

    if proc.returncode != 0 or json_line is None:
        print(f"  [{label} b={batch_size}] FAILED, returncode={proc.returncode}", file=sys.stderr)
        if proc.stderr:
            print(proc.stderr[-4000:], file=sys.stderr)
        return {"status": "oom", "label": label, "batch_size": batch_size,
                "returncode": proc.returncode, "stderr_tail": (proc.stderr or "")[-2000:]}

    result = json.loads(json_line)
    result["status"] = "ok"
    return result


def load_batch1_from_p2_4():
    """Reuse P2.4's already-measured batch=1 point instead of re-running it (see
    module docstring for why this is methodologically fine, and its caveat)."""
    p2_4 = json.loads(P2_4_RESULT_PATH.read_text())
    baseline = p2_4["baseline_bf16"]
    awq = p2_4["awq_int4_g128"]
    return (
        {
            "status": "ok", "label": "baseline_bf16", "model_path": p2_4["baseline_mlx_model"],
            "batch_size": 1,
            "aggregate": {
                "generation_tps": baseline["aggregate"]["generation_tps"],
                "prompt_tps": baseline["aggregate"]["prompt_tps"],
                "mlx_peak_memory_gb": baseline["aggregate"]["mlx_peak_memory_gb"],
            },
            "process_actual_rss_gb": baseline["process_actual_rss_gb"],
            "reused_from": str(P2_4_RESULT_PATH.relative_to(Path(__file__).resolve().parent.parent)),
            "reuse_caveat": "real short text prompts (11-13 tokens), not this script's synthetic 128-token prompts; max_tokens=128 matches",
        },
        {
            "status": "ok", "label": "awq_int4_g128", "model_path": str(QUANT_MODEL_DIR),
            "batch_size": 1,
            "aggregate": {
                "generation_tps": awq["aggregate"]["generation_tps"],
                "prompt_tps": awq["aggregate"]["prompt_tps"],
                "mlx_peak_memory_gb": awq["aggregate"]["mlx_peak_memory_gb"],
            },
            "process_actual_rss_gb": awq["process_actual_rss_gb"],
            "reused_from": str(P2_4_RESULT_PATH.relative_to(Path(__file__).resolve().parent.parent)),
            "reuse_caveat": "real short text prompts (11-13 tokens), not this script's synthetic 128-token prompts; max_tokens=128 matches",
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["bench"], default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--label", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    if args.phase == "bench":
        run_bench(args.model, args.label, args.batch_size)
        return

    if not (QUANT_MODEL_DIR / "config.json").exists():
        print(f"[error] {QUANT_MODEL_DIR} not found -- run p2_4_mlx_awq_crosscheck.py first "
              f"(this script reuses that model, it does not quantize its own)", file=sys.stderr)
        sys.exit(1)

    points = []
    b1_baseline, b1_awq = load_batch1_from_p2_4()
    points.append({"batch_size": 1, "baseline": b1_baseline, "awq": b1_awq})

    for batch_size in BATCH_SIZES:
        if batch_size == 1:
            continue
        baseline = bench_subprocess(BASELINE_MLX_MODEL, "baseline_bf16", batch_size)
        if baseline["status"] != "ok":
            points.append({"batch_size": batch_size, "baseline": baseline, "awq": None})
            print(f"[stop] batch={batch_size} baseline failed ({baseline['status']}); "
                  f"not attempting larger batch sizes (see module docstring)", file=sys.stderr)
            break
        awq = bench_subprocess(str(QUANT_MODEL_DIR), "awq_int4_g128", batch_size)
        points.append({"batch_size": batch_size, "baseline": baseline, "awq": awq})
        if awq["status"] != "ok":
            print(f"[stop] batch={batch_size} awq failed ({awq['status']}); "
                  f"not attempting larger batch sizes (see module docstring)", file=sys.stderr)
            break

    curve = []
    for p in points:
        b, a = p["baseline"], p["awq"]
        row = {
            "batch_size": p["batch_size"],
            "baseline_status": b["status"] if b else None,
            "awq_status": a["status"] if a else None,
        }
        if b and b["status"] == "ok":
            row["baseline_tps"] = b["aggregate"]["generation_tps"]["mean"]
            row["baseline_tps_std"] = b["aggregate"]["generation_tps"]["std"]
            row["baseline_peak_memory_gb"] = b["aggregate"]["mlx_peak_memory_gb"]["mean"]
        if a and a["status"] == "ok":
            row["awq_tps"] = a["aggregate"]["generation_tps"]["mean"]
            row["awq_tps_std"] = a["aggregate"]["generation_tps"]["std"]
            row["awq_peak_memory_gb"] = a["aggregate"]["mlx_peak_memory_gb"]["mean"]
        if row.get("baseline_tps") and row.get("awq_tps"):
            row["speedup_ratio_vs_baseline"] = row["awq_tps"] / row["baseline_tps"]
        curve.append(row)

    result = {
        "task": "P4.0/P4.1 (B track) -- quantization throughput curve + memory footprint",
        "scope_note": (
            "B's half only (quantization curve). A's speculative-decoding curve "
            "(TASKS.md M5 [A] line) is separate and not attempted here; the "
            "[A+B 联合] merge/cross-over-point line is blocked on that and out "
            "of scope for this script."
        ),
        "schema_note": (
            "Departs from project_plan_v9.md Appendix C1's batch-scan schema "
            "(draft_model_memory_mb / kv_cache_memory_mb / speedup_ratio_vs_no_spec), "
            "which is written for A's speculative-decoding curve. See module "
            "docstring for the field mapping rationale."
        ),
        "baseline_mlx_model": BASELINE_MLX_MODEL,
        "awq_model_dir": str(QUANT_MODEL_DIR),
        "bench_config": {
            "batch_sizes_requested": BATCH_SIZES,
            "prompt_tokens": PROMPT_TOKENS,
            "generation_tokens": GENERATION_TOKENS,
            "n_trials": N_TRIALS,
            "seed": SEED,
            "prompt_source": "synthetic uniform-length random token ids (mx.random.randint), matching mlx_lm.benchmark's own methodology -- avoids ragged-padding noise across batch sizes; batch=1 point is the exception, see below",
            "batch1_note": "batch=1 point reused as-is from P2.4's real-text-prompt measurement rather than re-run here, see module docstring",
            "subprocess_timeout_s": SUBPROCESS_TIMEOUT_S,
        },
        "platform": {
            "machine": platform.machine(),
            "processor": platform.processor(),
            "platform": platform.platform(),
        },
        "points": points,
        "curve": curve,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, indent=2))
    print(f"\n[result] written to {RESULT_PATH}")

    print("[summary] batch_size  baseline_tps      awq_tps           speedup   baseline_mem_gb  awq_mem_gb")
    for row in curve:
        b_tps = f"{row['baseline_tps']:.1f}" if "baseline_tps" in row else row["baseline_status"]
        a_tps = f"{row['awq_tps']:.1f}" if "awq_tps" in row else (row["awq_status"] or "-")
        speedup = f"{row['speedup_ratio_vs_baseline']:.2f}x" if "speedup_ratio_vs_baseline" in row else "-"
        b_mem = f"{row['baseline_peak_memory_gb']:.2f}" if "baseline_peak_memory_gb" in row else "-"
        a_mem = f"{row['awq_peak_memory_gb']:.2f}" if "awq_peak_memory_gb" in row else "-"
        print(f"[summary] {row['batch_size']:>10}  {b_tps:>14}  {a_tps:>14}  {speedup:>7}  {b_mem:>15}  {a_mem:>10}")


if __name__ == "__main__":
    main()
