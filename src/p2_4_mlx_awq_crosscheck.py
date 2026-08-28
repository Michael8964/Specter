"""P2.4 (Mac part) -- mlx_lm.awq cross-check.

project_plan_v9.md SS7 P2.4 / SS3 non-goal 5 / SS8's quant-baseline-tooling row: Mac cannot produce a real
quantized-inference speed number through the PyTorch/HF stack (SS9.1 Risk B --
PyTorch's legacy `torch.quantize_per_tensor` "Quantized" backend was never
ported to MPS). `mlx-lm` is the one named exception: it has its own AWQ
implementation (`mlx_lm.quant.awq`, exposed as the `mlx_lm.awq` console
script) that quantizes and runs entirely through MLX's own Metal kernels,
never touching PyTorch's quantized backend. This script is a thin driver
around that existing tool: it does not reimplement AWQ (that already
happened, by hand, in awq_quantize.py for P2.1 -- a different exercise with a
different goal, see PR #4 / P2.0-P2.3), it just shells out to the real tool
and records real speed/memory numbers before/after.

This number is a free cross-validation reference point ONLY. It is not a
substitute for the cloud-stage LLM Compressor GPTQ comparison (SS3 non-goal 5,
SS8's quant-baseline-tooling row).

Two phases, run as separate subprocesses so each gets a clean process (no
Metal cache / peak-memory carryover between the bf16 and quantized model, and
so `ru_maxrss` reflects one model's process, not both):

  1. `--phase quantize` -- shell out to the `mlx_lm.awq` console script.
  2. `--phase bench`    -- load one model, run batch=1 generation N_RUNS times
     per prompt (after one untimed warmup run per prompt), record mlx-lm's
     own `GenerationResponse` stats (prompt_tps / generation_tps /
     peak_memory) plus the process's actual RSS.

With no --phase, orchestrates both phases for baseline vs. AWQ and writes the
combined result JSON.
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

# --- config -------------------------------------------------------------

# Same target model as P1.0 / P2.1 (Qwen/Qwen2.5-3B-Instruct), cross-experiment
# comparable. mlx-community already publishes a pre-converted mlx bf16 build of
# this exact HF repo, so no manual `mlx_lm.convert` step was needed -- verified
# by `huggingface-cli scan-cache` / local HF cache before writing this script.
HF_SOURCE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
BASELINE_MLX_MODEL = "mlx-community/Qwen2.5-3B-Instruct-bf16"

RESULTS_DIR = Path(__file__).resolve().parent / "results"
QUANT_MODEL_DIR = RESULTS_DIR / "p2_4_awq_mlx_model"
RESULT_PATH = RESULTS_DIR / "p2_4_mlx_awq_result.json"

# bits/group_size aligned with P2.1's hand-written AWQ (bits=4, group_size=128)
# for cross-experiment comparability -- NOT the mlx_lm.awq CLI's own default
# group_size (64). Everything else below started from the mlx_lm.awq CLI's own
# defaults, per "borrow the tool's own defaults" (project_plan_v9.md SS7
# P2.4): calibration set is mlx-lm's built-in `calibration_data_v5_rc.txt`
# (~/.cache/mlx-lm/calibration_v5.txt, auto-downloaded from
# https://gist.githubusercontent.com/tristandruyen/9e207a95c7d75ddf37525d353e00659c/raw/571fda718462de863e5a0171078c175420c7649a/calibration_data_v5_rc.txt
# on first use), random non-overlapping token chunks from that text, AWQ
# per-channel scale search grid = 20 points, seed = 123. Source:
# mlx_lm/quant/awq.py `main()` argparse defaults and mlx_lm/quant/utils.py
# `load_data()`, mlx-lm 0.31.3.
#
# NUM_SAMPLES/SEQUENCE_LENGTH deviate from the CLI defaults (128 / 512): a
# first run with the CLI defaults was killed after 51 min at ~18GB resident
# and climbing, thrashing this 24GB Mac's swap. (This is a pitfall specific
# to mlx_lm.awq's own implementation, discovered while running this script --
# it is not one of project_plan_v9.md's numbered 坑1-12; 坑8 there is about
# draft-model/KV-cache memory contention, a different issue.)
# Root cause (read from mlx_lm/quant/awq.py awq_quantize(), not guessed): the
# per-layer `input_feat` calibration-activation cache is attached as an
# attribute directly on each Linear submodule object, which is still
# referenced by the live `model`, so it is never freed as the block-by-block
# loop advances -- by block 36/36 of Qwen2.5-3B, all 36 blocks' captured
# activations (sized num_samples * sequence_length * hidden_or_intermediate_dim
# each) are resident simultaneously. This is inherent to the tool's own
# implementation, not a bug in this script, and not something in scope to
# patch (P2.4 borrows the tool as-is). Cutting NUM_SAMPLES 128->32 and
# SEQUENCE_LENGTH 512->256 is an 8x reduction in that footprint.
AWQ_BITS = 4
AWQ_GROUP_SIZE = 128
AWQ_NUM_SAMPLES = 32
AWQ_SEQUENCE_LENGTH = 256
AWQ_N_GRID = 20
AWQ_SEED = 123

PROMPTS = [
    "Write a Python function that returns the nth Fibonacci number.",
    "Explain the difference between TCP and UDP in two sentences.",
    "Summarize the plot of Romeo and Juliet in three sentences.",
]
MAX_TOKENS = 128
N_RUNS = 3  # research-integrity.md Risk 2: report mean +/- std over >=3 runs, not best-of-N


def run_quantize():
    """Shell out to the real `mlx_lm.awq` console script -- no reimplementation."""
    if (QUANT_MODEL_DIR / "config.json").exists():
        print(f"[quantize] {QUANT_MODEL_DIR} already exists, skipping (delete it to re-quantize)")
        return 0.0

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    awq_bin = Path(sys.executable).parent / "mlx_lm.awq"
    cmd = [
        str(awq_bin),
        "--model", BASELINE_MLX_MODEL,
        "--mlx-path", str(QUANT_MODEL_DIR),
        "--bits", str(AWQ_BITS),
        "--group-size", str(AWQ_GROUP_SIZE),
        "--num-samples", str(AWQ_NUM_SAMPLES),
        "--sequence-length", str(AWQ_SEQUENCE_LENGTH),
        "--n-grid", str(AWQ_N_GRID),
        "--seed", str(AWQ_SEED),
    ]
    print(f"[quantize] {' '.join(cmd)}")
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True)
    elapsed = time.perf_counter() - t0
    print(f"[quantize] done in {elapsed:.1f}s -> {QUANT_MODEL_DIR}")
    return elapsed


def run_bench(model_path: str, label: str):
    """Runs in its own subprocess (see main()). Prints one JSON line to stdout."""
    import mlx.core as mx
    from mlx_lm.generate import stream_generate
    from mlx_lm.utils import load

    model, tokenizer = load(model_path)

    per_prompt = []
    for prompt in PROMPTS:
        # one untimed warmup run to absorb first-call / Metal-compile overhead
        for _ in stream_generate(model, tokenizer, prompt, max_tokens=MAX_TOKENS):
            pass

        runs = []
        for _ in range(N_RUNS):
            mx.reset_peak_memory()
            t0 = time.perf_counter()
            last = None
            for resp in stream_generate(model, tokenizer, prompt, max_tokens=MAX_TOKENS):
                last = resp
            wall_s = time.perf_counter() - t0
            runs.append({
                "wall_s": wall_s,
                "prompt_tokens": last.prompt_tokens,
                "prompt_tps": last.prompt_tps,
                "generation_tokens": last.generation_tokens,
                "generation_tps": last.generation_tps,
                "mlx_peak_memory_gb": last.peak_memory,
            })
        per_prompt.append({"prompt": prompt, "runs": runs})

    def agg(key):
        vals = [r[key] for p in per_prompt for r in p["runs"]]
        return {
            "mean": statistics.mean(vals),
            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "n": len(vals),
        }

    ru_maxrss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() != "Darwin":
        ru_maxrss_bytes *= 1024  # ru_maxrss is KB on Linux, bytes on macOS

    result = {
        "label": label,
        "model_path": model_path,
        "per_prompt": per_prompt,
        "aggregate": {
            "generation_tps": agg("generation_tps"),
            "prompt_tps": agg("prompt_tps"),
            "mlx_peak_memory_gb": agg("mlx_peak_memory_gb"),
        },
        "process_actual_rss_gb": ru_maxrss_bytes / 1e9,
    }
    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["quantize", "bench"], default=None)
    parser.add_argument("--model", default=None, help="bench phase only")
    parser.add_argument("--label", default=None, help="bench phase only")
    args = parser.parse_args()

    if args.phase == "quantize":
        run_quantize()
        return
    if args.phase == "bench":
        run_bench(args.model, args.label)
        return

    # orchestrate: quantize once, then bench baseline and quantized each in a
    # fresh subprocess so peak-memory/RSS numbers don't cross-contaminate.
    quantize_elapsed_s = run_quantize()

    def bench_subprocess(model_path, label):
        print(f"[bench] {label}: {model_path}")
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()),
             "--phase", "bench", "--model", model_path, "--label", label],
            check=True, capture_output=True, text=True,
        )
        # run_bench prints exactly one JSON line; forward any other stdout
        # (mlx-lm progress bars etc.) to our own stderr for visibility.
        json_line = None
        for line in proc.stdout.splitlines():
            line = line.strip()
            if line.startswith("{"):
                json_line = line
            elif line:
                print(f"  [{label}] {line}", file=sys.stderr)
        if proc.stderr:
            print(proc.stderr, file=sys.stderr)
        return json.loads(json_line)

    baseline = bench_subprocess(BASELINE_MLX_MODEL, "baseline_bf16")
    quantized = bench_subprocess(str(QUANT_MODEL_DIR), "awq_int4_g128")

    result = {
        "task": "P2.4 (Mac part) -- mlx_lm.awq cross-check",
        "note": (
            "This number is a free Mac-side cross-validation reference point "
            "only. It does NOT substitute for the cloud-stage LLM Compressor "
            "GPTQ comparison (project_plan_v9.md SS3 non-goal 5, SS8's "
            "quant-baseline-tooling row)."
        ),
        "hf_source_model": HF_SOURCE_MODEL,
        "baseline_mlx_model": BASELINE_MLX_MODEL,
        "quant_config": {
            "bits": AWQ_BITS,
            "group_size": AWQ_GROUP_SIZE,
            "group_size_note": (
                "overridden from mlx_lm.awq CLI default (64) to 128 to match "
                "P2.1's hand-written AWQ config for cross-experiment comparability"
            ),
            "num_samples": AWQ_NUM_SAMPLES,
            "sequence_length": AWQ_SEQUENCE_LENGTH,
            "n_grid": AWQ_N_GRID,
            "seed": AWQ_SEED,
            "calibration_set": (
                "mlx_lm.quant.utils.load_data() source: mlx-lm's built-in "
                "calibration_data_v5_rc.txt, auto-downloaded to "
                "~/.cache/mlx-lm/calibration_v5.txt from "
                "https://gist.githubusercontent.com/tristandruyen/9e207a95c7d75ddf37525d353e00659c/raw/"
                "571fda718462de863e5a0171078c175420c7649a/calibration_data_v5_rc.txt "
                f"on first use; {AWQ_NUM_SAMPLES} random non-overlapping "
                f"{AWQ_SEQUENCE_LENGTH}-token chunks drawn from it (sample "
                "count/length cut from the CLI's own defaults of 128/512, "
                "see AWQ_NUM_SAMPLES/AWQ_SEQUENCE_LENGTH comment above for "
                "why). NOT the C4/codeparrot set P2.1 used -- this is the "
                "tool's own calibration text, used as-is per the 'borrow "
                "the existing tool's own defaults' instruction."
            ),
            "quantize_wall_s": quantize_elapsed_s,
        },
        "bench_config": {
            "prompts": PROMPTS,
            "max_tokens": MAX_TOKENS,
            "n_runs_per_prompt": N_RUNS,
            "batch_size": 1,
        },
        "platform": {
            "machine": platform.machine(),
            "processor": platform.processor(),
            "platform": platform.platform(),
        },
        "baseline_bf16": baseline,
        "awq_int4_g128": quantized,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, indent=2))
    print(f"\n[result] written to {RESULT_PATH}")

    b_tps = baseline["aggregate"]["generation_tps"]
    q_tps = quantized["aggregate"]["generation_tps"]
    b_mem = baseline["process_actual_rss_gb"]
    q_mem = quantized["process_actual_rss_gb"]
    print(
        f"[summary] generation tok/s: baseline {b_tps['mean']:.2f} +/- {b_tps['std']:.2f} "
        f"-> awq {q_tps['mean']:.2f} +/- {q_tps['std']:.2f}"
    )
    print(f"[summary] process RSS (GB): baseline {b_mem:.2f} -> awq {q_mem:.2f}")


if __name__ == "__main__":
    main()
