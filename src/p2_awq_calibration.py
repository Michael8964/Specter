"""
P2.0-P2.3 -- AWQ activation-aware calibration on the target model, Mac-local,
$0 budget, HF Transformers + MPS. See notes/project_plan_v9.md §7 支柱2 for
the task spec this implements, and awq_quantize.py for the hand-written
scale-search algorithm itself (P2.0 activation stats + P2.1 scale search /
fake-quantize live there; this file is the experiment driver for P2.1's
headline perplexity number plus the P2.2 cross-distribution matrix and P2.3
calibration-size ablation).

Model choice (draft vs target) -- decided here, not left implicit
------------------------------------------------------------------
Quantizing the TARGET model (Qwen/Qwen2.5-3B-Instruct), not the draft
(Qwen/Qwen2.5-0.5B-Instruct). Reasoning, since the plan doesn't say this in
so many words:

1. §7 P5.2 reuses "P2 的量化模型" to scan optimal gamma for the *same*
   draft/target pair, and frames the result against SpecKV's Table 1 finding
   that compressing the model doing verification shifts the optimal draft
   length upward (FP16 gamma=2 -> INT8 gamma=8). SpecKV's BitsAndBytes arm
   compresses the model that speculative decoding calls once per step to
   verify gamma draft tokens -- i.e. the *target*/verifier, not the drafter.
   For the AWQ arm to be a meaningful analog (even though §9.3 坑10 / ADR-008
   explicitly say not to expect the same 4x magnitude), it has to compress
   the same role: the verifier.
2. System-level reasoning that's consistent with (1): optimal gamma trades
   off draft cost (paid gamma times per step) against verification cost
   (paid once per step, amortized over gamma). Quantizing the target makes
   the *amortized* cost cheaper, which is what should push optimal gamma up
   -- quantizing the draft model instead would cut the per-draft-token cost,
   which is a different (and, since the draft is already small at 0.5B and
   likely memory-bandwidth-bound rather than compute-bound at this size,
   probably smaller) effect.
3. The target model is also the one already gated by P1.0
   (src/results/p1_0_gate_result.json: overall_alpha=0.7024, PASS) and the
   one with the bigger memory footprint to compress (3B vs 0.5B) -- the
   practical "does this even help" story is stronger for it.

Calibration/eval datasets -- decided here
------------------------------------------
- Natural-language calibration: allenai/c4 (en), streaming, first N docs.
  This matches 坑6's "标准实践C4数据集" note directly (known-pitfalls.md 坑6),
  and is also what the AWQ paper itself calibrates on.
- Code calibration: codeparrot/codeparrot-clean-valid, streaming, first N
  files. Raw deduplicated GitHub Python, a different source and register
  from C4's web-crawl text -- this is the P2.2 "clearly different
  distribution" pairing.
- Natural-language eval (perplexity): Salesforce/wikitext,
  wikitext-2-raw-v1, test split. Distinct from the C4 calibration source;
  pairing "calibrate on C4, eval PPL on WikiText-2" is the standard
  AWQ/GPTQ-paper protocol, not something invented for this project.
- Code eval (perplexity): google-research-datasets/mbpp, test split `code`
  field. Deliberately NOT HumanEval -- HumanEval is reserved for this
  project's downstream AgentBench-OS-adjacent evaluation (§3 非目标2 /
  guardrail metric in §5), and reusing it here would contaminate that later
  measurement. MBPP is a different, non-overlapping source from both
  HumanEval and codeparrot-clean (curated benchmark vs raw GitHub scrape).

All four dataset ids were resolved by actually loading them in this
environment before committing to them (Salesforce/wikitext, not the
deprecated bare "wikitext" loader-script id, which errors under the
currently installed `datasets` version).
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from awq_quantize import quantize_model_awq

TARGET_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
DEVICE = "mps"
DTYPE = torch.bfloat16
SEED = 0

SEQ_LEN = 512
BITS = 4
GROUP_SIZE = 128
N_GRID = 20

MAIN_CALIB_N = 128          # P2.1 / P2.2 headline calibration size
ABLATION_CALIB_SIZES = [4, 8, 16, 32, 64, 128]   # P2.3, 坑6: "几十条量级" included
MAX_EVAL_WINDOWS = 40        # cap perplexity eval windows for wall-clock sanity

RESULTS_PATH = Path(__file__).parent / "results" / "p2_awq_calibration_result.json"


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

# Char-length floor before tokenizing: cheap proxy for "long enough to fill a
# SEQ_LEN=512-token window after tokenization" (~2-3 chars/token for English
# and code once whitespace/punctuation are counted). texts_to_calib_batches()
# still applies the real (post-tokenization) length filter below -- this is
# just to avoid pulling in a pool that's mostly too-short docs.
MIN_DOC_CHARS = 2000


def load_c4_texts(n: int) -> list[str]:
    ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
    out = []
    for ex in ds:
        text = ex["text"].strip()
        if len(text) > MIN_DOC_CHARS:
            out.append(text)
        if len(out) >= n:
            break
    return out


def load_code_texts(n: int) -> list[str]:
    ds = load_dataset("codeparrot/codeparrot-clean-valid", split="train", streaming=True)
    out = []
    for ex in ds:
        text = ex["content"].strip()
        if len(text) > MIN_DOC_CHARS:
            out.append(text)
        if len(out) >= n:
            break
    return out


def load_wikitext_eval_text() -> str:
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    return "\n".join(row["text"] for row in ds if row["text"].strip())


def load_mbpp_eval_text() -> str:
    ds = load_dataset("google-research-datasets/mbpp", split="test")
    return "\n\n".join(row["code"] for row in ds)


# ---------------------------------------------------------------------------
# Tokenization / windowing helpers
# ---------------------------------------------------------------------------

def texts_to_calib_batches(texts: list[str], tokenizer, seq_len: int, n_samples: int, device: str, seed: int = SEED):
    rng = random.Random(seed)
    shuffled = texts[:]
    rng.shuffle(shuffled)
    batches = []
    for text in shuffled:
        ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len).input_ids[0]
        if ids.shape[0] < seq_len // 4:  # too short to be a useful calibration sample
            continue
        batches.append(ids.to(device))
        if len(batches) >= n_samples:
            break
    if len(batches) < n_samples:
        raise RuntimeError(f"only found {len(batches)}/{n_samples} usable calibration sequences")
    return batches


@torch.no_grad()
def compute_perplexity(model, tokenizer, eval_text: str, seq_len: int, device: str, max_windows: int) -> dict:
    ids = tokenizer(eval_text, return_tensors="pt").input_ids[0]
    n_windows = min(max_windows, ids.shape[0] // seq_len)
    if n_windows == 0:
        raise RuntimeError("eval text too short for even one full window")

    total_nll = 0.0
    total_tokens = 0
    for w in range(n_windows):
        window = ids[w * seq_len:(w + 1) * seq_len].unsqueeze(0).to(device)
        logits = model(window).logits
        shift_logits = logits[:, :-1, :].float()
        shift_labels = window[:, 1:]
        nll = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            shift_labels.reshape(-1),
            reduction="sum",
        )
        total_nll += nll.item()
        total_tokens += shift_labels.numel()

    ppl = float(torch.exp(torch.tensor(total_nll / total_tokens)))
    return {"perplexity": ppl, "n_windows": n_windows, "n_tokens": total_tokens}


# ---------------------------------------------------------------------------
# Model lifecycle -- reload fresh from local HF cache per arm rather than
# keeping multiple 3B copies or backup weight snapshots resident. Weights
# are already on local disk after the first download, so this is a disk
# read + dtype cast (seconds), not a network round-trip.
# ---------------------------------------------------------------------------

def load_fresh_model():
    model = AutoModelForCausalLM.from_pretrained(TARGET_MODEL_ID, dtype=DTYPE).to(DEVICE).eval()
    return model


def run_quantized_arm(tokenizer, calib_texts: list[str], n_calib: int, eval_texts: dict[str, str]) -> dict:
    model = load_fresh_model()
    calib_batches = texts_to_calib_batches(calib_texts, tokenizer, SEQ_LEN, n_calib, DEVICE)

    t0 = time.time()
    diagnostics = quantize_model_awq(
        model, calib_batches, bits=BITS, group_size=GROUP_SIZE, n_grid=N_GRID, seed=SEED,
    )
    quant_seconds = time.time() - t0

    layer_losses = [d["best_loss"] for d in diagnostics.values()]
    ppl_results = {}
    for eval_name, eval_text in eval_texts.items():
        ppl_results[eval_name] = compute_perplexity(model, tokenizer, eval_text, SEQ_LEN, DEVICE, MAX_EVAL_WINDOWS)

    del model
    if DEVICE == "mps":
        torch.mps.empty_cache()

    return {
        "n_calib": n_calib,
        "quant_seconds": quant_seconds,
        "mean_layer_loss": sum(layer_losses) / len(layer_losses),
        "perplexity": ppl_results,
    }


def main():
    print(f"target model = {TARGET_MODEL_ID}, device = {DEVICE}, dtype = {DTYPE}")
    print(f"bits={BITS} group_size={GROUP_SIZE} n_grid={N_GRID} seq_len={SEQ_LEN}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL_ID)

    print("loading eval sets (wikitext-2 test, mbpp test)...")
    eval_texts = {
        "natural": load_wikitext_eval_text(),
        "code": load_mbpp_eval_text(),
    }

    print("loading calibration source pools (C4 natural, codeparrot code)...")
    natural_pool = load_c4_texts(int(max(MAIN_CALIB_N, max(ABLATION_CALIB_SIZES)) * 1.5) + 20)
    code_pool = load_code_texts(int(MAIN_CALIB_N * 1.5) + 20)

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    result = {
        "target_model": TARGET_MODEL_ID,
        "device": DEVICE,
        "dtype": str(DTYPE),
        "bits": BITS,
        "group_size": GROUP_SIZE,
        "n_grid": N_GRID,
        "seq_len": SEQ_LEN,
        "seed": SEED,
    }

    # --- baseline (fp16/bf16, unquantized) perplexity on both eval sets ---
    print("\n=== baseline (unquantized) perplexity ===")
    baseline_model = load_fresh_model()
    baseline_ppl = {}
    for eval_name, eval_text in eval_texts.items():
        r = compute_perplexity(baseline_model, tokenizer, eval_text, SEQ_LEN, DEVICE, MAX_EVAL_WINDOWS)
        print(f"  baseline ppl [{eval_name}] = {r['perplexity']:.4f}  ({r['n_tokens']} tokens)")
        baseline_ppl[eval_name] = r
    del baseline_model
    if DEVICE == "mps":
        torch.mps.empty_cache()
    result["baseline_perplexity"] = baseline_ppl

    # --- P2.2: cross-distribution robustness matrix (also serves as P2.1's
    # headline number: natural-calibrated -> natural-eval is the
    # same-distribution cell) ---
    print("\n=== P2.2: cross-distribution calibration matrix ===")
    print(f"  calibrating on NATURAL (C4), n={MAIN_CALIB_N}...")
    natural_arm = run_quantized_arm(tokenizer, natural_pool, MAIN_CALIB_N, eval_texts)
    for eval_name, r in natural_arm["perplexity"].items():
        delta = r["perplexity"] - baseline_ppl[eval_name]["perplexity"]
        print(f"    eval=[{eval_name}] ppl={r['perplexity']:.4f}  delta_vs_baseline={delta:+.4f}")

    print(f"  calibrating on CODE (codeparrot), n={MAIN_CALIB_N}...")
    code_arm = run_quantized_arm(tokenizer, code_pool, MAIN_CALIB_N, eval_texts)
    for eval_name, r in code_arm["perplexity"].items():
        delta = r["perplexity"] - baseline_ppl[eval_name]["perplexity"]
        print(f"    eval=[{eval_name}] ppl={r['perplexity']:.4f}  delta_vs_baseline={delta:+.4f}")

    result["p2_2_cross_distribution"] = {
        "calibrated_on_natural": natural_arm,
        "calibrated_on_code": code_arm,
    }

    # --- P2.3: calibration-size ablation (natural/C4 arm, matching 坑6's
    # "standard practice" pairing; eval on the matching natural eval set) ---
    print("\n=== P2.3: calibration-size ablation (natural/C4, eval=natural) ===")
    ablation_runs = []
    for n_calib in ABLATION_CALIB_SIZES:
        print(f"  n_calib={n_calib}...")
        arm = run_quantized_arm(tokenizer, natural_pool, n_calib, {"natural": eval_texts["natural"]})
        ppl = arm["perplexity"]["natural"]["perplexity"]
        delta = ppl - baseline_ppl["natural"]["perplexity"]
        print(f"    ppl={ppl:.4f}  delta_vs_baseline={delta:+.4f}  mean_layer_loss={arm['mean_layer_loss']:.6g}")
        ablation_runs.append(arm)
    result["p2_3_calibration_size_ablation"] = ablation_runs

    RESULTS_PATH.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
