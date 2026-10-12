"""Experiment 2 — constrained routing quality.

Experiment 1 showed that loading missed experts from flash costs ~200 MB/token, which a phone can't afford.
So route each token ONLY among a resident set of K experts per layer (zero flash reads during decode)
and measure the quality cost — against the full MoE, and against a dense model at equal RAM.

Configs (MoE model): full | task-K48 | task-K32 | global-K32
  task   = resident set calibrated on the same task (what Cortex would preload)
  global = one task-agnostic set (plain static expert pruning)
Dense model: run as-is for the equal-RAM baseline.

Metrics: GSM8K accuracy (generation), HumanEval canonical-solution NLL/token (teacher-forced, no code execution).
All Qwen3 models share a tokenizer, so per-token NLL is comparable across them.

Usage:
  python experiments/routing_quality.py --model mlx-community/Qwen3-30B-A3B-4bit
  python experiments/routing_quality.py --model mlx-community/Qwen3-8B-4bit
"""
import argparse
import glob
import json
import math
import re
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download
from mlx.utils import tree_flatten
from mlx_lm import generate, load
from mlx_lm.models import qwen3_moe

import expert_locality as el

N_CALIB_MATH, N_MATH, N_CALIB_CODE = 50, 100, 40  # ponytail: small subsets; full GSM8K test (1319) for the paper
RESULTS = Path(__file__).resolve().parent.parent / "results"
MATH_SUFFIX = "\nSolve step by step, then give the final line as 'Answer: <number>'."

_mask = {}  # layer -> additive router mask; empty = unconstrained


def patch_router(blocks, layer_of):
    """Qwen3MoeSparseMoeBlock.__call__ with an additive mask on router logits (same math otherwise)."""
    lid = {id(b): layer_of[id(b.switch_mlp)] for b in blocks}

    def masked_call(self, x):
        gates = self.gate(x)
        m = _mask.get(lid[id(self)])
        if m is not None:
            gates = gates + m
        gates = mx.softmax(gates, axis=-1, precise=True)
        k = self.top_k
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores /= mx.sum(scores, axis=-1, keepdims=True)
        y = self.switch_mlp(x, inds)
        return (y * scores[..., None]).sum(axis=-2)

    qwen3_moe.Qwen3MoeSparseMoeBlock.__call__ = masked_call


def set_mask(counts, k, n_experts):
    _mask.clear()
    if counts is None:
        return
    for layer, keep in enumerate(el.top_k(counts, k)):
        m = np.full(n_experts, -1e9, np.float32)
        m[keep] = 0
        _mask[layer] = mx.array(m)


def routing_counts(n_layers, n_experts):
    pre, dec, _ = el.counts_from_calls(n_layers, n_experts)
    return pre + dec


def chat(tok, text):
    return tok.apply_chat_template([{"role": "user", "content": text}],
                                   add_generation_prompt=True, enable_thinking=False)


def number(s):
    s = s.split("Answer:")[-1] if "Answer:" in s else s
    nums = re.findall(r"-?\d[\d,]*\.?\d*", s)
    return float(nums[-1].replace(",", "").rstrip(".")) if nums else None


def gsm8k(model, tok, rows, record=None):
    correct = 0
    for q, a in rows:
        out = generate(model, tok, chat(tok, q + MATH_SUFFIX), max_tokens=400)
        pred, ref = number(out), number(a.split("####")[-1])
        correct += pred is not None and abs(pred - ref) < 1e-6
        if record:
            record()
        else:
            el._calls.clear()
    return correct / len(rows)


def code_nll(model, tok, rows, record=None):
    """Mean NLL per token of the canonical solution given the prompt (teacher-forced)."""
    nll, n = 0.0, 0
    for prompt, solution in rows:
        split = len(tok.encode(prompt))  # ponytail: assumes prompt tokens are a prefix of prompt+solution tokens
        ids = tok.encode(prompt + solution)
        logits = model(mx.array(ids)[None])[0].astype(mx.float32)
        logp = logits[:-1] - mx.logsumexp(logits[:-1], axis=-1, keepdims=True)
        tgt = mx.array(ids[1:])
        tok_lp = mx.take_along_axis(logp, tgt[:, None], axis=-1)[:, 0]
        nll -= tok_lp[split - 1:].sum().item()
        n += len(ids) - split
        if record:
            record()
        else:
            el._calls.clear()
    return nll / n


def load_data():
    g = snapshot_download("openai/gsm8k", repo_type="dataset", allow_patterns=["main/*"])
    train = pd.read_parquet(glob.glob(f"{g}/main/train-*.parquet")[0])
    test = pd.read_parquet(glob.glob(f"{g}/main/test-*.parquet")[0])
    h = snapshot_download("openai/openai_humaneval", repo_type="dataset")
    he = pd.read_parquet(glob.glob(f"{h}/**/*.parquet", recursive=True)[0])
    math_calib = list(zip(train.question[:N_CALIB_MATH], train.answer[:N_CALIB_MATH]))
    math_eval = list(zip(test.question[:N_MATH], test.answer[:N_MATH]))
    code = list(zip(he.prompt, he.canonical_solution))
    return math_calib, math_eval, code[:N_CALIB_CODE], code[N_CALIB_CODE:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    name = args.model.split("/")[-1]
    math_calib, math_eval, code_calib, code_eval = load_data()

    model, tok = load(args.model)
    total = sum(v.nbytes for _, v in tree_flatten(model.parameters()))
    blocks = [m for _, m in model.named_modules() if isinstance(m, qwen3_moe.Qwen3MoeSparseMoeBlock)]

    if not blocks:  # dense baseline
        configs, ram = {"dense": None}, {"dense": total}
    else:
        mods, expert_bytes = el.instrument(model)
        L, E = len(mods), mods[0].up_proj.num_experts
        patch_router(blocks, {id(m): i for i, m in enumerate(mods)})

        # calibration: routing counts per task (full model, no mask)
        print("calibrating math…", flush=True)
        math_counts = np.zeros((L, E), np.int64)
        def add_math():
            nonlocal math_counts
            math_counts += routing_counts(L, E)
        gsm8k(model, tok, math_calib, record=add_math)
        print("calibrating code…", flush=True)
        code_counts = np.zeros((L, E), np.int64)
        def add_code():
            nonlocal code_counts
            code_counts += routing_counts(L, E)
        code_nll(model, tok, code_calib, record=add_code)
        exp1 = json.loads((RESULTS / f"expert_locality_{name}.json").read_text())
        exp1_counts = sum(np.array(r["decode"]) for r in exp1["runs"])
        norm = lambda c: c / c.sum(axis=1, keepdims=True)
        global_counts = norm(math_counts) + norm(code_counts) + norm(exp1_counts)

        # self-check: with a mask on, routing never leaves the resident set
        set_mask(math_counts, 32, E)
        model(mx.array(tok.encode("What is 17 * 23?"))[None])
        used = routing_counts(L, E) > 0
        resident = np.zeros((L, E), bool)
        np.put_along_axis(resident, el.top_k(math_counts, 32), True, axis=1)
        assert not (used & ~resident).any(), "mask leaked: routed to a non-resident expert"

        non_expert = total - expert_bytes
        configs = {"full": None, "task-K48": ("task", 48), "task-K32": ("task", 32), "global-K32": ("global", 32)}
        ram = {c: total if v is None else non_expert + expert_bytes * v[1] / E for c, v in configs.items()}

    out_path = RESULTS / "routing_quality.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    for cfg, spec in configs.items():
        t0 = time.time()
        for bench, counts_for_task in (("gsm8k", "math"), ("code", "code")):
            if spec:
                src = global_counts if spec[0] == "global" else (math_counts if counts_for_task == "math" else code_counts)
                set_mask(src, spec[1], E)
            else:
                _mask.clear()
            if bench == "gsm8k":
                acc = gsm8k(model, tok, math_eval)
            else:
                nll = code_nll(model, tok, code_eval)
        _mask.clear()
        key = f"{name}|{cfg}"
        results[key] = {"ram_gb": ram[cfg] / 1e9, "gsm8k_acc": acc, "n_gsm8k": len(math_eval),
                        "code_nll": nll, "code_ppl": math.exp(nll), "n_code": len(code_eval),
                        "minutes": (time.time() - t0) / 60}
        print(f"{key:32} RAM {ram[cfg] / 1e9:5.1f} GB  GSM8K {acc:6.1%}  code NLL {nll:.3f} "
              f"(ppl {math.exp(nll):.2f})  [{results[key]['minutes']:.1f} min]", flush=True)
        out_path.write_text(json.dumps(results, indent=1))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
