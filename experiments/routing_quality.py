"""Experiment 2 — constrained routing quality (+ rescue variants).

Experiment 1 showed that loading missed experts from flash costs ~200 MB/token, which a phone can't afford.
So route each token ONLY among a resident set of experts (zero flash reads during decode) and measure the
quality cost — against the full MoE, and against a dense model at equal RAM.

Config names:  full | dense | {task,global}-{K,A}<n>[-drop]
  task / global  resident set calibrated on the same task (what Cortex preloads) / one task-agnostic set
  K<n>           n experts in every layer
  A<n>           adaptive: same total budget as K<n> (n × layers slots), allocated greedily to the layers
                 where extra experts cover the most routing mass (floor of top_k per layer)
  -drop          keep the router's original top-k, zero out non-resident picks and renormalize
                 (default "reroute": pick top-k among resident experts only)

Metrics: GSM8K accuracy (generation), HumanEval canonical-solution NLL/token (teacher-forced, no code
execution). All Qwen3 models share a tokenizer, so per-token NLL is comparable across them.

Usage:
  python experiments/routing_quality.py --model mlx-community/Qwen3-8B-4bit
  python experiments/routing_quality.py --model mlx-community/Qwen3-30B-A3B-4bit \
      --configs full,task-K32,task-K32-drop,task-A32-drop [--adapter adapters/x --tag healed]
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
from mlx_lm.models.switch_layers import SwitchGLU

import expert_locality as el

N_CALIB_MATH, N_CALIB_CODE = 50, 40  # GSM8K train[:50] and HumanEval[:40] calibrate the resident sets
RESULTS = Path(__file__).resolve().parent.parent / "results"
MATH_SUFFIX = "\nSolve step by step, then give the final line as 'Answer: <number>'."

_mask = {}  # layer -> (additive router mask, 0/1 resident indicator); empty = unconstrained
_drop = [False]


def layer_index(model):
    """{id(SwitchGLU): layer} in named_modules order — the same order expert_locality records counts in."""
    mods = [m for _, m in model.named_modules() if isinstance(m, SwitchGLU)]
    return {id(m): i for i, m in enumerate(mods)}


def patch_router(model):
    """Qwen3MoeSparseMoeBlock.__call__ with the resident-set constraint (identical math when unmasked)."""
    idx = layer_index(model)
    blocks = [m for _, m in model.named_modules() if isinstance(m, qwen3_moe.Qwen3MoeSparseMoeBlock)]
    lid = {id(b): idx[id(b.switch_mlp)] for b in blocks}

    def masked_call(self, x):
        gates = self.gate(x)
        m = _mask.get(lid.get(id(self)))
        if m is not None and not _drop[0]:
            gates = gates + m[0]
        gates = mx.softmax(gates, axis=-1, precise=True)
        k = self.top_k
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if m is not None and _drop[0]:
            scores = scores * m[1][inds]  # non-resident picks contribute nothing
        if self.norm_topk_prob:
            scores /= mx.sum(scores, axis=-1, keepdims=True) + 1e-9
        y = self.switch_mlp(x, inds)
        return (y * scores[..., None]).sum(axis=-2)

    qwen3_moe.Qwen3MoeSparseMoeBlock.__call__ = masked_call
    return len(blocks)


def resident_sets(counts, n, adaptive, top_k=8):
    """Boolean [layer, expert] resident matrix: n per layer, or n × layers slots allocated greedily."""
    L, E = counts.shape
    res = np.zeros((L, E), bool)
    if not adaptive:
        np.put_along_axis(res, el.top_k(counts, n), True, axis=1)
        return res
    p = counts / counts.sum(axis=1, keepdims=True)
    np.put_along_axis(res, el.top_k(counts, top_k), True, axis=1)  # floor: routing needs ≥ top_k choices
    budget = n * L - res.sum()
    order = np.argsort(-np.where(res, -1, p), axis=None, kind="stable")[:budget]
    res.flat[order] = True
    return res


def set_mask(res=None, drop=False):
    _mask.clear()
    _drop[0] = drop
    if res is None:
        return
    for layer, keep in enumerate(res):
        _mask[layer] = (mx.array(np.where(keep, 0, -1e9).astype(np.float32)), mx.array(keep.astype(np.float32)))


def parse_config(cfg):
    if cfg in ("full", "dense"):
        return None
    src, budget, *rest = cfg.split("-")
    return {"src": src, "adaptive": budget[0] == "A", "n": int(budget[1:]), "drop": rest == ["drop"]}


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


def gsm8k(model, tok, rows, record=None, outputs=None):
    correct = 0
    for q, a in rows:
        out = generate(model, tok, chat(tok, q + MATH_SUFFIX), max_tokens=400)
        pred, ref = number(out), number(a.split("####")[-1])
        ok = pred is not None and abs(pred - ref) < 1e-6
        correct += ok
        if outputs is not None:
            outputs.append({"q": q, "out": out, "ok": bool(ok)})
        (record or el._calls.clear)()
    return correct / len(rows)


def code_nll(model, tok, rows, record=None):
    """Mean NLL per token of the canonical solution given the prompt (teacher-forced)."""
    nll, n = 0.0, 0
    for prompt, solution in rows:
        split = len(tok.encode(prompt))  # ponytail: assumes prompt tokens are a prefix of prompt+solution tokens
        ids = tok.encode(prompt + solution)
        logits = model(mx.array(ids)[None])[0].astype(mx.float32)
        logp = logits[:-1] - mx.logsumexp(logits[:-1], axis=-1, keepdims=True)
        tok_lp = mx.take_along_axis(logp, mx.array(ids[1:])[:, None], axis=-1)[:, 0]
        nll -= tok_lp[split - 1:].sum().item()
        n += len(ids) - split
        (record or el._calls.clear)()
    return nll / n


def load_data():
    g = snapshot_download("openai/gsm8k", repo_type="dataset", allow_patterns=["main/*"])
    train = pd.read_parquet(glob.glob(f"{g}/main/train-*.parquet")[0])
    test = pd.read_parquet(glob.glob(f"{g}/main/test-*.parquet")[0])
    h = snapshot_download("openai/openai_humaneval", repo_type="dataset")
    he = pd.read_parquet(glob.glob(f"{h}/**/*.parquet", recursive=True)[0])
    code = list(zip(he.prompt, he.canonical_solution))
    return list(zip(train.question, train.answer)), list(zip(test.question, test.answer)), code


def calibration(model, tok, name, L, E):
    """Per-task routing counts from the FULL model, cached in results/calib_<model>.json."""
    path = RESULTS / f"calib_{name}.json"
    if path.exists():
        c = json.loads(path.read_text())
        return {k: np.array(v) for k, v in c.items()}
    train, _, code = load_data()
    acc = {"math": np.zeros((L, E), np.int64), "code": np.zeros((L, E), np.int64)}
    def add(task):
        def f():
            acc[task] += routing_counts(L, E)
        return f
    _mask.clear()
    print("calibrating math…", flush=True)
    gsm8k(model, tok, train[:N_CALIB_MATH], record=add("math"))
    print("calibrating code…", flush=True)
    code_nll(model, tok, code[:N_CALIB_CODE], record=add("code"))
    exp1 = json.loads((RESULTS / f"expert_locality_{name}.json").read_text())
    norm = lambda c: c / c.sum(axis=1, keepdims=True)
    acc["global"] = norm(acc["math"]) + norm(acc["code"]) + norm(sum(np.array(r["decode"]) for r in exp1["runs"]))
    path.write_text(json.dumps({k: v.tolist() for k, v in acc.items()}))
    return acc


def self_check(model, tok, counts, L, E):
    """With a reroute mask on, routing never leaves the resident set; drop mode only zeroes weights."""
    res = resident_sets(counts, 32, adaptive=True)
    set_mask(res)
    model(mx.array(tok.encode("What is 17 * 23?"))[None])
    used = routing_counts(L, E) > 0
    assert not (used & ~res).any(), "mask leaked: routed to a non-resident expert"
    assert res.sum() == 32 * L and (res.sum(axis=1) >= 8).all()
    set_mask()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--configs", default=None, help="comma list; default: dense for dense models, "
                                                     "full,task-K48,task-K32,global-K32 for MoE")
    ap.add_argument("--adapter", default=None, help="LoRA adapter dir to load")
    ap.add_argument("--tag", default="", help="suffix for result keys, e.g. 'healed'")
    ap.add_argument("--n-math", type=int, default=100)
    ap.add_argument("--save-outputs", action="store_true")
    args = ap.parse_args()
    name = args.model.split("/")[-1]
    _, test, code = load_data()
    math_eval, code_eval = test[:args.n_math], code[N_CALIB_CODE:]

    model, tok = load(args.model, adapter_path=args.adapter)
    total = sum(v.nbytes for _, v in tree_flatten(model.parameters()))
    is_moe = patch_router(model) > 0
    configs = (args.configs or ("full,task-K48,task-K32,global-K32" if is_moe else "dense")).split(",")

    if is_moe:
        mods, expert_bytes = el.instrument(model)
        L, E = len(mods), mods[0].up_proj.num_experts
        counts = calibration(model, tok, name, L, E)
        self_check(model, tok, counts["math"], L, E)

    out_path = RESULTS / "routing_quality.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    for cfg in configs:
        spec, t0, outputs = parse_config(cfg), time.time(), []
        ram, scores = total, {}
        for bench, task in (("gsm8k", "math"), ("code", "code")):
            if spec:
                src = counts["global"] if spec["src"] == "global" else counts[task]
                res = resident_sets(src, spec["n"], spec["adaptive"])
                set_mask(res, spec["drop"])
                ram = total - expert_bytes + expert_bytes * res.sum() / (L * E)
            else:
                set_mask()
            if bench == "gsm8k":
                scores["acc"] = gsm8k(model, tok, math_eval, outputs=outputs)
            else:
                scores["nll"] = code_nll(model, tok, code_eval)
        set_mask()
        key = f"{name}|{cfg}" + (f"|{args.tag}" if args.tag else "")
        results[key] = {"ram_gb": ram / 1e9, "gsm8k_acc": scores["acc"], "n_gsm8k": len(math_eval),
                        "code_nll": scores["nll"], "code_ppl": math.exp(scores["nll"]), "n_code": len(code_eval),
                        "adapter": args.adapter, "minutes": (time.time() - t0) / 60}
        print(f"{key:40} RAM {ram / 1e9:5.1f} GB  GSM8K {scores['acc']:6.1%}  code NLL {scores['nll']:.3f}  "
              f"[{results[key]['minutes']:.1f} min]", flush=True)
        out_path.write_text(json.dumps(results, indent=1))
        if args.save_outputs:
            (RESULTS / f"outputs_{key.replace('|', '_')}.json").write_text(json.dumps(outputs, indent=1))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
