"""Expert-locality test — the go/no-go experiment for context-predicted expert residency.

Question: if Cortex tells us the *task* up front and we preload only that task's top-K experts per
layer into RAM, what fraction of decode-time expert reads hit RAM instead of flash?

Compares, per resident budget K:
  oracle  — the request's own top-K (upper bound, needs knowing the future)
  task    — top-K from OTHER requests of the same task (what Cortex can predict)
  global  — top-K from all other requests, task-agnostic (plain static expert pruning)
  random  — K / num_experts
task >> global  → task signal matters, thesis holds.   task ≈ global → static pruning is enough.
task ≈ random   → no locality, fall back to dense model + Cortex.

Usage: python experiments/expert_locality.py [--model mlx-community/Qwen3-30B-A3B-4bit] [--max-tokens 128]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from mlx.utils import tree_flatten
from mlx_lm import generate, load
from mlx_lm.models.switch_layers import SwitchGLU

# ponytail: 5 hand-written prompts per task; swap for real datasets (HumanEval, GSM8K, LongMemEval…) before the paper
PROMPTS = {
    "code": [
        "Write a Python function that merges overlapping intervals in a list of [start, end] pairs.",
        "Explain what this Swift code does and fix the bug: `for i in 0...array.count { print(array[i]) }`",
        "Write a SQL query returning the 3 customers with the highest total order value in the last 30 days.",
        "Implement a thread-safe LRU cache in Python with get and put methods.",
        "Convert this JavaScript callback code to async/await: fs.readFile('a.txt', (err, data) => { if (err) throw err; console.log(data); });",
    ],
    "math": [
        "A train leaves at 9:40 and arrives at 13:15, covering 287 km. What is its average speed in km/h? Show the steps.",
        "Solve for x: 3(x - 4) + 2x = 5x - 12 + x. Explain each step.",
        "What is the probability of rolling at least one six in four rolls of a fair die?",
        "A rectangle's length is 3 more than twice its width and its area is 65. Find its dimensions.",
        "Compute the sum of all integers from 1 to 500 that are divisible by 3 or 5.",
    ],
    "personal_notes": [
        "My notes: 'Passport no. on file, expires 14 Feb 2027. Car insurance renews 2 Jan. Dentist said floss, recheck in 6 months (last visit 3 Aug).' When do I need to renew my passport and what else is coming up?",
        "My notes: 'Aisha birthday 21 Oct, likes pottery and Murakami. Omar birthday 3 Dec, into cycling.' Suggest gifts for whoever's birthday is next.",
        "My notes: 'Dr. Khan, 2 Sep: reduce ibuprofen to 200mg twice daily for 2 weeks, then stop. Follow-up if pain persists.' What did the doctor tell me to do?",
        "My notes: 'Laptop bought 12 Mar 2025 from Croma, 2-year warranty, invoice in Drive/receipts. Headphones bought 5 May 2026, 1-year warranty.' Which warranties are still valid today?",
        "My notes: 'Rent 18,000 due on 5th. Electricity ~2,400. Gym 1,500 on 1st. SIP 5,000 on 10th.' How much do I pay in total each month and what's due first?",
    ],
    "chat": [
        "I've been feeling unmotivated at work lately. Any practical advice?",
        "What's a good weekly routine to get better at running a 5k?",
        "Can you explain how vaccines train the immune system, in simple terms?",
        "I'm visiting Istanbul for three days. What should I prioritize?",
        "How do I politely decline a meeting invite from my manager?",
    ],
    "writing": [
        "Write a short poem about the first rain of the monsoon.",
        "Write the opening paragraph of a mystery novel set on a night train.",
        "Write a product description for a minimalist leather wallet.",
        "Write a two-sentence horror story about a smart home.",
        "Write a heartfelt thank-you note to a teacher who changed your life.",
    ],
}
BUDGETS = (16, 32, 48, 64)

_calls = []  # (layer_id, indices) recorded during the current request


def instrument(model):
    """Patch SwitchGLU (shared by Qwen3-MoE, Gemma 4, OLMoE, LFM2-MoE…) to record routed expert ids."""
    mods = [m for _, m in model.named_modules() if isinstance(m, SwitchGLU)]
    lid = {id(m): i for i, m in enumerate(mods)}
    orig = SwitchGLU.__call__

    def recording_call(self, x, indices):
        _calls.append((lid[id(self)], indices))
        return orig(self, x, indices)

    SwitchGLU.__call__ = recording_call
    expert_bytes = sum(v.nbytes for m in mods for _, v in tree_flatten(m.parameters()))
    return mods, expert_bytes


def counts_from_calls(n_layers, n_experts):
    """Split recorded routing into prefill (multi-token calls) and decode (1 token/call) counts [layer, expert]."""
    prefill = np.zeros((n_layers, n_experts), np.int64)
    decode = np.zeros((n_layers, n_experts), np.int64)
    n_tok = sum(1 for layer, ind in _calls if layer == 0 and ind.shape[-2] == 1)
    for layer, ind in _calls:
        ind = np.array(ind)
        target = decode if ind.shape[-2] == 1 else prefill
        np.add.at(target[layer], ind.ravel(), 1)
    _calls.clear()
    return prefill, decode, n_tok


def hit_rate(counts, resident):
    """Fraction of expert selections served by the resident set (resident: [layer, K] expert ids)."""
    return np.take_along_axis(counts, resident, axis=1).sum() / counts.sum()


def top_k(counts, k):
    return np.argsort(-counts, axis=1, kind="stable")[:, :k]


def experts_for(counts, frac=0.9):
    """Mean over layers of the fewest experts covering `frac` of that layer's selections."""
    s = -np.sort(-counts, axis=1)
    cum = np.cumsum(s, axis=1) / s.sum(axis=1, keepdims=True)
    return float(np.mean((cum < frac).sum(axis=1) + 1))


def _selftest():
    c = np.array([[3, 1, 0, 0], [0, 0, 2, 2]])
    assert hit_rate(c, top_k(c, 1)) == 5 / 8
    assert experts_for(c, 0.75) == 1.5  # layer0 needs 1, layer1 needs 2
    assert top_k(c, 2).tolist() == [[0, 1], [2, 3]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3-30B-A3B-4bit")
    ap.add_argument("--max-tokens", type=int, default=128)
    args = ap.parse_args()
    _selftest()

    model, tok = load(args.model)
    mods, expert_bytes = instrument(model)
    n_layers, n_experts = len(mods), mods[0].up_proj.num_experts
    total_bytes = sum(v.nbytes for _, v in tree_flatten(model.parameters()))
    per_slot = expert_bytes / n_experts  # bytes for one expert id across all layers

    runs = []
    for task, prompts in PROMPTS.items():
        for i, p in enumerate(prompts):
            msgs = [{"role": "user", "content": p}]
            # enable_thinking=False: Qwen3 otherwise spends the whole budget inside <think>
            prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False)
            generate(model, tok, prompt, max_tokens=args.max_tokens)  # greedy by default → deterministic
            pre, dec, n_tok = counts_from_calls(n_layers, n_experts)
            runs.append({"task": task, "i": i, "prefill": pre, "decode": dec, "n_tok": n_tok})
            print(f"{task}[{i}] decode tokens={n_tok}  prefill experts touched "
                  f"{(pre > 0).mean():.0%}  k90={experts_for(dec):.1f}/{n_experts}", flush=True)

    tasks = list(PROMPTS)
    print(f"\nmodel={args.model}  layers={n_layers}  experts={n_experts}  "
          f"model={total_bytes / 1e9:.1f} GB  experts={expert_bytes / 1e9:.1f} GB  "
          f"non-expert={(total_bytes - expert_bytes) / 1e9:.1f} GB")
    summary = {}
    for K in BUDGETS:
        print(f"\nK={K}/{n_experts} per layer → resident experts {K * per_slot / 1e9:.1f} GB "
              f"+ non-expert {(total_bytes - expert_bytes) / 1e9:.1f} GB")
        print(f"{'task':16} {'oracle':>7} {'task':>7} {'global':>7} {'random':>7}")
        rows = {}
        for t in tasks + ["ALL"]:
            sel = [r for r in runs if t in ("ALL", r["task"])]
            o, tk, g = [], [], []
            for r in sel:
                others_t = sum(x["decode"] for x in runs if x["task"] == r["task"] and x is not r)
                others_all = sum(x["decode"] for x in runs if x is not r)
                o.append(hit_rate(r["decode"], top_k(r["decode"], K)))
                tk.append(hit_rate(r["decode"], top_k(others_t, K)))
                g.append(hit_rate(r["decode"], top_k(others_all, K)))
            rows[t] = {"oracle": np.mean(o), "task": np.mean(tk), "global": np.mean(g), "random": K / n_experts}
            print(f"{t:16} " + " ".join(f"{v:7.1%}" for v in rows[t].values()))
        miss = 1 - rows["ALL"]["task"]
        sel_tok = runs[0]["decode"].sum() / runs[0]["n_tok"]  # expert selections per token, all layers
        print(f"task-predicted residency: {miss * sel_tok:.0f} expert loads/token from flash "
              f"≈ {miss * sel_tok * per_slot / n_layers / 1e6:.0f} MB/token")
        summary[K] = {t: {k: float(v) for k, v in row.items()} for t, row in rows.items()}

    out = Path(__file__).resolve().parent.parent / "results" / f"expert_locality_{args.model.split('/')[-1]}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "model": args.model, "max_tokens": args.max_tokens, "n_layers": n_layers, "n_experts": n_experts,
        "total_bytes": total_bytes, "expert_bytes": expert_bytes, "summary": summary,
        "runs": [{"task": r["task"], "i": r["i"], "prefill": r["prefill"].tolist(),
                  "decode": r["decode"].tolist()} for r in runs],
    }))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
