"""Self-distillation data for LoRA healing: the FULL 30B model's own GSM8K-train answers, kept only when correct.

Training the constrained model to imitate the unconstrained one (rather than GSM8K's terse reference
format) isolates "recover what masking broke" from "learn a new answer style". The dense 8B baseline is
fine-tuned on the same file, so neither side gets data the other didn't.

Usage: python experiments/distill.py [--n 800]   → data/gsm8k_distill/{train,valid}.jsonl
"""
import argparse
import json
from pathlib import Path

from mlx_lm import generate, load

import routing_quality as rq

OUT = Path(__file__).resolve().parent.parent / "data" / "gsm8k_distill"
N_VALID = 50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3-30B-A3B-4bit")
    ap.add_argument("--n", type=int, default=800)
    args = ap.parse_args()
    train, _, _ = rq.load_data()
    rows = train[rq.N_CALIB_MATH:rq.N_CALIB_MATH + args.n]  # disjoint from calibration and from the test set
    model, tok = load(args.model)
    kept = []
    for i, (q, a) in enumerate(rows):
        prompt = q + rq.MATH_SUFFIX
        out = generate(model, tok, rq.chat(tok, prompt), max_tokens=400)
        pred, ref = rq.number(out), rq.number(a.split("####")[-1])
        if pred is not None and abs(pred - ref) < 1e-6:
            kept.append({"prompt": prompt, "completion": out.strip()})
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(rows)} kept {len(kept)}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in kept[:-N_VALID]))
    (OUT / "valid.jsonl").write_text("".join(json.dumps(r) + "\n" for r in kept[-N_VALID:]))
    print(f"kept {len(kept)}/{len(rows)} → {OUT}")


if __name__ == "__main__":
    main()
