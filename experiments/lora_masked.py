"""LoRA "healing" under the resident-expert constraint: plain mlx_lm.lora, with the router mask active
during training, so the adapter learns to work with only the resident experts.

Usage: python experiments/lora_masked.py task-A32-drop math -- --model mlx-community/Qwen3-30B-A3B-4bit \
           --train --data data/gsm8k_distill --adapter-path adapters/30b_task-A32-drop ...
"""
import json
import sys

import mlx_lm.lora as lora

import routing_quality as rq

cfg, task = sys.argv[1], sys.argv[2]
rest = sys.argv[sys.argv.index("--") + 1:]
name = rest[rest.index("--model") + 1].split("/")[-1]
spec = rq.parse_config(cfg)
_load = lora.load


def load_masked(*a, **k):
    model, tok = _load(*a, **k)
    rq.patch_router(model)
    counts = json.loads((rq.RESULTS / f"calib_{name}.json").read_text())[task]
    rq.set_mask(rq.resident_sets(rq.np.array(counts), spec["n"], spec["adaptive"]), spec["drop"])
    return model, tok


lora.load = load_masked
sys.argv = [sys.argv[0]] + rest
lora.main()
