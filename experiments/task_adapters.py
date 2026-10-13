"""Experiment 3 — dense core + per-task LoRA modules picked once per request (vs one multi-task LoRA).

Core: Qwen3-4B-4bit. Tasks and data (train → eval, never overlapping):
  math   GSM8K train[50:2050] (reference solutions)         → GSM8K test[:N]           accuracy
  code   MBPP all splits (974 problems)                      → HumanEval (164)          pass@1, executed
  notes  synthetic notes (train pools) + DROP train, 50/50   → synthetic notes (held-out pools/phrasings) EM
                                                              → DROP validation (number/span answers) EM, F1
Adapters (same recipe, all layers): math / code / notes at rank R, ITERS steps each;
  multi-r{R}  one LoRA on the union, same per-request cost as a task adapter
  multi-r{3R} one LoRA on the union, same total storage as the three task adapters
  Both multi runs train 3 × ITERS steps, i.e. see as many examples as the three task adapters combined.

`eval` runs EVERY adapter on EVERY eval set (resumable, per-item scores), so routing needs no reloads:
  oracle  = each item scored under its own task's adapter
  routed  = each item scored under the adapter a Cortex-style router picks (MiniLM nearest task centroid)
  and the off-diagonal cells give the cross-task interference matrix.

Usage:
  python experiments/task_adapters.py data | distill   (reference completions | base model's own correct ones)
  python experiments/task_adapters.py train [names…]      (logs → results/exp3_train_<name>.log)
  python experiments/task_adapters.py eval  [configs…]
  python experiments/task_adapters.py report
"""
import glob
import hashlib
import json
import random
import re
import string
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download

import notes_synth
import routing_quality as rq

MODEL = "mlx-community/Qwen3-4B-4bit"
ROOT = Path(__file__).resolve().parent.parent
DATA, ADAPTERS, RESULTS = ROOT / "data" / "exp3", ROOT / "adapters" / "exp3", ROOT / "results"
ITEMS = RESULTS / "exp3_items.json"
TASKS = ["math", "code", "notes"]
R, ITERS, LR = 8, 300, 2e-5  # round 1 (reference data) used 600 iters at 1e-4
RUNS = {"math": R, "code": R, "notes": R, f"multi-r{R}": R, f"multi-r{3 * R}": 3 * R}
N_GSM8K, N_NOTES, N_DROP, N_VALID, MAX_SEQ = 250, 200, 300, 50, 1024
NOTES_SUFFIX = "\nAnswer concisely, then give the final line as 'Answer: <answer>'."
CODE_PREFIX = "Complete the following Python function. Return the whole function in a ```python block.\n\n"
EVAL_SETS = {"gsm8k": "math", "humaneval": "code", "notes": "notes", "drop": "notes"}  # eval set → gold task


# ---------- data ----------

def drop_rows(split):
    d = snapshot_download("ucinlp/drop", repo_type="dataset")
    df = pd.read_parquet(f"{d}/data/{split}-00000-of-00001.parquet")
    # ponytail: date-type answers (~2%) skipped; add DROP's date normalisation if they matter
    keep = df.answers_spans.map(lambda a: len(a["types"]) > 0 and a["types"][0] in ("number", "span"))
    return [(r.passage, r.question, [str(s) for s in r.answers_spans["spans"]]) for r in df[keep].itertuples()]


def mbpp_rows():
    m = snapshot_download("google-research-datasets/mbpp", repo_type="dataset", allow_patterns=["full/*"])
    df = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(f"{m}/full/*.parquet")))
    return [(t, list(tests), c.replace("\r\n", "\n").expandtabs(4), s or "")
            for t, tests, c, s in zip(df.text, df.test_list, df.code, df.test_setup_code)]


def humaneval():
    h = snapshot_download("openai/openai_humaneval", repo_type="dataset")
    return pd.read_parquet(glob.glob(f"{h}/**/*.parquet", recursive=True)[0])


notes_prompt = lambda r: f"My notes:\n{r['context']}\n\nQuestion: {r['question']}"
drop_prompt = lambda p, q: f"Passage:\n{p}\n\nQuestion: {q}"


def train_rows(rng):
    """{task: [(prompt, reference completion, max_tokens, is_correct(model_output))]}"""
    train, _, _ = rq.load_data()
    num_ok = lambda ref: lambda o: rq.number(o) is not None and abs(rq.number(o) - rq.number(ref)) < 1e-6
    em_ok = lambda golds: lambda o: em_f1(final_answer(o), golds)[0] == 1.0
    gsm = [(q + rq.MATH_SUFFIX, re.sub(r"<<.*?>>", "", a).split("####")[0].strip() + f"\nAnswer: {a.split('####')[-1].strip()}",
            400, num_ok(a.split("####")[-1])) for q, a in train[rq.N_CALIB_MATH:rq.N_CALIB_MATH + 2000 + N_VALID]]
    code = [(f"{t}\nYour code should pass these tests:\n" + "\n".join(tests), f"```python\n{c.strip()}\n```", 512,
             lambda o, s=setup, tests=tests: runs(s + "\n" + extract_code(o) + "\n\n" + "\n".join(tests)))
            for t, tests, c, setup in mbpp_rows()]
    synth = [(notes_prompt(r) + NOTES_SUFFIX, f"{r['rationale']}.\nAnswer: {r['answer']}", 200, em_ok([r["answer"]]))
             for r in notes_synth.dataset(1000 + N_VALID // 2, "train")]
    drop = [(drop_prompt(p, q) + NOTES_SUFFIX, f"Answer: {a[0]}", 200, em_ok(a))
            for p, q, a in rng.sample(drop_rows("train"), 1000 + N_VALID // 2)]
    return {"math": gsm, "code": code, "notes": synth + drop}


def build_data(distill=False):
    """Training rows. Reference completions (round 1) taught the 4B a WORSE style than its own and degraded it
    on every task, so `distill` (round 2) keeps the base model's own outputs, only where they are correct."""
    rng = random.Random(0)
    raw, sets, stats = train_rows(rng), {}, {}
    if distill:
        from mlx_lm import load
        model, tok = load(MODEL)
    for k, rows in raw.items():
        if distill:
            outs = gen(model, tok, [p for p, *_ in rows], [m for _, _, m, _ in rows], f"distill_{k}")
            fits = lambda p, o: len(tok.encode(p)) + len(tok.encode(o)) <= MAX_SEQ  # longer would be truncated in training
            sets[k] = [{"prompt": p, "completion": o.strip()} for (p, _, _, ok), o in zip(rows, outs) if ok(o) and fits(p, o)]
            stats[k] = {"kept": len(sets[k]), "of": len(rows)}
            print(f"distill {k}: kept {len(sets[k])}/{len(rows)}", flush=True)
        else:
            sets[k] = [{"prompt": p, "completion": c} for p, c, *_ in rows]
        sets[k] = rng.sample(sets[k], len(sets[k]))
    if distill:
        (RESULTS / "exp3_distill_stats.json").write_text(json.dumps(stats, indent=1))
    split = {k: (v[:-N_VALID], v[-N_VALID:]) for k, v in sets.items()}
    split["multi"] = tuple(sum((split[k][i] for k in TASKS), []) for i in (0, 1))
    for name, (tr, va) in split.items():
        (DATA / name).mkdir(parents=True, exist_ok=True)
        for fn, rows in (("train", rng.sample(tr, len(tr))), ("valid", va)):  # shuffled: multi interleaves tasks
            (DATA / name / f"{fn}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(f"{name:6} train {len(tr):5}  valid {len(va)}")


def eval_items():
    """[(set, user_text, max_tokens, gold)] — user_text is also what the router sees."""
    _, test, _ = rq.load_data()
    items = [("gsm8k", q + rq.MATH_SUFFIX, 400, a.split("####")[-1]) for q, a in test[:N_GSM8K]]
    items += [("humaneval", CODE_PREFIX + f"```python\n{r.prompt}```", 512, i) for i, r in enumerate(humaneval().itertuples())]
    items += [("notes", notes_prompt(r) + NOTES_SUFFIX, 200, r["answer"]) for r in notes_synth.dataset(N_NOTES, "test")]
    items += [("drop", drop_prompt(p, q) + NOTES_SUFFIX, 200, a) for p, q, a in random.Random(1).sample(drop_rows("validation"), N_DROP)]
    return items


# ---------- scoring ----------

def final_answer(out):
    return out.split("Answer:")[-1].strip().split("\n")[0].strip() if "Answer:" in out else out.strip().split("\n")[-1]


def normalize(s):
    s = "".join(c for c in s.lower() if c not in set(string.punctuation) - {"."}).rstrip(".")
    return " ".join(w for w in s.split() if w not in ("a", "an", "the"))


def em_f1(pred, golds):
    """SQuAD/DROP-style: max over gold answers; numbers compared numerically."""
    p = normalize(pred)
    best = (0.0, 0.0)
    for g in golds:
        g = normalize(g)
        pn, gn = rq.number(p), rq.number(g)
        if pn is not None and gn is not None and re.fullmatch(r"-?[\d,.]+", g):
            hit = float(abs(pn - gn) < 1e-6)
            best = max(best, (hit, hit))
            continue
        pt, gt = p.split(), g.split()
        common = sum(min(pt.count(w), gt.count(w)) for w in set(gt))
        f1 = 0.0 if not common else 2 * common / (len(pt) + len(gt))
        best = max(best, (float(p == g), f1))
    return best


def extract_code(out):
    m = re.search(r"```(?:python)?\n(.*?)```", out, re.S)
    return m.group(1) if m else out


def runs(program):
    """True if the program exits cleanly in a fresh, isolated interpreter."""
    with tempfile.TemporaryDirectory() as d:  # ponytail: no syscall sandbox — HumanEval-standard subprocess + timeout
        try:
            return subprocess.run([sys.executable, "-I", "-c", program], cwd=d, capture_output=True, timeout=10).returncode == 0
        except subprocess.TimeoutExpired:
            return False


def passes(code_out, row):
    """Run the model's function against HumanEval's tests."""
    code = extract_code(code_out)
    # the prompt's stub is valid Python (docstring body) and carries the imports; the model's def overrides it
    return runs(row.prompt + ("\n" if f"def {row.entry_point}" in code else "") + code + "\n\n" + row.test + f"\ncheck({row.entry_point})\n")


def gen(model, tok, texts, max_toks, name, B=32, budget=40_000):
    """Greedy chat generations in order, appended to results/exp3_gen/<name>-<hash of prompts>.jsonl after every
    chunk so a crash resumes instead of restarting.
    Chunks hold ≤ B prompts AND ≤ `budget` padded tokens: batch KV caches pad to the longest member, so one
    3,700-token MBPP prompt among 31 short ones blew past Metal's memory (the three OOMs)."""
    import mlx.core as mx
    from mlx_lm import batch_generate
    path = RESULTS / "exp3_gen" / f"{name}-{hashlib.md5(json.dumps([texts, max_toks]).encode()).hexdigest()[:8]}.jsonl"
    path.parent.mkdir(exist_ok=True)
    outs = [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
    prompts = [rq.chat(tok, t) for t in texts]
    i, n_chunks = len(outs), 0
    with open(path, "a") as f:
        while i < len(texts):
            j, longest = i, 0
            while j < len(texts) and j - i < B:
                longest = max(longest, len(prompts[j]) + max_toks[j])
                if j > i and longest * (j - i + 1) > budget:
                    break
                j += 1
            new = batch_generate(model, tok, prompts[i:j], max_tokens=max_toks[i:j]).texts
            f.write("".join(json.dumps(t) + "\n" for t in new))
            f.flush()
            outs, i, n_chunks = outs + new, j, n_chunks + 1
            mx.clear_cache()
            if n_chunks % 10 == 1:
                print(f"  gen {name} {i}/{len(texts)}  peak {mx.get_peak_memory() / 1e9:.1f} GB", flush=True)
    return outs


def score(item, out, he):
    s, _, _, gold = item
    if s == "gsm8k":
        pred, ref = rq.number(out), rq.number(gold)
        return float(pred is not None and abs(pred - ref) < 1e-6)
    if s == "humaneval":
        return float(passes(out, he.iloc[gold]))
    if s == "notes":
        return em_f1(final_answer(out), [gold])[0]
    return list(em_f1(final_answer(out), gold))  # drop: [em, f1]


# ---------- train / eval ----------

def train(names):
    import yaml
    for name in names or RUNS:
        rank, multi = RUNS[name], name.startswith("multi")
        cfg = {"model": MODEL, "train": True, "data": str(DATA / ("multi" if multi else name)),
               "adapter_path": str(ADAPTERS / name), "fine_tune_type": "lora", "num_layers": -1,
               "batch_size": 4, "iters": ITERS * (3 if multi else 1), "learning_rate": LR, "mask_prompt": True,
               "max_seq_length": MAX_SEQ, "steps_per_eval": 100, "val_batches": -1, "seed": 0, "grad_checkpoint": True,
               # constant alpha/r: mlx's scale multiplies the update directly; rank 24 at scale 20 diverged at step ~40
               "lora_parameters": {"rank": rank, "scale": 20.0 * R / rank, "dropout": 0.0}}
        cfg_path = RESULTS / f"exp3_train_{name}.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg))
        print(f"training {name} (rank {rank}, {cfg['iters']} iters)…", flush=True)
        with open(RESULTS / f"exp3_train_{name}.log", "w") as log:
            subprocess.run([sys.executable, "-m", "mlx_lm", "lora", "-c", str(cfg_path)], stdout=log, stderr=subprocess.STDOUT, check=True)


def evaluate(configs):
    from mlx_lm import load
    items, he = eval_items(), humaneval()
    done = json.loads(ITEMS.read_text()) if ITEMS.exists() else {}
    for cfg in configs or ["base", *RUNS]:
        if cfg in done:
            print(f"{cfg}: cached")
            continue
        model, tok = load(MODEL, adapter_path=None if cfg == "base" else str(ADAPTERS / cfg))
        outs = gen(model, tok, [text for _, text, _, _ in items], [m for _, _, m, _ in items], f"eval_{cfg}")
        scores = {s: [] for s in EVAL_SETS}
        for it, out in zip(items, outs):
            scores[it[0]].append(score(it, out, he))
        done[cfg] = scores
        ITEMS.write_text(json.dumps(done))
        (RESULTS / f"exp3_outputs_{cfg}.json").write_text(json.dumps([{"set": it[0], "out": o} for it, o in zip(items, outs)], indent=1))
        print(cfg, {s: round(float(np.mean(np.array(v).reshape(len(v), -1)[:, 0])), 3) for s, v in scores.items()}, flush=True)


# ---------- routing + report ----------

def route(items):
    """Cortex-style: embed the request, pick the task whose training-prompt centroid is nearest (cosine)."""
    from sentence_transformers import SentenceTransformer
    emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    rng = random.Random(0)
    cents = []
    for t in TASKS:
        rows = [json.loads(l)["prompt"] for l in (DATA / t / "train.jsonl").read_text().splitlines()]
        cents.append(emb.encode(rng.sample(rows, 300), normalize_embeddings=True).mean(0))
    cents = np.array(cents) / np.linalg.norm(cents, axis=1, keepdims=True)
    e = emb.encode([text for _, text, _, _ in items], normalize_embeddings=True)
    return [TASKS[i] for i in (e @ cents.T).argmax(1)]


def report():
    done, items = json.loads(ITEMS.read_text()), eval_items()
    picks = route(items)
    by_set = {s: [p for it, p in zip(items, picks) if it[0] == s] for s in EVAL_SETS}
    col = lambda v, k=0: float(np.mean(np.array(v, float).reshape(len(v), -1)[:, k]))

    def row(per_item):  # per_item: set → list of scores
        return {"gsm8k": col(per_item["gsm8k"]), "humaneval": col(per_item["humaneval"]), "notes": col(per_item["notes"]),
                "drop_em": col(per_item["drop"]), "drop_f1": col(per_item["drop"], 1)}

    items_by = dict(done)
    if all(t in done for t in TASKS):
        items_by["oracle"] = {s: done[t][s] for s, t in EVAL_SETS.items()}
        items_by["routed"] = {s: [done[p][s][i] for i, p in enumerate(by_set[s])] for s in EVAL_SETS}
    table = {c: row(v) for c, v in items_by.items()}
    acc = {s: float(np.mean([p == EVAL_SETS[s] for p in v])) for s, v in by_set.items()}

    rng = np.random.default_rng(0)
    def paired_ci(a, b, s):  # mean difference (EM/acc) with a 95% paired-bootstrap interval over items
        d = np.array(items_by[a][s], float).reshape(-1, 1 if s != "drop" else 2)[:, 0] - \
            np.array(items_by[b][s], float).reshape(-1, 1 if s != "drop" else 2)[:, 0]
        bs = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
        return [float(d.mean()), *map(float, np.percentile(bs, [2.5, 97.5]))]
    pairs = [(a, b) for a, b in [("oracle", "base"), ("routed", "base"), ("oracle", "multi-r8"), ("routed", "multi-r8"),
                                 ("oracle", "multi-r24"), ("routed", "multi-r24")] if a in items_by and b in items_by]
    cis = {f"{a} - {b}": {s: paired_ci(a, b, s) for s in EVAL_SETS} for a, b in pairs}
    out = {"table": table, "router_acc": acc, "paired_ci": cis, "n": {s: len(v) for s, v in by_set.items()},
           "adapter_mb": {n: sum(Path(f).stat().st_size for f in glob.glob(str(ADAPTERS / n / "adapters.safetensors"))) / 1e6
                          for n in RUNS}}
    (RESULTS / "exp3.json").write_text(json.dumps(out, indent=1))
    print("| config | GSM8K | HumanEval | notes EM | DROP EM | DROP F1 |\n|---|---|---|---|---|---|")
    for c, r in table.items():
        print(f"| {c} | " + " | ".join(f"{v:.1%}" for v in r.values()) + " |")
    print("router accuracy:", {s: f"{a:.1%}" for s, a in acc.items()}, "| adapter MB:", {k: round(v, 1) for k, v in out["adapter_mb"].items()})
    for k, v in cis.items():  # * = 95% interval excludes 0
        print(f"{k:22}" + "".join(f"  {s} {m * 100:+.1f} [{lo * 100:+.1f},{hi * 100:+.1f}]{'*' if lo > 0 or hi < 0 else ''}"
                                  for s, (m, lo, hi) in v.items()))


def _check():
    assert em_f1("The 3,000.", ["3000"]) == (1.0, 1.0) and em_f1("4", ["3"]) == (0.0, 0.0)
    assert em_f1("Chaz Schilens", ["JaMarcus Russell", "Chaz Schilens"])[0] == 1.0
    assert 0 < em_f1("Chaz", ["Chaz Schilens"])[1] < 1
    assert final_answer("blah\nAnswer: rent\n") == "rent"
    he = humaneval()
    for i in (0, 1, 2):  # canonical solutions pass, both as a bare body and as a full fenced function; junk fails
        r = he.iloc[i]
        assert passes(r.canonical_solution, r)
        assert passes(f"```python\n{r.prompt}{r.canonical_solution}```", r)
        assert not passes("```python\ndef nope(): pass\n```", r)
    for task, rows in train_rows(random.Random(0)).items():  # the checkers accept the references themselves
        ok = sum(check(ref) for _, ref, _, check in rows) / len(rows)
        assert ok > (0.97 if task == "code" else 0.999), f"{task}: only {ok:.1%} of references pass their checker"
        assert not any(check("Answer: 123456789\n```python\npass\n```") for *_, check in rows[:50]), task
    from mlx_lm.utils import load_tokenizer
    from huggingface_hub import snapshot_download as sd
    tok = load_tokenizer(Path(sd(MODEL)))
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
    train_ids, eval_ids = tok.apply_chat_template(msgs), rq.chat(tok, "hi")
    assert train_ids[:len(eval_ids)] == eval_ids, "training format must extend the eval prompt (empty think block)"
    print("task_adapters self-check OK")


if __name__ == "__main__":
    cmd, args = (sys.argv[1:] or ["check"])[0], sys.argv[2:]
    {"data": build_data, "distill": lambda: build_data(distill=True), "check": _check, "report": report}.get(cmd, lambda: None)()
    if cmd == "train":
        train(args)
    elif cmd == "eval":
        evaluate(args)
