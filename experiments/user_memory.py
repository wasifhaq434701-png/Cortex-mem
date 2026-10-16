"""Experiment 4 — per-user memory: tier-1 nodes + salience gate + Cortex-style selection into a fixed token budget.

Reader: Qwen3-4B-4bit. Benchmarks: LongMemEval-S cleaned (500 Qs, ~105K-token history each, MIT) and LoCoMo
(10 conversations, categories 1-4 = 1,540 Qs; category 5 adversarial dropped as in mem0 and later work; CC BY-NC).
Tuning happens ONLY on LongMemEval dev (every 5th question, 100); reported numbers are LME test (400) + all LoCoMo.

Tier-1 store (what would ship on the phone): SQLite rows (one per ≤1,000-char piece of a turn) + an fp16 memmap of
MiniLM embeddings + edges (same-session adjacency for expansion, cross-session similarity for recurrence).
Gate = nightly consolidation: rank a user's chunks by the mean percentile of four signals and keep the top `keep`
fraction — self (first-person statement by the memory's owner; stands in for an explicit "remember this", which the
benchmarks don't have), novelty (1 − max cosine to anything earlier), recurrence (other sessions holding a similar
chunk), surprise (mean token NLL under a small LM, `surprise` subcommand).

Budgets are memory-context tokens; KV MB = tokens × 144 KB (Qwen3-4B fp16 K+V). The whole LME history (~105K tokens,
~15 GB KV) cannot run on a 12 GB phone, which is the point of selecting.

Methods (name,key=value,… e.g. "cortex,keep=0.5,edges=0"):
  rag      top cosine chunks            session  whole sessions ranked by best chunk     bm25  top BM25 chunks
  cortex   gate → hybrid (cosine+BM25 reciprocal-rank fusion) → Cortex 1.5-cluster diversity rule → edge expansion
  full     whole history (LoCoMo only; prefix-cached)              oracle  evidence sessions only (upper bound)
Evidence recall is counted by turn IDs (LME has_answer / LoCoMo evidence), never by substring (Cortex trap #2).

Usage:
  python experiments/user_memory.py ingest | embed | check   (embed resumes an interrupted ingest)
  python experiments/user_memory.py surprise [model]           (resumable; writes chunks.nll)
  python experiments/user_memory.py recall [methods…]          (no LLM: gate recall curve + retrieval recall, LME dev)
  python experiments/user_memory.py run <method> <budget…> [--set lme-dev|lme-test|test|all]
  python experiments/user_memory.py judge                      (Qwen3-30B-A3B, LongMemEval's official prompts)
  python experiments/user_memory.py label <config>             (hand-label 100 answers, blind to the judge)
  python experiments/user_memory.py latency retrieval|ttft   (ttft needs the GPU to itself)
  python experiments/user_memory.py report
"""
import json
import re
import sqlite3
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

import routing_quality as rq
import task_adapters as ta

MAIN = "cortex,div=0,edges=0"  # frozen on LME dev 2026-10-14: diversity and edges did not help
MODEL, JUDGE, EMB = ta.MODEL, "mlx-community/Qwen3-30B-A3B-4bit", "sentence-transformers/all-MiniLM-L6-v2"
ROOT = Path(__file__).resolve().parent.parent
D, OUT = ROOT / "data" / "exp4", ROOT / "results" / "exp4"
DB, EMBF, QS, QEMB = D / "store.sqlite", D / "emb.f16", D / "questions.json", D / "qemb.npy"
LOCOMO = D / "raw" / "locomo10.json"
CHUNK_CHARS, DIM, SIM_EDGE, HEADER_TOK, MAX_NEW = 1000, 384, 0.6, 20, 200
KV_BYTES_PER_TOK = 36 * 2 * 8 * 128 * 2  # layers × (K,V) × kv heads × head dim × fp16
LOCOMO_CATS = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}  # mem0's reading of the ids
FIRST = re.compile(r"\b(i|i'm|i've|i'd|i'll|my|me|mine|myself|we|our|us)\b", re.I)
STOP = set("the a an and or of to in on for is are was were be been it this that with as at by from i you me my your "
           "we our he she they them his her their what when where which who how do did does have has had not no".split())
PROMPT = ("Here are excerpts from past conversations, oldest first; each session is headed by its date.\n{ctx}\n\n"
          "Current date: {date}\nQuestion: {question}\n"
          "Answer using the excerpts. Think briefly if needed, then give the final answer on the last line as "
          "'Answer: <answer>'. If the question asks for advice or a recommendation, give it, tailored to what the excerpts "
          "reveal about the user. If it asks for a fact the excerpts do not contain, say that you don't know.")


# ---------- ingest: benchmarks → one SQLite store + fp16 embedding memmap ----------

def lme_file(name="longmemeval_s_cleaned.json"):
    return hf_hub_download("xiaowu0162/longmemeval-cleaned", name, repo_type="dataset")


def pieces(text, n=CHUNK_CHARS):
    out = []
    while len(text) > n:
        cut = text.rfind(" ", 0, n)
        cut = cut if cut > n // 2 else n
        out.append(text[:cut])
        text = text[cut:].lstrip()
    return out + [text]


def load_benchmarks():
    """sessions: key → (date, [(role, owner, text)]); users: user → [session keys, chronological]; questions."""
    sessions, users, qs, clashes = {}, {}, [], 0

    def add(sid, date, turns, user):
        nonlocal clashes
        if sid in sessions and sessions[sid][1] != turns:  # same id, different text: keep both apart
            clashes, sid = clashes + 1, f"{sid}@{user}"
        sessions.setdefault(sid, (date, turns))
        return sid

    for i, x in enumerate(json.load(open(lme_file()))):
        u, keys, ev = x["question_id"], {}, []
        for sid, date, s in sorted(zip(x["haystack_session_ids"], x["haystack_dates"], x["haystack_sessions"]), key=lambda r: r[1]):
            keys[sid] = add(sid, date, [(t["role"], t["role"] == "user", t["content"]) for t in s], u)
            ev += [[keys[sid], j] for j, t in enumerate(s) if t.get("has_answer")]
        users[u] = list(keys.values())
        qs.append({"qid": u, "bench": "lme", "user": u, "type": x["question_type"], "question": x["question"],
                   "date": x["question_date"], "answer": str(x["answer"]), "abs": u.endswith("_abs"),
                   "ev_turns": ev, "ev_sessions": [keys[s] for s in x["answer_session_ids"] if s in keys],
                   "split": "dev" if i % 5 == 0 else "test"})

    for x in json.load(open(LOCOMO)):
        u, c, dia = x["sample_id"], x["conversation"], {}
        n = sorted(int(k.split("_")[1]) for k in c if re.fullmatch(r"session_\d+", k))
        for s in n:
            date = datetime.strptime(c[f"session_{s}_date_time"], "%I:%M %p on %d %B, %Y").strftime("%Y/%m/%d (%a) %H:%M")
            turns = [(t["speaker"], True, t["text"] + (f" [shares a photo: {t['blip_caption']}]" if t.get("blip_caption") else ""))
                     for t in c[f"session_{s}"]]
            key = add(f"{u}_s{s}", date, turns, u)
            dia.update({t["dia_id"]: (key, j) for j, t in enumerate(c[f"session_{s}"])})
        users[u] = [f"{u}_s{s}" for s in n]
        for j, q in enumerate(x["qa"]):
            if q["category"] == 5:
                continue
            ev = [dia[f"D{a}:{b}"] for a, b in re.findall(r"D(\d+):(\d+)", " ".join(q.get("evidence", []))) if f"D{a}:{b}" in dia]
            qs.append({"qid": f"{u}_q{j}", "bench": "locomo", "user": u, "type": LOCOMO_CATS[q["category"]],
                       "question": q["question"], "date": sessions[users[u][-1]][0], "answer": str(q["answer"]),
                       "abs": False, "ev_turns": [list(e) for e in ev], "ev_sessions": sorted({k for k, _ in ev}),
                       "split": "test"})
    print(f"{len(sessions)} sessions, {len(users)} users, {len(qs)} questions, {clashes} session-id clashes")
    return sessions, users, qs


def ingest():
    from transformers import AutoTokenizer
    from huggingface_hub import snapshot_download
    sessions, users, qs = load_benchmarks()
    D.mkdir(parents=True, exist_ok=True)
    DB.unlink(missing_ok=True)
    db = sqlite3.connect(DB)
    db.executescript("""
        CREATE TABLE sessions(sid TEXT PRIMARY KEY, date TEXT);
        CREATE TABLE haystack(user TEXT, pos INT, sid TEXT);
        CREATE TABLE chunks(id INTEGER PRIMARY KEY, sid TEXT, turn INT, part INT, role TEXT, owner INT, text TEXT,
                            ntok INT, nll REAL);
        CREATE INDEX chunks_sid ON chunks(sid); CREATE INDEX haystack_user ON haystack(user);""")
    db.executemany("INSERT INTO sessions VALUES (?,?)", [(k, d) for k, (d, _) in sessions.items()])
    db.executemany("INSERT INTO haystack VALUES (?,?,?)", [(u, p, s) for u, ss in users.items() for p, s in enumerate(ss)])
    rows = [(sid, t, p, role, int(owner), piece) for sid, (_, turns) in sessions.items()
            for t, (role, owner, text) in enumerate(turns) for p, piece in enumerate(pieces(text))]
    tok = AutoTokenizer.from_pretrained(snapshot_download(MODEL))
    texts = [f"{r[3]}: {r[5]}" for r in rows]
    ntok = [len(ids) for ids in tok(texts, add_special_tokens=False)["input_ids"]]
    db.executemany("INSERT INTO chunks(id, sid, turn, part, role, owner, text, ntok) VALUES (?,?,?,?,?,?,?,?)",
                   [(i, *r, n) for i, (r, n) in enumerate(zip(rows, ntok))])
    db.commit()
    print(f"{len(rows)} chunks, {sum(ntok) / 1e6:.1f}M tokens", flush=True)
    QS.write_text(json.dumps(qs))
    EMBF.unlink(missing_ok=True)
    embed()


def embed(block=10_000):
    """Resumable: blocks are written in order and a block is never half-written past `done`.
    Clears the MPS cache per block — PyTorch's allocator otherwise grew to 46 GB over 50K variable-length texts."""
    import torch
    from sentence_transformers import SentenceTransformer
    db = sqlite3.connect(DB)
    rows = db.execute("SELECT role, text FROM chunks ORDER BY id").fetchall()
    done_f = D / "emb.done"
    done = int(done_f.read_text()) if EMBF.exists() and done_f.exists() else 0
    E = np.memmap(EMBF, np.float16, "r+" if done else "w+", shape=(len(rows), DIM))
    emb = SentenceTransformer(EMB)
    for i in range(done, len(rows), block):
        E[i:i + block] = emb.encode([f"{r}: {t}" for r, t in rows[i:i + block]], batch_size=128, normalize_embeddings=True)
        E.flush()
        done_f.write_text(str(min(i + block, len(rows))))
        torch.mps.empty_cache()
        if (i // block) % 5 == 0:
            print(f"  embedded {min(i + block, len(rows))}/{len(rows)}", flush=True)
    qs = json.loads(QS.read_text())
    np.save(QEMB, emb.encode([q["question"] for q in qs], normalize_embeddings=True).astype(np.float16))
    print("embedding done", flush=True)


def store():
    db = sqlite3.connect(DB)
    n = db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    return db, np.memmap(EMBF, np.float16, "r", shape=(n, DIM))


def load_user(db, E, user):
    """One user's tier-1 store, chronological. Arrays indexed by local chunk position."""
    rows = db.execute("""SELECT c.id, c.sid, c.turn, c.role, c.owner, c.text, c.ntok, c.nll, s.date FROM haystack h
                         JOIN chunks c ON c.sid = h.sid JOIN sessions s ON s.sid = h.sid
                         WHERE h.user = ? ORDER BY h.pos, c.turn, c.part""", (user,)).fetchall()
    ids, sid, turn, role, owner, text, ntok, nll, date = map(list, zip(*rows))
    sess = np.unique(sid, return_inverse=True)[1]
    m = {"id": np.array(ids), "sid": sid, "turn": np.array(turn), "role": role, "owner": np.array(owner, bool),
         "text": text, "ntok": np.array(ntok), "nll": np.array([np.nan if v is None else v for v in nll]),
         "date": date, "sess": sess, "E": np.asarray(E[ids], np.float32)}
    m["key"] = {(s, t) for s, t in zip(sid, turn)}
    return m


# ---------- gate ----------

def pct(x):
    return np.argsort(np.argsort(x, kind="stable"), kind="stable") / max(len(x) - 1, 1)


def signals(m):
    E, sess = m["E"], m["sess"]
    S = E @ E.T
    earlier = np.where(np.tril(np.ones_like(S, bool), -1), S, -1)
    novelty = 1 - np.maximum(earlier.max(1), 0)
    similar = (S >= SIM_EDGE) & (sess[:, None] != sess[None, :])
    onehot = np.eye(sess.max() + 1, dtype=np.float32)[sess]
    recurrence = ((similar.astype(np.float32) @ onehot) > 0).sum(1)
    self_ = m["owner"] * np.array([len(FIRST.findall(t)) for t in m["text"]])
    out = {"self": self_, "novelty": novelty, "recurrence": recurrence}
    if not np.isnan(m["nll"]).any():
        out["surprise"] = m["nll"]
    return out


def gate(m, keep, which="all"):
    """Boolean admit mask: top `keep` fraction by mean percentile rank of the chosen signals."""
    if keep >= 1 or which == "none":
        return np.ones(len(m["id"]), bool)
    sig = m.setdefault("_sig", signals(m))
    names = list(sig) if which == "all" else which.split("+")
    score = np.mean([pct(sig[s]) for s in names], 0)
    return score >= np.quantile(score, 1 - keep)


# ---------- selection ----------

def words(t):
    return [w for w in re.findall(r"\w+", t.lower()) if w not in STOP]


def bm25(m, q, k1=1.2, b=0.75):
    if "_bm" not in m:
        docs = [Counter(words(t)) for t in m["text"]]
        df = Counter(w for d in docs for w in d)
        m["_bm"] = (docs, df, np.array([sum(d.values()) for d in docs], float))
    docs, df, dl = m["_bm"]
    n, avg = len(docs), dl.mean() or 1
    s = np.zeros(n)
    for w in set(words(q)):
        if w in df:
            tf = np.array([d[w] for d in docs], float)
            s += np.log(1 + (n - df[w] + .5) / (df[w] + .5)) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avg))
    return s


def rrf(*scores, k=60):
    return sum(1 / (k + np.argsort(np.argsort(-s, kind="stable"), kind="stable")) for s in scores)


def fill(m, order, budget, expand=False):
    """Greedy: take chunks in `order` while they fit (chunk tokens + a date header per new session).
    expand: after each pick, also take its same-session neighbours (end of the previous turn, start of the next)."""
    picked, sessions, used = set(), set(), 0

    def take(i):
        nonlocal used
        cost = m["ntok"][i] + (HEADER_TOK if m["sess"][i] not in sessions else 0)
        if i in picked or used + cost > budget:
            return False
        picked.add(i)
        sessions.add(m["sess"][i])
        used += cost
        return True

    for i in order:
        if take(i) and expand:
            for d in (-1, 1):
                nb = [j for j in (i + d * np.arange(1, 4)) if 0 <= j < len(m["id"]) and m["sess"][j] == m["sess"][i]
                      and m["turn"][j] == m["turn"][i] + d]
                if nb:
                    take(nb[0])
        if used >= budget - 8:
            break
    return sorted(picked), used


def select(m, q, qe, budget, method, opt):
    cos = m["E"] @ qe
    if method == "full":
        return list(range(len(cos))), int(m["ntok"].sum())
    if method == "oracle":
        ev = set(q["ev_sessions"])
        return fill(m, [i for i, s in enumerate(m["sid"]) if s in ev], budget)
    if method == "rag":
        return fill(m, np.argsort(-cos), budget)
    if method == "bm25":
        return fill(m, np.argsort(-bm25(m, q["question"])), budget)
    if method == "session":
        best = np.full(m["sess"].max() + 1, -2.0)
        np.maximum.at(best, m["sess"], cos)
        return fill(m, sorted(range(len(cos)), key=lambda i: (-best[m["sess"][i]], i)), budget)
    assert method == "cortex", method
    admit = gate(m, float(opt.get("keep", 1)), opt.get("gate", "all"))
    score = rrf(cos, bm25(m, q["question"])) if int(opt.get("bm25", 1)) else cos
    order = [i for i in np.argsort(-score) if admit[i]]
    if int(opt.get("div", 1)):
        order = diversify(m, order, admit)
    return fill(m, order, budget, expand=bool(int(opt.get("edges", 1))))


def kmeans(X, k, iters=30, seed=42):
    """k-means++ init + Lloyd. (sklearn's KMeans next to MLX segfaults at interpreter exit, losing buffered logs.)"""
    rng = np.random.default_rng(seed)
    C = X[[rng.integers(len(X))]]
    for _ in range(1, k):
        d = ((X[:, None] - C[None]) ** 2).sum(-1).min(1)
        C = np.vstack([C, X[rng.choice(len(X), p=d / d.sum()) if d.sum() > 0 else rng.integers(len(X))]])
    for _ in range(iters):
        lab = ((X[:, None] - C[None]) ** 2).sum(-1).argmin(1)
        C = np.array([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
    return lab


def diversify(m, order, admit, k=6, top=3):
    """Cortex rule: if flat top-3 sits inside one k-means cluster (< 1.5 clusters), lead with the best chunk of each
    cluster (clusters best-first), then continue in rank order; otherwise keep the flat order."""
    if "_km" not in m:
        idx = np.flatnonzero(admit)
        lab = np.full(len(admit), -1)
        lab[idx] = kmeans(m["E"][idx], min(k, len(idx)))
        m["_km"] = lab
    lab = m["_km"]
    if len({lab[i] for i in order[:top]}) >= 1.5:
        return order
    lead, seen = [], set()
    for i in order:
        if lab[i] not in seen:
            seen.add(lab[i])
            lead.append(i)
    return lead + [i for i in order if i not in set(lead)]


def render(m, idx):
    out, last = [], None
    for i in idx:
        if m["sess"][i] != last:
            out.append(f"\n### Session ({m['date'][i]})")
            last = m["sess"][i]
        out.append(f"{m['role'][i]}: {m['text'][i]}")
    return "\n".join(out)


def ev_recall(m, q, idx):
    ev = {tuple(e) for e in q["ev_turns"]}
    return len(ev & {(m["sid"][i], m["turn"][i]) for i in idx}) / len(ev) if ev else None


def parse(cfg):
    name, *kv = cfg.split(",")
    return name, dict(x.split("=") for x in kv)


def questions(split):
    qs, qe = json.loads(QS.read_text()), np.load(QEMB).astype(np.float32)
    keep = {"lme-dev": lambda q: q["bench"] == "lme" and q["split"] == "dev", "test": lambda q: q["split"] == "test",
            "lme-test": lambda q: q["bench"] == "lme" and q["split"] == "test",  # ablations: LME test only (time)
            "all": lambda q: True}[split]
    return [(q, e) for q, e in zip(qs, qe) if keep(q)]


def contexts(cfg, budget, split):
    """[(q, prompt text, memory tokens, evidence recall)] — users loaded once each."""
    name, opt = parse(cfg)
    db, E = store()
    out, m = [], None
    for q, qe in sorted(questions(split), key=lambda r: r[0]["user"]):
        if m is None or m["_user"] != q["user"]:
            m = load_user(db, E, q["user"])
            m["_user"] = q["user"]
        idx, used = select(m, q, qe, budget, name, opt)
        out.append((q, PROMPT.format(ctx=render(m, idx), date=q["date"], question=q["question"]), int(used),
                    ev_recall(m, q, idx)))
    return out


# ---------- runs ----------

def gen_prefixed(model, tok, items):
    """Full-context LoCoMo: prefill each conversation once, then answer its questions from a trimmed copy of the KV."""
    import mlx.core as mx
    from mlx_lm.generate import generate_step
    from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache
    outs, cache, prefix = [], None, None
    for _, text, _, _ in items:
        full = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                       enable_thinking=False, tokenize=False)
        cut = full.index("\n\nCurrent date:")
        if full[:cut] != prefix:
            prefix, cache = full[:cut], make_prompt_cache(model)
            ptoks = mx.array(tok.encode(prefix, add_special_tokens=False))[None]
            for i in range(0, ptoks.shape[1], 2048):
                model(ptoks[:, i:i + 2048], cache=cache)
                mx.eval([c.state for c in cache])
            base = cache[0].offset
        suffix = tok.encode(full[cut:], add_special_tokens=False)
        new = []
        for t, _ in generate_step(mx.array(suffix), model, max_tokens=MAX_NEW, prompt_cache=cache):
            t = t.item() if hasattr(t, "item") else t
            if t in tok.eos_token_ids:
                break
            new.append(t)
        outs.append(tok.decode(new))
        trim_prompt_cache(cache, cache[0].offset - base)
        if len(outs) % 100 == 0:
            print(f"  full {len(outs)}/{len(items)}", flush=True)
    return outs


def run(cfg, budgets, split):
    from mlx_lm import load
    model, tok = load(MODEL)
    OUT.mkdir(parents=True, exist_ok=True)
    for budget in budgets:
        key = f"{cfg}@{budget}" + ("" if split == "test" else f"~{split}")
        path = OUT / f"{key}.json"
        if path.exists():
            print(f"{key}: cached")
            continue
        items = contexts(cfg, budget, split)
        if cfg == "full":
            items = [it for it in items if it[0]["bench"] == "locomo"]  # LME histories (~105K tokens) exceed 40K context
            outs = gen_prefixed(model, tok, items)
        else:
            outs = ta.gen(model, tok, [t for _, t, _, _ in items], [MAX_NEW] * len(items), f"exp4_{key}")
        path.write_text(json.dumps({q["qid"]: {"out": o, "mem_tok": u, "recall": r} for (q, _, u, r), o in zip(items, outs)}))
        print(f"{key}: {len(items)} answers, mean memory tokens {np.mean([u for _, _, u, _ in items]):.0f}", flush=True)


# ---------- judge (prompts from LongMemEval src/evaluation/evaluate_qa.py, MIT) ----------

_ASK = ("I will give you a question, a correct answer, and a response from a model. Please answer yes if the response "
        "contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or "
        "contains all the intermediate steps to get the correct answer, you should also answer yes. If the response "
        "only contains a subset of the information required by the answer, answer no. ")
_END = "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
JUDGE_PROMPTS = {
    "default": _ASK + _END,
    "temporal-reasoning": _ASK + "In addition, do not penalize off-by-one errors for the number of days. If the "
    "question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting "
    "19 days when the answer is 18), the model's response is still correct. " + _END,
    "knowledge-update": "I will give you a question, a correct answer, and a response from a model. Please answer yes "
    "if the response contains the correct answer. Otherwise, answer no. If the response contains some previous "
    "information along with an updated answer, the response should be considered as correct as long as the updated "
    "answer is the required answer." + _END,
    "single-session-preference": "I will give you a question, a rubric for desired personalized response, and a "
    "response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. "
    "The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls "
    "and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\n"
    "Is the model response correct? Answer yes or no only.",
    "abstention": "I will give you an unanswerable question, an explanation, and a response from a model. Please "
    "answer yes if the model correctly identifies the question as unanswerable. The model could say that the "
    "information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}"
    "\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? "
    "Answer yes or no only.",
}


def judge_prompt(q, response):
    kind = "abstention" if q["abs"] else q["type"] if q["bench"] == "lme" and q["type"] in JUDGE_PROMPTS else "default"
    return JUDGE_PROMPTS[kind].format(q["question"], q["answer"], response)


def judge():
    from mlx_lm import load
    qs = {q["qid"]: q for q in json.loads(QS.read_text())}
    model = tok = None
    for path in sorted(OUT.glob("*@*.json")):
        res = json.loads(path.read_text())
        todo = [k for k, v in res.items() if "judge" not in v]
        if not todo:
            continue
        if model is None:
            model, tok = load(JUDGE)
        outs = ta.gen(model, tok, [judge_prompt(qs[k], res[k]["out"]) for k in todo], [8] * len(todo), f"exp4_judge_{path.stem}")
        for k, o in zip(todo, outs):
            res[k]["judge"] = int("yes" in o.lower())
        path.write_text(json.dumps(res))
        print(f"judged {path.stem}: {np.mean([res[k]['judge'] for k in todo]):.1%}", flush=True)


def check_items(cfg):
    """The fixed judge-check set: 100 answers (50 LME test stratified by type, 50 LoCoMo by category).
    Raters: `human` (the author, via `label`) and `claude` (second annotator); both label blind to the judge."""
    import random
    path = OUT / "judge_check.json"
    if path.exists():
        return json.loads(path.read_text())
    qs = {q["qid"]: q for q in json.loads(QS.read_text())}
    res = json.loads((OUT / f"{cfg}.json").read_text())
    rng, picked = random.Random(0), []
    for bench in ("lme", "locomo"):
        pool = sorted(k for k in res if qs[k]["bench"] == bench)
        types = sorted({qs[k]["type"] for k in pool})
        for j in range(50):
            t = types[j % len(types)]
            picked.append(rng.choice([k for k in pool if qs[k]["type"] == t and k not in picked]))
    items = [{"qid": k, "config": cfg, "human": None, "claude": None} for k in picked]
    path.write_text(json.dumps(items, indent=1))
    return items


def label(cfg):
    qs = {q["qid"]: q for q in json.loads(QS.read_text())}
    res = json.loads((OUT / f"{cfg}.json").read_text())
    path, items = OUT / "judge_check.json", check_items(cfg)
    for it in items:
        if it["human"] is not None:
            continue
        q = qs[it["qid"]]
        print(f"\n{'=' * 80}\n[{q['bench']} / {q['type']}{' / UNANSWERABLE' if q['abs'] else ''}]  {q['question']}\n"
              f"GOLD: {q['answer']}\n--- response ---\n{res[it['qid']]['out']}\n")
        a = input("correct? [y/n, q=quit] ").strip().lower()
        if a == "q":
            break
        it["human"] = int(a == "y")
        path.write_text(json.dumps(items, indent=1))
    print(f"{sum(i['human'] is not None for i in items)}/100 labelled → {path}")


# ---------- surprise (gate signal) ----------

def surprise(model_name="mlx-community/Qwen3-0.6B-4bit", tok_budget=4096):
    """Mean token NLL per chunk, context-free. Token budget bounds the (batch × len × 152K-vocab) logits."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load
    model, tok = load(model_name)
    db = sqlite3.connect(DB)
    rows = db.execute("SELECT id, role, text FROM chunks WHERE nll IS NULL ORDER BY ntok").fetchall()
    print(f"{len(rows)} chunks to score", flush=True)
    i = steps = 0
    while i < len(rows):
        ids = [tok.encode(f"{r[1]}: {r[2]}") for r in rows[i:i + 512]]
        n = max(2, min(len(ids), tok_budget // max(len(ids[-1]), 1)))
        ids = ids[:n]
        L = max(len(x) for x in ids)
        x = mx.array([t + [0] * (L - len(t)) for t in ids])
        mask = mx.array([[1] * (len(t) - 1) + [0] * (L - len(t)) for t in ids], mx.float32)
        ce = nn.losses.cross_entropy(model(x[:, :-1]), x[:, 1:]) * mask
        nll = (ce.sum(1) / mx.maximum(mask.sum(1), 1)).tolist()
        db.executemany("UPDATE chunks SET nll = ? WHERE id = ?", [(v, r[0]) for v, r in zip(nll, rows[i:i + n])])
        db.commit()
        i, steps = i + n, steps + 1
        if steps % 500 == 0:
            print(f"  {i}/{len(rows)}", flush=True)


# ---------- offline recall (no LLM) ----------

def recall(methods):
    db, E = store()
    qs = questions("lme-dev")
    keeps = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75]
    curves = {}
    for q, _ in qs:
        m = load_user(db, E, q["user"])
        if not q["ev_turns"]:
            continue
        ev = {tuple(e) for e in q["ev_turns"]}
        names = list(signals(m)) + ["all"]
        for s in names:
            for k in keeps:
                a = gate(m, k, s)
                curves.setdefault(s, {}).setdefault(k, []).append(
                    len(ev & {(m["sid"][i], m["turn"][i]) for i in np.flatnonzero(a)}) / len(ev))
    print("gate: evidence-turn recall vs kept fraction (LME dev)")
    print("signal      " + "".join(f"{k:>7}" for k in keeps))
    for s, c in curves.items():
        print(f"{s:12}" + "".join(f"{np.mean(c[k]):7.2f}" for k in keeps))
    for cfg in methods:
        for b in (1000, 2000, 4000, 8000):
            r = [x[3] for x in contexts(cfg, b, "lme-dev") if x[3] is not None]
            print(f"retrieval {cfg:28} @{b:5}: evidence recall {np.mean(r):.3f}")


# ---------- latency (M4 Max; the iPhone numbers come later, on the device) ----------

LAT = ROOT / "results" / "exp4_latency.json"


def ttft_prompts(lme, loc):
    """5 LME test prompts per hybrid budget + 5 whole-history LoCoMo prompts (one per conversation, ~20K tokens)."""
    want = {q["qid"] for q in lme[:5]}
    prompts = {f"hybrid@{b}": [t for q, t, _, _ in contexts(MAIN, b, "lme-test") if q["qid"] in want] for b in (1000, 2000, 4000, 8000)}
    one_per_conv = {q["user"]: q["qid"] for q in loc}
    want = set(list(one_per_conv.values())[:5])
    prompts["full@locomo"] = [t for q, t, _, _ in contexts("full", 0, "test") if q["qid"] in want]
    return prompts


def latency(part):
    """retrieval (CPU): per-query store load, query embedding, selection — the BM25 index build is timed apart
    (it is an ingest-time cost on the phone). ttft (GPU, nothing else may run on it): time to first token and decode
    speed for real prompts at each budget, fresh KV cache every time (no prefix-cache reuse: Cortex trap #1),
    plus peak memory. Results merge into results/exp4_latency.json."""
    import time
    out = json.loads(LAT.read_text()) if LAT.exists() else {}
    ms = lambda xs: {"median_ms": float(np.median(xs) * 1e3), "p90_ms": float(np.percentile(xs, 90) * 1e3), "n": len(xs)}
    rng = np.random.default_rng(0)
    qs = [q for q, _ in questions("test")]
    lme = [qs[i] for i in rng.choice([i for i, q in enumerate(qs) if q["bench"] == "lme"], 100, replace=False)]
    loc = [qs[i] for i in rng.choice([i for i, q in enumerate(qs) if q["bench"] == "locomo"], 100, replace=False)]
    if part == "retrieval":
        import torch
        from sentence_transformers import SentenceTransformer
        torch.set_num_threads(1)  # one core, closer to a phone's budget than the M4 Max's 14
        emb = SentenceTransformer(EMB, device="cpu")
        emb.encode(["warm-up"])
        db, E = store()
        res = {}
        for bench, sample in (("lme", lme), ("locomo", loc)):
            t = {"load": [], "embed": [], "index": [], "select": []}
            for q in sample:
                t0 = time.perf_counter()
                m = load_user(db, E, q["user"])
                t1 = time.perf_counter()
                qe = emb.encode([q["question"]], normalize_embeddings=True)[0]
                t2 = time.perf_counter()
                bm25(m, "warm")  # builds the per-user BM25 index
                t3 = time.perf_counter()
                for b in (1000, 2000, 4000, 8000):
                    s0 = time.perf_counter()
                    select(m, q, qe, b, *parse(MAIN))
                    t["select"].append(time.perf_counter() - s0)
                t["load"].append(t1 - t0); t["embed"].append(t2 - t1); t["index"].append(t3 - t2)
            res[bench] = {k: ms(v) for k, v in t.items()} | {"chunks_per_user": float(np.mean([len(load_user(db, E, q["user"])["id"]) for q in sample[:10]]))}
        out["retrieval"] = res | {"note": "1 CPU thread, M4 Max; select = hybrid at 1K/2K/4K/8K, index build excluded"}
    else:
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.generate import generate_step
        from mlx_lm.models.cache import make_prompt_cache
        model, tok = load(MODEL)
        prompts = ttft_prompts(lme, loc)
        res = {}
        for name, texts in prompts.items():
            rows = []
            for k, text in enumerate([texts[0]] + texts):  # first run of each size is a discarded warm-up
                ids = mx.array(rq.chat(tok, text))
                mx.clear_cache(); mx.reset_peak_memory()
                cache = make_prompt_cache(model)
                t0 = time.perf_counter()
                gen = generate_step(ids, model, max_tokens=65, prompt_cache=cache)
                next(gen)
                t1 = time.perf_counter()
                n = sum(1 for _ in gen)
                t2 = time.perf_counter()
                if k:
                    rows.append((len(ids), t1 - t0, n / (t2 - t1), mx.get_peak_memory() / 2 ** 30))
            L, ttft, dec, peak = map(np.array, zip(*rows))
            res[name] = {"prompt_tokens": float(L.mean()), "ttft_s": float(np.median(ttft)), "ttft_s_max": float(ttft.max()),
                         "prefill_tok_s": float(np.median(L / ttft)), "decode_tok_s": float(np.median(dec)),
                         "peak_gb": float(peak.max()), "n": len(rows)}
            print(name, res[name], flush=True)
        out["ttft"] = res | {"note": "M4 Max GPU, Qwen3-4B-4bit, fresh cache per prompt, median of 5 real prompts"}
    LAT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out[part if part == "retrieval" else "ttft"], indent=1))


# ---------- report ----------

def report():
    qs = {q["qid"]: q for q in json.loads(QS.read_text())}
    rows, per_item = {}, {}
    for path in sorted(OUT.glob("*@*.json")):
        res = json.loads(path.read_text())
        if not all("judge" in v for v in res.values()):
            continue
        per_item[path.stem] = {k: v["judge"] for k, v in res.items()}
        r = {}
        for bench in ("lme", "locomo"):
            ks = [k for k in res if qs[k]["bench"] == bench]
            if not ks:
                continue
            r[bench] = float(np.mean([res[k]["judge"] for k in ks]))
            r[f"{bench}_types"] = {t: float(np.mean([res[k]["judge"] for k in ks if qs[k]["type"] == t]))
                                   for t in sorted({qs[k]["type"] for k in ks})}
            if bench == "locomo":
                r["locomo_f1"] = float(np.mean([ta.em_f1(ta.final_answer(res[k]["out"]), [qs[k]["answer"]])[1] for k in ks]))
            r[f"{bench}_mem_tok"] = float(np.mean([res[k]["mem_tok"] for k in ks]))
            r[f"{bench}_kv_mb"] = r[f"{bench}_mem_tok"] * KV_BYTES_PER_TOK / 2 ** 20
            rec = [res[k]["recall"] for k in ks if res[k]["recall"] is not None]
            r[f"{bench}_recall"] = float(np.mean(rec)) if rec else None
        rows[path.stem] = r
    agree = {}
    if (OUT / "judge_check.json").exists():
        items = json.loads((OUT / "judge_check.json").read_text())
        res = {c: json.loads((OUT / f"{c}.json").read_text()) for c in {i["config"] for i in items}}
        for i in items:
            i["judge"] = res[i["config"]][i["qid"]].get("judge")
        for a, b in [("human", "judge"), ("claude", "judge"), ("human", "claude")]:
            pairs = [(i[a], i[b]) for i in items if i.get(a) is not None and i.get(b) is not None]
            if pairs:
                x, y = np.array(pairs, float).T
                po, pe = (x == y).mean(), x.mean() * y.mean() + (1 - x.mean()) * (1 - y.mean())
                agree[f"{a} vs {b}"] = {"n": len(pairs), "agreement": float(po),
                                        "kappa": float((po - pe) / (1 - pe)) if pe < 1 else None,
                                        f"{b}_minus_{a}": float(y.mean() - x.mean())}
    rng = np.random.default_rng(0)
    cis = {}
    for cfg, ref in [(c, r) for c in per_item for r in ("rag", MAIN)]:  # paired bootstrap vs RAG and vs the main config
        budget = cfg.split("@")[1]  # same budget; a "~lme-test" ablation pairs with the full-test run on shared items
        base = next((b for b in (f"{ref}@{budget}", f"{ref}@{budget.split('~')[0]}") if b in per_item and b != cfg), None)
        if base is None or cfg.startswith(f"{ref}@"):
            continue
        for bench in ("lme", "locomo"):
            ks = [k for k in per_item[cfg] if qs[k]["bench"] == bench and k in per_item[base]]
            if ks:
                d = np.array([per_item[cfg][k] - per_item[base][k] for k in ks], float)
                bs = d[rng.integers(0, len(d), (10000, len(d)))].mean(1)
                cis[f"{cfg} - {base} [{bench}]"] = [float(d.mean()), *map(float, np.percentile(bs, [2.5, 97.5]))]
    (ROOT / "results" / "exp4.json").write_text(json.dumps({"table": rows, "judge_check": agree, "paired_ci": cis}, indent=1))
    print("| config | LME acc | LoCoMo J | LoCoMo F1 | memory tok (LME/LoCoMo) | KV MB (LME) | evidence recall (LME/LoCoMo) |")
    print("|---|---|---|---|---|---|---|")
    f = lambda v, p=True: "–" if v is None else (f"{v:.1%}" if p else f"{v:.0f}")
    for c, r in rows.items():
        print(f"| {c} | {f(r.get('lme'))} | {f(r.get('locomo'))} | {f(r.get('locomo_f1'))} | "
              f"{f(r.get('lme_mem_tok'), 0)}/{f(r.get('locomo_mem_tok'), 0)} | {f(r.get('lme_kv_mb'), 0)} | "
              f"{f(r.get('lme_recall'))}/{f(r.get('locomo_recall'))} |")
    for k, v in agree.items():
        print(f"{k}: " + ", ".join(f"{m} {x:.2f}" if isinstance(x, float) else f"{m} {x}" for m, x in v.items()))
    for k, (mu, lo, hi) in cis.items():
        print(f"{k:45} {mu * 100:+.1f} [{lo * 100:+.1f},{hi * 100:+.1f}]{'*' if lo > 0 or hi < 0 else ''}")


def _check():
    ps = pieces("ab " * 900)
    assert all(len(x) <= CHUNK_CHARS for x in ps) and " ".join(ps).split() == ["ab"] * 900
    assert pieces("x" * 2500) == ["x" * 1000, "x" * 1000, "x" * 500] and pieces("") == [""]
    E = np.eye(4, DIM, dtype=np.float32)
    m = {"id": np.arange(4), "sid": ["s0", "s0", "s1", "s1"], "turn": np.array([0, 1, 0, 1]), "role": ["user"] * 4,
         "owner": np.array([1, 0, 1, 0], bool), "text": ["I love jazz", "ok", "my cat Tom", "fine"],
         "ntok": np.array([10, 10, 10, 10]), "nll": np.full(4, np.nan), "date": ["d0", "d0", "d1", "d1"],
         "sess": np.array([0, 0, 1, 1]), "E": E}
    assert fill(m, [2, 0, 1, 3], 2 * (10 + HEADER_TOK))[0] == [0, 2]  # two chunks + two headers fit, third doesn't
    assert fill(m, [2], 100, expand=True)[0] == [2, 3]  # neighbour turn pulled in
    assert gate(m, 0.5, "self").tolist() == [True, False, True, False]
    assert bm25(m, "jazz").argmax() == 0
    assert ev_recall(m, {"ev_turns": [["s1", 0], ["s0", 1]]}, [2]) == 0.5
    assert judge_prompt({"abs": True, "bench": "lme", "type": "multi-session", "question": "q", "answer": "a"}, "r").startswith("I will give you an unanswerable")
    X = np.vstack([np.zeros((5, 2)), np.ones((5, 2)) * 10])
    lab = kmeans(X, 2)
    assert len(set(lab[:5])) == len(set(lab[5:])) == 1 and lab[0] != lab[5]
    print("checks pass")


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "run":
        split = args[args.index("--set") + 1] if "--set" in args else "test"
        args = [a for a in args if a not in ("--set", split)]
        run(args[0], [int(b) for b in args[1:]] or [0], split)
    else:
        {"ingest": ingest, "embed": embed, "check": _check, "surprise": lambda: surprise(*args), "recall": lambda: recall(args),
         "judge": judge, "latency": lambda: latency(args[0]), "label": lambda: label(args[0]), "report": report}[cmd]()
