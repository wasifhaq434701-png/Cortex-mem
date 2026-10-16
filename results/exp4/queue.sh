# Exp 4 test sweep, priority order; every step resumes (results/exp4/*.json, results/exp3_gen/*.jsonl, chunks.nll).
set -u
cd "$(dirname "$0")/../../experiments"
H="cortex,div=0,edges=0"
r() { python -u user_memory.py run "$@" 2>&1 | grep -vE "Warn|Fetching"; }
j() { python -u user_memory.py judge 2>&1 | grep -vE "Warn|Fetching"; }
echo "== stage 1: core @2K"; r "$H" 2000; r rag 2000; r bm25 2000; j
echo "== stage 2: full context (LoCoMo) + oracle"; r full 0; r oracle 32000; j
echo "== stage 3: surprise signal"; python -u user_memory.py surprise 2>&1 | grep -vE "Warn|Fetching" | tail -3
echo "== stage 4: budget sweep"; r "$H" 1000 4000; r rag 1000 4000; j
echo "== stage 5: ablations @2K"; r cortex 2000; r "cortex,div=0" 2000; r "$H,keep=0.5" 2000; r "$H,keep=0.3" 2000; r session 2000; j
echo "== stage 6: extras"; r bm25 1000 4000; r "$H" 8000; j
echo "== queue done"
