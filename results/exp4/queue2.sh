# Stages 5-6 on LongMemEval test only (400 Qs) — user decision 2026-10-15, cuts ~18 h. Stages 1-4 ran from queue.sh.
set -u
cd "$(dirname "$0")/../../experiments"
H="cortex,div=0,edges=0"
r() { python -u user_memory.py run "$@" --set lme-test 2>&1 | grep -vE "Warn|Fetching"; }
j() { python -u user_memory.py judge 2>&1 | grep -vE "Warn|Fetching"; }
echo "== stage 5: ablations @2K (LME test)"; r cortex 2000; r "cortex,div=0" 2000; r "$H,keep=0.5" 2000; r "$H,keep=0.3" 2000; r session 2000; j
echo "== stage 6: extras (LME test)"; r bm25 1000 4000; r "$H" 8000; j
echo "== queue done"
