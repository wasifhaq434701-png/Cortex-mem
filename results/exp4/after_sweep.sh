# GPU timing needs the GPU to itself: wait for the sweep, then TTFT, then a clean (idle-machine) retrieval re-run.
D="$(cd "$(dirname "$0")" && pwd)"; L="$D/queue.log"
until grep -q "== queue done" "$L"; do sleep 30; done
cd "$(dirname "$0")/../../experiments"
{ echo "== latency: ttft"; python -u user_memory.py latency ttft 2>&1 | grep -vE "Warn|Fetching|Loading weights"
  echo "== latency: retrieval (idle machine)"; python -u user_memory.py latency retrieval 2>&1 | grep -vE "Warn|Fetching|Loading weights" | tail -3
  echo "== latency done"; } >> "$L" 2>&1
