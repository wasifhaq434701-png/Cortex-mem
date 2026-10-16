# queue.sh can't be edited while bash reads it: stop it the moment stage 5 starts, then run queue2.sh.
L="$(dirname "$0")/queue.log"
until grep -q "== stage 5" "$L"; do sleep 5; done
pkill -f "queue.sh"; sleep 1; pkill -f "user_memory.py run"; sleep 5
echo "== handoff: queue.sh stopped at stage 5, starting queue2.sh (LME test only)" >> "$L"
caffeinate -i bash "$(dirname "$0")/queue2.sh" >> "$L" 2>&1
