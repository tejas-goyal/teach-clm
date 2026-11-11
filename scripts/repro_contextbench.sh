#!/usr/bin/env bash
# M2: run the paper's CLM agent (ClmStream, unmodified) on ContextBench tasks through Harbor,
# against our Modal-served Qwen3.5-9B. Needs Docker running and `modal deploy infra/serve_vllm.py`.
#
#   scripts/repro_contextbench.sh <out_name> <task> [<task> ...]     # task = dir name under third_party/ContextBench/tasks
#   PARALLEL=4 scripts/repro_contextbench.sh slice1 kv-store-R1000-seed42 kv-store-R1000-seed43
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; . ./.env; set +a

OUT=runs/m2/$1; shift
API_BASE=${API_BASE:-$(uv run python -c "import modal; print(modal.Function.from_name('teachclm-vllm','serve').get_web_url())")/v1}
export OPENAI_API_KEY=$TEACHCLM_API_KEY
mkdir -p "$OUT"
echo "endpoint: $API_BASE   out: $OUT   tasks: $#"

run_one() {
  if uv run harbor trial start -p "third_party/ContextBench/tasks/$1" -e docker \
      -a clm_contextbench:ClmStream -m openai/qwen35-9b \
      --agent-kwarg "api_base=$API_BASE" \
      --trial-name "$1" --trials-dir "$OUT" > "$OUT/$1.log" 2>&1 &&
     [ ! -e "$OUT/$1/exception.txt" ]   # harbor exits 0 even when the trial raised
  then echo "done $1"; else echo "FAILED $1 (see $OUT/$1.log)"; fi
}
export -f run_one; export OUT API_BASE
printf '%s\n' "$@" | xargs -n 1 -P "${PARALLEL:-1}" bash -c 'run_one "$0"'
