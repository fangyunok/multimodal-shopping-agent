#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
PORT="${PORT:-8001}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$PROJECT_ROOT/.venv-gpu"
OUTPUT_DIR="$PROJECT_ROOT/outputs/planner"
SERVER_LOG="$OUTPUT_DIR/vllm-server.log"
SERVER_PID=""

cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_DIR"
if ! command -v uv >/dev/null 2>&1; then
  python -m pip install --index-url https://pypi.org/simple uv
fi
if [[ ! -x "$VENV/bin/python" ]]; then
  uv venv "$VENV" --python 3.12 --seed
fi
uv pip install --python "$VENV/bin/python" vllm --torch-backend=auto
uv pip install --python "$VENV/bin/python" -e .

"$VENV/bin/vllm" serve "$MODEL" --host 127.0.0.1 --port "$PORT" \
  --dtype auto --max-model-len 4096 --gpu-memory-utilization 0.85 \
  >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

echo "Waiting for vLLM (PID $SERVER_PID)..."
for _ in $(seq 1 120); do
  if curl --fail --silent "http://127.0.0.1:$PORT/v1/models" >/dev/null; then break; fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "vLLM exited early. See $SERVER_LOG" >&2
    exit 1
  fi
  sleep 5
done
curl --fail --silent "http://127.0.0.1:$PORT/v1/models" >/dev/null || {
  echo "vLLM did not become ready. See $SERVER_LOG" >&2
  exit 1
}

AGENT="$VENV/bin/shopping-agent"
BASE_URL="http://127.0.0.1:$PORT"
"$AGENT" --catalog data/products.jsonl --planner-backend rule \
  --evaluate-planner data/planner_test.jsonl >"$OUTPUT_DIR/rule-test.json"
"$AGENT" --catalog data/products.jsonl --planner-backend llm --llm-base-url "$BASE_URL" \
  --llm-model "$MODEL" --evaluate-planner data/planner_dev.jsonl >"$OUTPUT_DIR/llm-dev.json"
"$AGENT" --catalog data/products.jsonl --planner-backend llm --llm-base-url "$BASE_URL" \
  --llm-model "$MODEL" --evaluate-planner data/planner_test.jsonl >"$OUTPUT_DIR/llm-test.json"

"$VENV/bin/python" - <<'PY'
import json
from pathlib import Path
root = Path("outputs/planner")
for name in ("rule-test", "llm-dev", "llm-test"):
    overall = json.loads((root / f"{name}.json").read_text())["overall"]
    print(name, json.dumps(overall, ensure_ascii=False))
PY

echo "Results saved under $OUTPUT_DIR"
echo "vLLM will stop now. Shut down the AutoDL instance to stop cloud billing."
