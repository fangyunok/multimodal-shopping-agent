$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

if (-not (Test-Path -LiteralPath ".venv\Scripts\python.exe")) {
    python -m venv .venv
}

& ".venv\Scripts\python.exe" -m pip install -e ".[dev]"
& ".venv\Scripts\python.exe" -m pytest
& ".venv\Scripts\shopping-agent.exe" --catalog data/products.jsonl --build-benchmark outputs/public_benchmark.jsonl
& ".venv\Scripts\shopping-agent.exe" --catalog data/products.jsonl --benchmark outputs/public_benchmark.jsonl
& ".venv\Scripts\shopping-agent.exe" --catalog data/products.jsonl --evaluate data/eval.jsonl
