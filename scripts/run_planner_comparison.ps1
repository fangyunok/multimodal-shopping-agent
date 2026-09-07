param(
    [Parameter(Mandatory = $true)][string]$BaseUrl,
    [Parameter(Mandatory = $true)][string]$Model
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$agentExecutable = ".venv\Scripts\shopping-agent.exe"

if (-not (Test-Path -LiteralPath $agentExecutable -PathType Leaf)) {
    throw "Run scripts/run_cpu_baseline.ps1 first."
}

function Assert-LastExit([string]$Step) {
    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed with exit code $LASTEXITCODE"
    }
}

Write-Output "Rule Planner: locked 100-case test set"
& $agentExecutable --catalog data/products.jsonl --planner-backend rule --evaluate-planner data/planner_test.jsonl
Assert-LastExit "Rule Planner evaluation"

Write-Output "LLM Planner: 20-case development set"
& $agentExecutable --catalog data/products.jsonl --planner-backend llm --llm-base-url $BaseUrl --llm-model $Model --evaluate-planner data/planner_dev.jsonl
Assert-LastExit "LLM Planner development evaluation"

Write-Output "LLM Planner: locked 100-case test set"
& $agentExecutable --catalog data/products.jsonl --planner-backend llm --llm-base-url $BaseUrl --llm-model $Model --evaluate-planner data/planner_test.jsonl
Assert-LastExit "LLM Planner locked evaluation"
