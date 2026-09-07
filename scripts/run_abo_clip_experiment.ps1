param(
    [Parameter(Mandatory = $true)][string]$AboRoot,
    [Parameter(Mandatory = $true)][string]$ModelPath,
    [int]$Limit = 100,
    [int]$Workers = 8,
    [int]$BatchSize = 4
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

if (-not (Test-Path -LiteralPath $AboRoot -PathType Container)) {
    throw "ABO root does not exist: $AboRoot"
}
if (-not (Test-Path -LiteralPath $ModelPath -PathType Container)) {
    throw "Local Chinese-CLIP model directory does not exist: $ModelPath"
}
if ($Limit -lt 1 -or $Workers -lt 1 -or $BatchSize -lt 1) {
    throw "Limit, Workers and BatchSize must all be positive."
}

function Assert-LastExit([string]$Step) {
    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed with exit code $LASTEXITCODE"
    }
}

if (-not (Test-Path -LiteralPath ".venv\Scripts\python.exe")) {
    python -m venv .venv
}

& ".venv\Scripts\python.exe" -m pip install -e ".[clip,dev]"
Assert-LastExit "Dependency installation"
& ".venv\Scripts\python.exe" -m pytest
Assert-LastExit "Test suite"
$agentExecutable = ".venv\Scripts\shopping-agent.exe"
$catalog = "data/processed/products.jsonl"
$modelId = "OFA-Sys/chinese-clip-vit-base-patch16"

& $agentExecutable --fetch-abo-images $AboRoot --abo-limit $Limit --download-workers $Workers --include-other-images
Assert-LastExit "ABO image download"
& $agentExecutable --convert-abo $AboRoot --abo-limit $Limit --abo-output data/raw/abo.jsonl
Assert-LastExit "ABO conversion"
& $agentExecutable --prepare-data data/raw/abo.jsonl --output-catalog $catalog --image-dir data/processed/images
Assert-LastExit "Dataset preparation"
& $agentExecutable --catalog $catalog --build-benchmark outputs/retrieval_benchmark.jsonl
Assert-LastExit "Attribute benchmark generation"
& $agentExecutable --catalog $catalog --build-cross-view-benchmark $AboRoot --cross-view-output outputs/cross_view_benchmark.jsonl
Assert-LastExit "Cross-view benchmark generation"

foreach ($configuration in @(
    @{ Name = "text-only"; Directory = "data/index/chinese-clip-text-only"; ImageWeight = "0" },
    @{ Name = "image-only"; Directory = "data/index/chinese-clip-image-only"; ImageWeight = "1" },
    @{ Name = "text-image"; Directory = "data/index/chinese-clip-base"; ImageWeight = "0.5" }
)) {
    Write-Output "Building $($configuration.Name) index"
    & $agentExecutable --catalog $catalog --build-index $configuration.Directory --model $ModelPath --model-id $modelId --device cpu --batch-size $BatchSize --image-weight $configuration.ImageWeight
    Assert-LastExit "$($configuration.Name) index build"
}

Write-Output "Attribute benchmark: lexical CPU baseline"
& $agentExecutable --catalog $catalog --benchmark outputs/retrieval_benchmark.jsonl
Assert-LastExit "Lexical attribute benchmark"
Write-Output "Attribute benchmark: lexical + Chinese-CLIP fusion"
& $agentExecutable --catalog $catalog --retriever-backend fusion --lexical-weight 0.65 --index-dir data/index/chinese-clip-base --model $ModelPath --model-id $modelId --device cpu --batch-size $BatchSize --benchmark outputs/retrieval_benchmark.jsonl
Assert-LastExit "Fusion attribute benchmark"
Write-Output "Cross-view benchmark: RGB CPU baseline"
& $agentExecutable --catalog $catalog --benchmark outputs/cross_view_benchmark.jsonl
Assert-LastExit "RGB cross-view benchmark"
Write-Output "Cross-view benchmark: Chinese-CLIP image-only index"
& $agentExecutable --catalog $catalog --retriever-backend clip --index-dir data/index/chinese-clip-image-only --model $ModelPath --model-id $modelId --device cpu --batch-size $BatchSize --benchmark outputs/cross_view_benchmark.jsonl
Assert-LastExit "Image-only CLIP cross-view benchmark"
Write-Output "Cross-view benchmark: Chinese-CLIP text-image index"
& $agentExecutable --catalog $catalog --retriever-backend clip --index-dir data/index/chinese-clip-base --model $ModelPath --model-id $modelId --device cpu --batch-size $BatchSize --benchmark outputs/cross_view_benchmark.jsonl
Assert-LastExit "Text-image CLIP cross-view benchmark"
