param(
    [int]$Epochs = 30,
    [int]$BatchSize = 16,
    [int]$Workers = 0,
    [int]$Patience = 8
)

$ErrorActionPreference = "Stop"
$Python = "C:\Users\13295\anaconda3\envs\SC\python.exe"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$FeatureRoot = "D:\SceneDetection\features"
$FeatureBase = Join-Path $FeatureRoot "yolo17_nine_full"
$LabelsCsv = Join-Path $FeatureBase "labels.csv"
$VideosCsv = Join-Path $FeatureBase "videos.csv"
$OutputDir = Join-Path $ProjectRoot "scene_detect\runs_locomotion_softmax"

if (-not (Test-Path -LiteralPath $LabelsCsv -PathType Leaf)) {
    throw "Full labels not found: $LabelsCsv"
}
if (-not (Test-Path -LiteralPath $VideosCsv -PathType Leaf)) {
    throw "Full manifest not found: $VideosCsv"
}

Set-Location -LiteralPath $ProjectRoot
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:TORCH_HOME = Join-Path $ProjectRoot "scene_detect\.torch_cache"

Write-Host "`n=== Train running / walking / kicking specialist ==="
$trainArgs = @(
    "scene_detect\train.py",
    "--labels", $LabelsCsv,
    "--features-root", $FeatureRoot,
    "--labels-cols", "running,walking,kicking",
    "--include-any", "running,walking,kicking",
    "--group-column", "video_uid",
    "--model", "dual",
    "--single-label",
    "--label-smoothing", "0.05",
    "--clip-length", "32",
    "--epochs", "$Epochs",
    "--batch-size", "$BatchSize",
    "--lr", "0.0003",
    "--workers", "$Workers",
    "--patience", "$Patience",
    "--dropout", "0.45",
    "--weight-decay", "0.0005",
    "--out", $OutputDir
)

$initial = Join-Path $ProjectRoot "scene_detect\runs_nine_full_softmax\best.pt"
if (Test-Path -LiteralPath $initial -PathType Leaf) {
    $trainArgs += @(
        "--init-from", $initial,
        "--freeze-epochs", "2"
    )
    Write-Host "Warm start: $initial"
}

& $Python @trainArgs
if ($LASTEXITCODE -ne 0) {
    throw "Locomotion specialist training failed."
}

Write-Host "`nFinished."
Write-Host "Model:  $(Join-Path $OutputDir 'best.pt')"
Write-Host "Report: $(Join-Path $OutputDir 'report.json')"
