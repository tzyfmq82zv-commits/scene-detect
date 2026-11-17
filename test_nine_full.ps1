param(
    [string]$Classifier = "",
    [string]$Video = "",
    [string]$Timeline = "",
    [string]$OutputPrefix = "",
    [int]$TemporalConfirmations = 3,
    [int]$InferenceSize = 640
)

$ErrorActionPreference = "Stop"
$Python = "C:\Users\13295\anaconda3\envs\SC\python.exe"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$WorkspaceRoot = Split-Path -Parent $ProjectRoot

if (-not $Classifier) {
    $full_model = Join-Path $ProjectRoot "scene_detect\runs_nine_full_softmax\best.pt"
    $Classifier = $full_model
}
if (-not $Video) {
    $Video = Join-Path $WorkspaceRoot "scene_all_categories_no_gap.mp4"
}
if (-not $Timeline) {
    $Timeline = Join-Path $WorkspaceRoot "scene_all_categories_no_gap_timeline.json"
}
if (-not $OutputPrefix) {
    $OutputPrefix = Join-Path $WorkspaceRoot "scene_all_categories_no_gap_result"
}

foreach ($path in @($Classifier, $Video, $Timeline)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required file not found: $path"
    }
}

Set-Location -LiteralPath $ProjectRoot
$env:YOLO_CONFIG_DIR = Join-Path $env:TEMP "scene_detect_yolo"
$env:TORCH_HOME = Join-Path $ProjectRoot "scene_detect\.torch_cache"

$annotated = "${OutputPrefix}_annotated.mp4"
$events = "${OutputPrefix}_events.json"
$summary = "${OutputPrefix}_summary.json"

Write-Host "=== Run end-to-end nine-action test ==="
& $Python "scene_detect\run_video.py" `
    --source $Video `
    --classifier $Classifier `
    --output $annotated `
    --events $events `
    --alarm-fighting `
    --alarm-running `
    --temporal-confirmations $TemporalConfirmations `
    --imgsz $InferenceSize
if ($LASTEXITCODE -ne 0) {
    throw "End-to-end video test failed."
}

Write-Host "`n=== Evaluate against timeline ==="
& $Python "scene_detect\adapters\evaluate_timeline.py" `
    --timeline $Timeline `
    --events $events `
    --out $summary
if ($LASTEXITCODE -ne 0) {
    throw "Timeline evaluation failed."
}

Write-Host "`nFinished."
Write-Host "Annotated: $annotated"
Write-Host "Events:    $events"
Write-Host "Summary:   $summary"
