param(
    [string]$Video = "",
    [string]$Classifier = ""
)

$ErrorActionPreference = "Stop"
$Python = "C:\Users\13295\anaconda3\envs\SC\python.exe"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$WorkspaceRoot = Split-Path -Parent $ProjectRoot

if (-not $Video) {
    $Video = Join-Path $WorkspaceRoot "scene_all_categories_no_gap.mp4"
}
if (-not $Classifier) {
    $Classifier = Join-Path $ProjectRoot "scene_detect\runs_nine_full_softmax\best.pt"
}
foreach ($path in @($Python, $Video, $Classifier)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required file not found: $path"
    }
}

Set-Location -LiteralPath $ProjectRoot
$env:YOLO_CONFIG_DIR = Join-Path $env:TEMP "scene_detect_yolo"
$env:TORCH_HOME = Join-Path $ProjectRoot "scene_detect\.torch_cache"

& $Python "scene_detect\app_qt.py" `
    --source $Video `
    --classifier $Classifier
