param(
    [int]$Epochs = 40,
    [int]$BatchSize = 16,
    [int]$Workers = 0,
    [int]$Patience = 10,
    [switch]$SkipPreparation,
    [switch]$SkipAppearance
)

$ErrorActionPreference = "Stop"
$Python = "C:\Users\13295\anaconda3\envs\SC\python.exe"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$HmdbFrames = "D:\HMDB51\HMDB51"
$HmdbVideos = "D:\SceneDetection\datasets\hmdb51_org"
$NoddingVideos = "D:\SceneDetection\datasets\nodding_videos"
$FeatureRoot = "D:\SceneDetection\features"
$FeatureBase = Join-Path $FeatureRoot "yolo17_nine_full"
$LabelsCsv = Join-Path $FeatureBase "labels.csv"
$VideosCsv = Join-Path $FeatureBase "videos.csv"
$HoldoutFile = Join-Path $ProjectRoot "nine_holdout_sources.txt"
$OutputDir = Join-Path $ProjectRoot "scene_detect\runs_nine_full_softmax"
$HmdbActions = "run,fall_floor,sit,walk,kick,kick_ball,punch,wave,shake_hands"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python environment not found: $Python"
}
if (-not (Test-Path -LiteralPath $HoldoutFile -PathType Leaf)) {
    throw "Holdout source list not found: $HoldoutFile"
}

Set-Location -LiteralPath $ProjectRoot
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:TORCH_HOME = Join-Path $ProjectRoot "scene_detect\.torch_cache"
$env:YOLO_CONFIG_DIR = Join-Path $env:TEMP "scene_detect_yolo"

if (-not $SkipPreparation) {
    Write-Host "`n=== 1/5 Convert all selected HMDB51 clips ==="
    & $Python "scene_detect\adapters\prepare_hmdb_videos.py" `
        --source $HmdbFrames `
        --output $HmdbVideos `
        --actions $HmdbActions `
        --per-class 0 `
        --min-frames 48
    if ($LASTEXITCODE -ne 0) {
        throw "HMDB51 conversion failed."
    }

    Write-Host "`n=== 2/5 Build full manifest with fixed holdout excluded ==="
    & $Python "scene_detect\adapters\build_yolo17_nine_manifest.py" `
        --hmdb-root $HmdbVideos `
        --nodding-root $NoddingVideos `
        --per-class 0 `
        --exclude-sources $HoldoutFile `
        --out $VideosCsv
    if ($LASTEXITCODE -ne 0) {
        throw "Full nine-class manifest generation failed."
    }

    Write-Host "`n=== 3/5 Extract YOLO17 skeleton features ==="
    & $Python "scene_detect\adapters\extract_new.py" `
        --videos $VideosCsv `
        --out-root $FeatureBase `
        --labels $LabelsCsv `
        --datasets "hmdb51,nodding" `
        --clip-length 32 `
        --stride 16 `
        --max-clips 8
    if ($LASTEXITCODE -ne 0) {
        throw "YOLO17 feature extraction failed."
    }
}

if (-not (Test-Path -LiteralPath $LabelsCsv -PathType Leaf)) {
    throw "Full labels not found. Run without -SkipPreparation first: $LabelsCsv"
}
if (-not (Test-Path -LiteralPath $VideosCsv -PathType Leaf)) {
    throw "Full manifest not found. Run without -SkipPreparation first: $VideosCsv"
}

if (-not $SkipAppearance) {
    Write-Host "`n=== 4/5 Extract RGB appearance features ==="
    & $Python "scene_detect\adapters\extract_appearance_new.py" `
        --labels $LabelsCsv `
        --videos $VideosCsv `
        --datasets "hmdb51,nodding"
    if ($LASTEXITCODE -ne 0) {
        throw "Appearance feature extraction failed."
    }
}

Write-Host "`n=== 5/5 Train full-data nine-class softmax model ==="
$columns = "running,falling,fighting,sitting,walking,kicking,waving,nodding,handshaking"
$trainArgs = @(
    "scene_detect\train.py",
    "--labels", $LabelsCsv,
    "--features-root", $FeatureRoot,
    "--labels-cols", $columns,
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

$initial = Join-Path $ProjectRoot "scene_detect\runs_nine_softmax\best.pt"
if (Test-Path -LiteralPath $initial -PathType Leaf) {
    $trainArgs += @("--init-from", $initial)
    Write-Host "Warm start: $initial"
}

& $Python @trainArgs
if ($LASTEXITCODE -ne 0) {
    throw "Full-data nine-class training failed."
}

Write-Host "`nFinished."
Write-Host "Model:  $(Join-Path $OutputDir 'best.pt')"
Write-Host "Report: $(Join-Path $OutputDir 'report.json')"
