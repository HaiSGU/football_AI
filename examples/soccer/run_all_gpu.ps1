# Run All Modes with GPU - Batch Processing Script
# Chạy tất cả 6 modes của Soccer AI bằng GPU

$modes = @(
    "PITCH_DETECTION",
    "PLAYER_DETECTION", 
    "BALL_DETECTION",
    "PLAYER_TRACKING",
    "TEAM_CLASSIFICATION",
    "RADAR"
)

$video = "data\0bfacc_0.mp4"

# Check if CUDA is available
Write-Host "🔍 Checking CUDA availability..." -ForegroundColor Yellow
$cudaCheck = python -c "import torch; print(torch.cuda.is_available())" 2>$null

if ($cudaCheck -ne "True") {
    Write-Host "⚠️  WARNING: CUDA not available! Processing will be VERY slow on CPU." -ForegroundColor Red
    Write-Host "   Install PyTorch with CUDA: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121" -ForegroundColor Yellow
    Write-Host ""
    $continue = Read-Host "Continue with CPU? (y/n)"
    if ($continue -ne "y") {
        exit
    }
    $device = "cpu"
} else {
    $gpuName = python -c "import torch; print(torch.cuda.get_device_name(0))" 2>$null
    Write-Host "✅ CUDA available! GPU: $gpuName" -ForegroundColor Green
    $device = "cuda"
}

Write-Host ""
Write-Host "🚀 Starting batch processing..." -ForegroundColor Green
Write-Host "   Device: $device" -ForegroundColor Cyan
Write-Host "   Input video: $video" -ForegroundColor Cyan
Write-Host "   Total modes: $($modes.Count)" -ForegroundColor Cyan
Write-Host ""

$startTime = Get-Date
$successCount = 0
$failCount = 0

foreach ($mode in $modes) {
    $output = "output\$($mode.ToLower()).mp4"
    
    Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Gray
    Write-Host "▶️  Processing: $mode" -ForegroundColor Cyan
    Write-Host "   Input:  $video" -ForegroundColor Gray
    Write-Host "   Output: $output" -ForegroundColor Gray
    Write-Host ""
    
    $modeStartTime = Get-Date
    
    python main.py `
        --source_video_path $video `
        --target_video_path $output `
        --device $device `
        --mode $mode `
        --no-display
    
    $modeEndTime = Get-Date
    $duration = ($modeEndTime - $modeStartTime).TotalSeconds
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "✅ $mode completed in $([math]::Round($duration, 1))s" -ForegroundColor Green
        $successCount++
    } else {
        Write-Host ""
        Write-Host "❌ $mode failed!" -ForegroundColor Red
        $failCount++
    }
    Write-Host ""
}

$endTime = Get-Date
$totalDuration = ($endTime - $startTime).TotalSeconds

Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Gray
Write-Host ""
Write-Host "🎉 Batch processing completed!" -ForegroundColor Green
Write-Host ""
Write-Host "📊 Summary:" -ForegroundColor Cyan
Write-Host "   ✅ Success: $successCount" -ForegroundColor Green
Write-Host "   ❌ Failed:  $failCount" -ForegroundColor Red
Write-Host "   ⏱️  Total time: $([math]::Round($totalDuration, 1))s ($([math]::Round($totalDuration/60, 1)) minutes)" -ForegroundColor Yellow
Write-Host ""
Write-Host "📁 Output files saved in: output\" -ForegroundColor Cyan
Write-Host ""

# List output files
if (Test-Path "output") {
    $outputFiles = Get-ChildItem "output\*.mp4" | Sort-Object Name
    if ($outputFiles.Count -gt 0) {
        Write-Host "📹 Generated videos:" -ForegroundColor Cyan
        foreach ($file in $outputFiles) {
            $sizeMB = [math]::Round($file.Length / 1MB, 2)
            Write-Host "   - $($file.Name) ($sizeMB MB)" -ForegroundColor Gray
        }
    }
}

Write-Host ""
Write-Host "✨ Done! You can now view the videos in the output folder." -ForegroundColor Green
