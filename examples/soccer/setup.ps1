# PowerShell setup script for Soccer AI project
# More modern alternative to .bat file

# Get the directory where the script is located
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Check if 'data' directory exists, if not create it
$DataDir = Join-Path $ScriptDir "data"
if (-not (Test-Path $DataDir)) {
    Write-Host "Creating 'data' directory..." -ForegroundColor Green
    New-Item -ItemType Directory -Path $DataDir | Out-Null
} else {
    Write-Host "'data' directory already exists." -ForegroundColor Yellow
}

# Function to download file using gdown
function Download-File {
    param (
        [string]$OutputPath,
        [string]$FileId,
        [string]$FileName
    )
    Write-Host "Downloading $FileName..." -ForegroundColor Cyan
    & gdown -O $OutputPath "https://drive.google.com/uc?id=$FileId"
}

# Download the models
Write-Host "`n=== Downloading Models ===" -ForegroundColor Magenta
Download-File -OutputPath "$DataDir\football-ball-detection.pt" -FileId "1isw4wx-MK9h9LMr36VvIWlJD6ppUvw7V" -FileName "football-ball-detection.pt"
Download-File -OutputPath "$DataDir\football-player-detection.pt" -FileId "17PXFNlx-jI7VjVo_vQnB1sONjRyvoB-q" -FileName "football-player-detection.pt"
Download-File -OutputPath "$DataDir\football-pitch-detection.pt" -FileId "1Ma5Kt86tgpdjCTKfum79YMgNnSjcoOyf" -FileName "football-pitch-detection.pt"

# Download the videos
Write-Host "`n=== Downloading Videos ===" -ForegroundColor Magenta
Download-File -OutputPath "$DataDir\0bfacc_0.mp4" -FileId "12TqauVZ9tLAv8kWxTTBFWtgt2hNQ4_ZF" -FileName "0bfacc_0.mp4"
Download-File -OutputPath "$DataDir\2e57b9_0.mp4" -FileId "19PGw55V8aA6GZu5-Aac5_9mCy3fNxmEf" -FileName "2e57b9_0.mp4"
Download-File -OutputPath "$DataDir\08fd33_0.mp4" -FileId "1OG8K6wqUw9t7lp9ms1M48DxRhwTYciK-" -FileName "08fd33_0.mp4"
Download-File -OutputPath "$DataDir\573e61_0.mp4" -FileId "1yYPKuXbHsCxqjA9G-S6aeR2Kcnos8RPU" -FileName "573e61_0.mp4"
Download-File -OutputPath "$DataDir\121364_0.mp4" -FileId "1vVwjW1dE1drIdd4ZSILfbCGPD4weoNiu" -FileName "121364_0.mp4"

Write-Host "`nSetup completed successfully!" -ForegroundColor Green
