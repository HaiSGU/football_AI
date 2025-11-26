@echo off
REM Windows setup script for Soccer AI project

REM Get the directory where the script is located
set "SCRIPT_DIR=%~dp0"

REM Check if 'data' directory exists, if not create it
if not exist "%SCRIPT_DIR%data" (
    echo Creating 'data' directory...
    mkdir "%SCRIPT_DIR%data"
) else (
    echo 'data' directory already exists.
)

REM Download the models
echo Downloading football-ball-detection model...
gdown -O "%SCRIPT_DIR%data\football-ball-detection.pt" "https://drive.google.com/uc?id=1isw4wx-MK9h9LMr36VvIWlJD6ppUvw7V"

echo Downloading football-player-detection model...
gdown -O "%SCRIPT_DIR%data\football-player-detection.pt" "https://drive.google.com/uc?id=17PXFNlx-jI7VjVo_vQnB1sONjRyvoB-q"

echo Downloading football-pitch-detection model...
gdown -O "%SCRIPT_DIR%data\football-pitch-detection.pt" "https://drive.google.com/uc?id=1Ma5Kt86tgpdjCTKfum79YMgNnSjcoOyf"

REM Download the videos
echo Downloading video 0bfacc_0.mp4...
gdown -O "%SCRIPT_DIR%data\0bfacc_0.mp4" "https://drive.google.com/uc?id=12TqauVZ9tLAv8kWxTTBFWtgt2hNQ4_ZF"

echo Downloading video 2e57b9_0.mp4...
gdown -O "%SCRIPT_DIR%data\2e57b9_0.mp4" "https://drive.google.com/uc?id=19PGw55V8aA6GZu5-Aac5_9mCy3fNxmEf"

echo Downloading video 08fd33_0.mp4...
gdown -O "%SCRIPT_DIR%data\08fd33_0.mp4" "https://drive.google.com/uc?id=1OG8K6wqUw9t7lp9ms1M48DxRhwTYciK-"

echo Downloading video 573e61_0.mp4...
gdown -O "%SCRIPT_DIR%data\573e61_0.mp4" "https://drive.google.com/uc?id=1yYPKuXbHsCxqjA9G-S6aeR2Kcnos8RPU"

echo Downloading video 121364_0.mp4...
gdown -O "%SCRIPT_DIR%data\121364_0.mp4" "https://drive.google.com/uc?id=1vVwjW1dE1drIdd4ZSILfbCGPD4weoNiu"

echo.
echo Setup completed successfully!
pause
