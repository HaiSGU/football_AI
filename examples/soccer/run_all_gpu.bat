@echo off
REM Run All Modes with GPU - Batch Processing Script
REM Chay tat ca 6 modes cua Soccer AI bang GPU

echo ========================================
echo   Soccer AI - Batch GPU Processing
echo ========================================
echo.

REM Check CUDA
echo Checking CUDA availability...
python -c "import torch; print('CUDA:', torch.cuda.is_available())" 2>nul
if errorlevel 1 (
    echo WARNING: Could not check CUDA. Make sure PyTorch is installed.
    echo.
)

echo.
echo Starting batch processing...
echo Device: cuda
echo Input: data\0bfacc_0.mp4
echo.
echo ========================================
echo.

REM Mode 1: PITCH_DETECTION
echo [1/6] Processing PITCH_DETECTION...
python main.py --source_video_path data\0bfacc_0.mp4 --target_video_path output\pitch_detection.mp4 --device cuda --mode PITCH_DETECTION --no-display
if errorlevel 1 (
    echo ERROR: PITCH_DETECTION failed!
) else (
    echo SUCCESS: PITCH_DETECTION completed!
)
echo.
echo ----------------------------------------
echo.

REM Mode 2: PLAYER_DETECTION
echo [2/6] Processing PLAYER_DETECTION...
python main.py --source_video_path data\0bfacc_0.mp4 --target_video_path output\player_detection.mp4 --device cuda --mode PLAYER_DETECTION --no-display
if errorlevel 1 (
    echo ERROR: PLAYER_DETECTION failed!
) else (
    echo SUCCESS: PLAYER_DETECTION completed!
)
echo.
echo ----------------------------------------
echo.

REM Mode 3: BALL_DETECTION
echo [3/6] Processing BALL_DETECTION...
python main.py --source_video_path data\0bfacc_0.mp4 --target_video_path output\ball_detection.mp4 --device cuda --mode BALL_DETECTION --no-display
if errorlevel 1 (
    echo ERROR: BALL_DETECTION failed!
) else (
    echo SUCCESS: BALL_DETECTION completed!
)
echo.
echo ----------------------------------------
echo.

REM Mode 4: PLAYER_TRACKING
echo [4/6] Processing PLAYER_TRACKING...
python main.py --source_video_path data\0bfacc_0.mp4 --target_video_path output\player_tracking.mp4 --device cuda --mode PLAYER_TRACKING --no-display
if errorlevel 1 (
    echo ERROR: PLAYER_TRACKING failed!
) else (
    echo SUCCESS: PLAYER_TRACKING completed!
)
echo.
echo ----------------------------------------
echo.

REM Mode 5: TEAM_CLASSIFICATION
echo [5/6] Processing TEAM_CLASSIFICATION...
python main.py --source_video_path data\0bfacc_0.mp4 --target_video_path output\team_classification.mp4 --device cuda --mode TEAM_CLASSIFICATION --no-display
if errorlevel 1 (
    echo ERROR: TEAM_CLASSIFICATION failed!
) else (
    echo SUCCESS: TEAM_CLASSIFICATION completed!
)
echo.
echo ----------------------------------------
echo.

REM Mode 6: RADAR
echo [6/6] Processing RADAR...
python main.py --source_video_path data\0bfacc_0.mp4 --target_video_path output\radar.mp4 --device cuda --mode RADAR --no-display
if errorlevel 1 (
    echo ERROR: RADAR failed!
) else (
    echo SUCCESS: RADAR completed!
)
echo.
echo ========================================
echo.

echo All modes completed!
echo.
echo Output files saved in: output\
echo.

REM List output files
if exist output\*.mp4 (
    echo Generated videos:
    dir /B output\*.mp4
)

echo.
echo Done! You can now view the videos in the output folder.
echo.
pause
