@echo off
REM ============================================================
REM  Fish4Knowledge Full Training Pipeline
REM  Estimated total time: 5-6 hours on RTX 4060 Laptop GPU
REM ============================================================

set DATA_ROOT=d:\prakhar ipa\my dataset\fish4konwledge
set SRC_DIR=%DATA_ROOT%\src

echo ============================================================
echo  Fish4Knowledge Training Pipeline
echo  Started: %date% %time%
echo ============================================================

REM Step 1: Train ConvNeXt-Base (primary model)
echo.
echo [Step 1/4] Training ConvNeXt-Base (estimated: 2.5-5 hours)
echo ============================================================
python "%SRC_DIR%\train.py" --model convnext_base --image-size 288 --batch-size 4 --epochs 50 --lr 1e-4 --loss class_balanced --grad-accum 4 --warmup-epochs 3 --patience 8 --data-root "%DATA_ROOT%"
if errorlevel 1 (
    echo ERROR: ConvNeXt-Base training failed!
    pause
    exit /b 1
)

REM Step 2: Evaluate ConvNeXt-Base
echo.
echo [Step 2/4] Evaluating ConvNeXt-Base
echo ============================================================
for /f "delims=" %%i in ('dir /b /od /ad "%DATA_ROOT%\runs\*convnext*" 2^>nul') do set CONVNEXT_RUN=%%i
python "%SRC_DIR%\evaluate.py" --checkpoint "%DATA_ROOT%\runs\%CONVNEXT_RUN%\best_macro_f1.pt" --model convnext_base --image-size 288 --split val --tta
if errorlevel 1 (
    echo WARNING: ConvNeXt-Base evaluation had issues, continuing...
)

REM Step 3: Train EfficientNetV2-M (secondary model for ensemble)
echo.
echo [Step 3/4] Training EfficientNetV2-M (estimated: 2-4 hours)
echo ============================================================
python "%SRC_DIR%\train.py" --model efficientnetv2_m --image-size 288 --batch-size 4 --epochs 50 --lr 1e-4 --loss class_balanced --grad-accum 4 --warmup-epochs 3 --patience 8 --data-root "%DATA_ROOT%"
if errorlevel 1 (
    echo ERROR: EfficientNetV2-M training failed!
    pause
    exit /b 1
)

REM Step 4: Ensemble evaluation
echo.
echo [Step 4/4] Running Ensemble + TTA
echo ============================================================
for /f "delims=" %%i in ('dir /b /od /ad "%DATA_ROOT%\runs\*efficientnet*" 2^>nul') do set EFFNET_RUN=%%i
python "%SRC_DIR%\inference.py" --checkpoint-a "%DATA_ROOT%\runs\%CONVNEXT_RUN%\best_macro_f1.pt" --checkpoint-b "%DATA_ROOT%\runs\%EFFNET_RUN%\best_macro_f1.pt" --model-a convnext_base --model-b tf_efficientnetv2_m --image-size 288 --ensemble --tta --split test
if errorlevel 1 (
    echo WARNING: Ensemble evaluation had issues
)

echo.
echo ============================================================
echo  Pipeline Complete!
echo  Finished: %date% %time%
echo  Check runs/ for results and plots
echo ============================================================
pause
