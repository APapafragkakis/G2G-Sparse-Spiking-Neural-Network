@echo off
echo =========================================
echo   Running IndexSNN Experiments (p')
echo   epochs=10 | batch_size=128 | T=25
echo =========================================

REM ---- Create logs directory if it doesn't exist ----
if not exist logs (
    mkdir logs
)

echo.
echo ---- RUN 1: p_inter = 0.00 ----
python train.py --model index --epochs 10 --batch_size 128 --T 25 --p_inter 0.00 > logs/p_000.txt
echo Log saved to logs/p_000.txt
echo -----------------------------

echo.
echo ---- RUN 2: p_inter = 0.10 ----
python train.py --model index --epochs 10 --batch_size 128 --T 25 --p_inter 0.10 > logs/p_010.txt
echo Log saved to logs/p_010.txt
echo -----------------------------

echo.
echo ---- RUN 3: p_inter = 0.25 ----
python train.py --model index --epochs 10 --batch_size 128 --T 25 --p_inter 0.25 > logs/p_025.txt
echo Log saved to logs/p_025.txt
echo -----------------------------

echo.
echo All experiments completed!
pause
