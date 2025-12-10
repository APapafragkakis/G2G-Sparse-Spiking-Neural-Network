@echo off
cd /d "%~dp0"

echo RUN 1: p_inter = 0.00
python "src\train.py" --model index --epochs 10 --batch_size 128 --T 25 --p_inter 0.00

echo RUN 2: p_inter = 0.10
python "src\train.py" --model index --epochs 10 --batch_size 128 --T 25 --p_inter 0.10

echo RUN 3: p_inter = 0.25
python "src\train.py" --model index --epochs 10 --batch_size 128 --T 25 --p_inter 0.25

pause
