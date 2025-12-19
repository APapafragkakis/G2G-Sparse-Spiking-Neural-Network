@echo off
setlocal enabledelayedexpansion

REM =========================
REM GLOBAL DEFAULTS
REM =========================
set EPOCHS=20
set T_MAIN=50
set BS=256
set ENC_SCALE=1.0
set ENC_BIAS=0.0
set SPARSITY=static

REM p' list for main sweep
set P_LIST_MAIN=0.00 0.05 0.10 0.15 0.20 0.25 0.30 0.40 0.50 0.70 0.80 0.90 1.00

REM =========================
REM ROOT RESULTS FOLDER
REM =========================
set ROOT=results
if not exist %ROOT% mkdir %ROOT%

REM ============================================================
REM PHASE 1 — MAIN EXPERIMENT
REM dataset × model × p' (current encoding)
REM ============================================================
echo.
echo ===== PHASE 1: MAIN EXPERIMENT =====

set PHASE1=%ROOT%\phase1_main
if not exist %PHASE1% mkdir %PHASE1%

for %%D in (fashionmnist cifar10 cifar100) do (
  mkdir %PHASE1%\%%D
  for %%M in (dense index random mixer) do (
    for %%P in (%P_LIST_MAIN%) do (
      echo [PHASE 1] %%D | %%M | p'=%%P

      python train.py ^
        --dataset %%D ^
        --model %%M ^
        --p_inter %%P ^
        --epochs %EPOCHS% ^
        --T %T_MAIN% ^
        --batch_size %BS% ^
        --enc current ^
        --enc_scale %ENC_SCALE% ^
        --enc_bias %ENC_BIAS% ^
        --sparsity_mode %SPARSITY% ^
        > %PHASE1%\%%D\%%M_pinter_%%P.txt 2>&1
    )
  )
)

REM ============================================================
REM PHASE 2 — TIME WINDOW SENSITIVITY
REM T sweep (current injection) + p' sweep
REM ============================================================
echo.
echo ===== PHASE 2: TIME WINDOW SENSITIVITY =====
set PHASE3=%ROOT%\phase3_time
if not exist %PHASE3% mkdir %PHASE3%

if not exist %PHASE3%\fashionmnist mkdir %PHASE3%\fashionmnist

REM Choose the p' values you want to test here
set P_LIST_T=0.00 0.10 0.25 0.50

for %%P in (%P_LIST_T%) do (
  for %%T in (10 20 50) do (
    echo [PHASE 3] FashionMNIST ^| Index ^| p'=%%P ^| T=%%T

    python train.py ^
      --dataset fashionmnist ^
      --model index ^
      --p_inter %%P ^
      --epochs %EPOCHS% ^
      --T %%T ^
      --batch_size %BS% ^
      --enc current ^
      --enc_scale %ENC_SCALE% ^
      --enc_bias %ENC_BIAS% ^
      --sparsity_mode %SPARSITY% ^
      > %PHASE3%\fashionmnist\index_pinter_%%P_T_%%T.txt 2>&1
  )
)

echo.
echo ===== ALL EXPERIMENTS FINISHED =====
