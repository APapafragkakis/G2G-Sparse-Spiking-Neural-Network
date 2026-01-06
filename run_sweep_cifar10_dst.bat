@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM =========================
REM PATHS
REM =========================
set "PY=python"
set "TRAIN=src\train.py"

if not exist "%TRAIN%" (
  echo ERROR: cannot find "%TRAIN%" from %CD%
  pause
  exit /b 1
)

REM =========================
REM CONFIG
REM =========================
set "DATASET=cifar10"
set "MODEL=index"
set "EPOCHS=20"
set "T=50"
set "BS=256"

REM Encoding
set "ENC=current"
set "ENC_SCALE=0.25"
set "ENC_BIAS=0.0"

REM DST
set "SPARSITY=dynamic"
set "P_LIST=0.00 0.10 0.25 0.35 0.50 0.75"
set "PRUNE_LIST=set random hebb"
set "GROW_LIST=hebb random"

REM Logs
set "LOGDIR=logs_cifar10_index_dst"
if not exist "%LOGDIR%" mkdir "%LOGDIR%"

echo ============================================
echo CIFAR-10 DST SWEEP
echo model=%MODEL% epochs=%EPOCHS% T=%T% bs=%BS%
echo ============================================

for %%P in (%P_LIST%) do (
  for %%A in (%PRUNE_LIST%) do (
    for %%B in (%GROW_LIST%) do (

      set "LOG=%LOGDIR%\%DATASET%_%MODEL%_p%%P_T%T%_%ENC%_cp%%A_cg%%B.log"

      echo.
      echo ===== RUN: p'=%%P cp=%%A cg=%%B =====
      echo LOG: !LOG!

      %PY% "%TRAIN%" --dataset %DATASET% --model %MODEL% --epochs %EPOCHS% --p_inter %%P --sparsity_mode %SPARSITY% --cp %%A --cg %%B --T %T% --batch_size %BS% --enc %ENC% --enc_scale %ENC_SCALE% --enc_bias %ENC_BIAS% > "!LOG!" 2>&1

      echo Finished: p'=%%P cp=%%A cg=%%B
    )
  )
)

echo.
echo ALL DONE.
pause
