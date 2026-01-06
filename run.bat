@echo off
cd /d "%~dp0"
setlocal EnableExtensions EnableDelayedExpansion

REM =========================
REM PATHS
REM =========================
set "PY=python"
set "TRAIN=src\train.py"

if not exist "%TRAIN%" (
  echo [ERROR] Cannot find "%TRAIN%"
  echo Current dir: %CD%
  pause
  exit /b 1
)

REM =========================
REM GLOBAL CONFIG
REM =========================
set "T=50"
set "BS=256"
set "EPOCHS=20"
set "USE_RESNET=--use_resnet"
set "SPARSEMODE=--sparsity_mode static"
set "H_SPARSE=2048"
set "P_LIST=0 0.05 0.15 0.25 0.35 0.5 0.75"

echo =====================================================
echo START SWEEP (CIFAR10 then CIFAR100)
echo epochs=%EPOCHS%  T=%T%  batch=%BS%
echo p_inter grid: %P_LIST%
echo =====================================================

for %%D in (cifar10 cifar100) do (
  set "DATASET=%%D"
  set "LOGDIR=logs_resnet_%%D_T%T%_bs%BS%_e%EPOCHS%_static_capmatch_pgrid"
  if not exist "!LOGDIR!" mkdir "!LOGDIR!"

  echo.
  echo =====================================================
  echo DATASET=%%D  epochs=%EPOCHS%
  echo Logs: !LOGDIR!
  echo =====================================================

  for %%P in (%P_LIST%) do (
    set "H_DENSE="
    if "%%P"=="0"     set "H_DENSE=724"
    if "%%P"=="0.05"  set "H_DENSE=841"
    if "%%P"=="0.15"  set "H_DENSE=1037"
    if "%%P"=="0.25"  set "H_DENSE=1201"
    if "%%P"=="0.35"  set "H_DENSE=1345"
    if "%%P"=="0.5"   set "H_DENSE=1536"
    if "%%P"=="0.75"  set "H_DENSE=1810"

    if not defined H_DENSE (
      echo [ERROR] No dense mapping for p_inter=%%P
      pause
      exit /b 1
    )

    set "PSTR=%%P"
    set "PSTR=!PSTR:.=_!"

    echo.
    echo ---- %%D - p_inter=%%P - dense_h=!H_DENSE! - sparse_h=%H_SPARSE% ----

    echo [RUN] dense
    %PY% "%TRAIN%" --dataset %%D --model dense %USE_RESNET% %SPARSEMODE% ^
      --p_inter %%P --epochs %EPOCHS% --T %T% --batch_size %BS% --hidden_dim !H_DENSE! ^
      > "!LOGDIR!\dense_capmatch_p!PSTR!_h!H_DENSE!.log" 2>&1
    
    if errorlevel 1 (
      echo [ERROR] dense failed for dataset=%%D p=%%P
      echo Check log: !LOGDIR!\dense_capmatch_p!PSTR!_h!H_DENSE!.log
      pause
      exit /b 1
    )

    for %%M in (index random mixer) do (
      echo [RUN] %%M
      %PY% "%TRAIN%" --dataset %%D --model %%M %USE_RESNET% %SPARSEMODE% ^
        --p_inter %%P --epochs %EPOCHS% --T %T% --batch_size %BS% --hidden_dim %H_SPARSE% ^
        > "!LOGDIR!\%%M_resnet_p!PSTR!_h%H_SPARSE%.log" 2>&1
      
      if errorlevel 1 (
        echo [ERROR] %%M failed for dataset=%%D p=%%P
        echo Check log: !LOGDIR!\%%M_resnet_p!PSTR!_h%H_SPARSE%.log
        pause
        exit /b 1
      )
    )
  )
)

echo.
echo =====================================================
echo ALL RUNS COMPLETED SUCCESSFULLY
echo =====================================================
pause
endlocal