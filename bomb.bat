@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM =========================
REM PATHS
REM =========================
set "TRAIN=src\train.py"
if not exist "%TRAIN%" (
  echo ERROR: cannot find "%TRAIN%" from %CD%
  echo Make sure you run this .bat from the project root.
  exit /b 1
)

REM =========================
REM GLOBAL DEFAULTS
REM =========================
set "EPOCHS=20"
set "T_MAIN=50"
set "BS=256"
set "ENC_SCALE=1.0"
set "ENC_BIAS=0.0"
set "SPARSITY=static"

REM p' lists
set "P_LIST_MAIN=0.00 0.05 0.10 0.15 0.20 0.35 0.50 0.75 1.00"
set "P_LIST_T=0.00 0.10 0.25 0.50"

REM Root output folder (relative to this project folder)
set "ROOT=results"
if not exist "%ROOT%" mkdir "%ROOT%"

REM ============================================================
REM PHASE 1 — MAIN EXPERIMENT
REM dataset × model × p' × encoding (current + rate)
REM - dense runs ONCE per dataset/encoding (p' ignored)
REM - non-dense models sweep over p'
REM - skips runs whose output log already exists
REM ============================================================
echo.
echo ===== PHASE 1: MAIN EXPERIMENT =====

set "PHASE1=%ROOT%\phase1_main"
if not exist "%PHASE1%" mkdir "%PHASE1%"

for %%E in (current rate) do (
  if not exist "%PHASE1%\%%E" mkdir "%PHASE1%\%%E"

  for %%D in (fashionmnist cifar10 cifar100) do (
    if not exist "%PHASE1%\%%E\%%D" mkdir "%PHASE1%\%%E\%%D"

    REM -------------------------
    REM DENSE: run once
    REM -------------------------
    set "OUT=%PHASE1%\%%E\%%D\dense.txt"
    if exist "!OUT!" (
      echo [SKIP] PHASE1 ^| enc=%%E ^| dataset=%%D ^| model=dense ^| exists: !OUT!
    ) else (
      echo [PHASE1] enc=%%E ^| dataset=%%D ^| model=dense

      python "%TRAIN%" ^
        --dataset %%D ^
        --model dense ^
        --p_inter 0.00 ^
        --epochs %EPOCHS% ^
        --T %T_MAIN% ^
        --batch_size %BS% ^
        --enc %%E ^
        --enc_scale %ENC_SCALE% ^
        --enc_bias %ENC_BIAS% ^
        --sparsity_mode %SPARSITY% ^
        > "!OUT!" 2>&1
    )

    REM -------------------------
    REM SPARSE MODELS: p' sweep
    REM -------------------------
    for %%M in (index random mixer) do (
      for %%P in (%P_LIST_MAIN%) do (
        set "OUT=%PHASE1%\%%E\%%D\%%M_pinter_%%P.txt"

        if exist "!OUT!" (
          echo [SKIP] PHASE1 ^| enc=%%E ^| dataset=%%D ^| model=%%M ^| p'=%%P ^| exists: !OUT!
        ) else (
          echo [PHASE1] enc=%%E ^| dataset=%%D ^| model=%%M ^| p'=%%P

          python "%TRAIN%" ^
            --dataset %%D ^
            --model %%M ^
            --p_inter %%P ^
            --epochs %EPOCHS% ^
            --T %T_MAIN% ^
            --batch_size %BS% ^
            --enc %%E ^
            --enc_scale %ENC_SCALE% ^
            --enc_bias %ENC_BIAS% ^
            --sparsity_mode %SPARSITY% ^
            > "!OUT!" 2>&1
        )
      )
    )
  )
)

REM ============================================================
REM PHASE 2 — TIME WINDOW SENSITIVITY
REM T sweep (current injection) + p' sweep
REM - skips runs whose output log already exists
REM ============================================================
echo.
echo ===== PHASE 2: TIME WINDOW SENSITIVITY =====

set "PHASE2=%ROOT%\phase2_time"
if not exist "%PHASE2%" mkdir "%PHASE2%"
if not exist "%PHASE2%\fashionmnist" mkdir "%PHASE2%\fashionmnist"

for %%P in (%P_LIST_T%) do (
  for %%T in (10 20 50) do (
    set "OUT=%PHASE2%\fashionmnist\index_pinter_%%P_T_%%T.txt"

    if exist "!OUT!" (
      echo [SKIP] PHASE2 ^| dataset=fashionmnist ^| model=index ^| p'=%%P ^| T=%%T ^| exists: !OUT!
    ) else (
      echo [PHASE2] dataset=fashionmnist ^| model=index ^| p'=%%P ^| T=%%T

      python "%TRAIN%" ^
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
        > "!OUT!" 2>&1
    )
  )
)

echo.
echo ===== ALL EXPERIMENTS FINISHED =====
endlocal
