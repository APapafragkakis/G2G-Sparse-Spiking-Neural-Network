@echo off
setlocal ENABLEDELAYEDEXPANSION

REM Output folder
set OUTDIR=results_cv_5fold_5ep_p_sweep

if not exist "%OUTDIR%" (
    mkdir "%OUTDIR%"
)

echo Running 5-fold cross-validation (5 epochs) with p' sweep for Index, Mixer and Random...
echo Output folder: %OUTDIR%
echo.

REM Models list
set MODELS=index mixer random

REM p' values
set PVALUES=0.00 0.05 0.10 0.15 0.20 0.25

for %%M in (%MODELS%) do (
    for %%P in (%PVALUES%) do (
        echo Model=%%M, p_inter=%%P
        echo 5-fold CV, 5 epochs
        python evaluation/crossval.py --model %%M --p_inter %%P --epochs 5 --k_folds 5 > "%OUTDIR%/%%M_p%%P.txt"
    )
)

echo Running Dense baseline with 5-fold CV, 5 epochs...
python evaluation/crossval.py --model dense --epochs 5 --k_folds 5 > "%OUTDIR%/dense.txt"

echo.
echo Done! All cross-validation logs saved in %OUTDIR%

endlocal
