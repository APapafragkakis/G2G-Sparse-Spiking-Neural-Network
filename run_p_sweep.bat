@echo off
setlocal ENABLEDELAYEDEXPANSION

REM Folder for logs
set OUTDIR=results_p_sweep
if not exist %OUTDIR% (
    mkdir %OUTDIR%
)

echo Running p' sweep for Index, Mixer and Random with 20 epochs...
echo Output folder: %OUTDIR%
echo.

REM Models to evaluate
set MODELS=index mixer random

REM p' values (same range as in G2GNet)
set PVALUES=0.00 0.05 0.10 0.15 0.20 0.25

for %%M in (%MODELS%) do (
    for %%P in (%PVALUES%) do (
        echo ===========================================
        echo Model=%%M, p_inter=%%P
        echo ===========================================
        python evaluation\train.py --model %%M --p_inter %%P --epochs 20 > %OUTDIR%\%%M_p%%P.txt
    )
)

echo -------------------------------------------
echo Running Dense baseline with 20 epochs...
python evaluation\train.py --model dense --epochs 20 > %OUTDIR%\dense.txt

echo.
echo Done! All logs saved in %OUTDIR%.
endlocal
