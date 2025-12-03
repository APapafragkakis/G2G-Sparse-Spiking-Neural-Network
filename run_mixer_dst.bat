@echo off
setlocal enabledelayedexpansion

set OUTDIR=results_mixer_dst_full

if not exist %OUTDIR% (
    mkdir %OUTDIR%
)

echo =============================================
echo STARTING FULL DST RUN
echo Date: %date%
echo Time: %time%
echo =============================================

REM Save start time to log file
echo START: %date% %time% > run_times.txt

REM p' values
set PVALUES=0.00 0.05 0.10 0.15 0.20 0.25

REM pruning strategies
set PRUNE=set random hebb

REM growth strategies
set GROW=random hebb

for %%P in (%PVALUES%) do (
    for %%CP in (%PRUNE%) do (
        for %%CG in (%GROW%) do (

            set FNAME=mixer_p%%P_cp%%CP_cg%%CG.txt

            echo ============================================================
            echo Running p'=%%P%%, prune=%%CP%%, grow=%%CG%%
            echo Log: %OUTDIR%\!FNAME!
            echo ============================================================

            python evaluation/train.py ^
                --model mixer ^
                --epochs 20 ^
                --p_inter %%P ^
                --sparsity_mode dynamic ^
                --cp %%CP ^
                --cg %%CG ^
                > "%OUTDIR%\!FNAME!"

            echo.
        )
    )
)

echo =============================================
echo FINISHED FULL DST RUN
echo Date: %date%
echo Time: %time%
echo =============================================

REM Append end time to log file
echo END: %date% %time% >> run_times.txt

pause
