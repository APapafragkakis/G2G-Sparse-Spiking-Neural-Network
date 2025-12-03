@echo off

set OUTDIR=results_mixer_dst_full

if not exist "%OUTDIR%" (
    mkdir "%OUTDIR%"
)

echo =============================================
echo STARTING FULL DST RUN
echo Date: %date%
echo Time: %time%
echo =============================================

REM Save start time to log file
echo START: %date% %time% > run_times.txt

REM Outer loop: p' values
for %%P in (0.00 0.05 0.10 0.15 0.20 0.25) do (

    REM Middle loop: pruning strategies (cp)
    for %%CP in (set random hebb) do (

        REM Inner loop: growth strategies (cg)
        for %%CG in (random hebb) do (

            echo ------------------------------------------------------------
            echo Running p'=%%P, prune=%%CP, grow=%%CG
            echo Log: %OUTDIR%\mixer_p%%P_cp%%CP_cg%%CG.txt
            echo ------------------------------------------------------------

            python evaluation\train.py ^
                --model mixer ^
                --epochs 20 ^
                --p_inter %%P ^
                --sparsity_mode dynamic ^
                --cp %%CP ^
                --cg %%CG ^
                > "%OUTDIR%\mixer_p%%P_cp%%CP_cg%%CG.txt"

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
