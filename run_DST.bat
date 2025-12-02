@echo off

for %%C in (set random hebb) do (
  for %%G in (hebb random) do (
    echo =================================================
    echo Running CP=%%C, CG=%%G

    python train.py ^
      --model mixer ^
      --sparsity_mode dynamic ^
      --cp %%C ^
      --cg %%G ^
      --epochs 2

    echo.
  )
)
