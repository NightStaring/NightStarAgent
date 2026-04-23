@echo off
setlocal

REM 进入当前脚本所在目录（RAGKnowledge）
cd /d "%~dp0"

REM 切到项目根目录，保证 python -m RAGKnowledge.run 可用
cd /d ".."

echo [RAGKnowledge] Starting...

if not exist ".\logs" mkdir ".\logs"

REM 优先激活 conda 环境
call conda activate agent >nul 2>&1

REM 如果 conda 未生效，则回退到已知环境解释器路径
where python >nul 2>&1
if errorlevel 1 (
    set "PYTHON_EXE=C:\Users\28912\.conda\envs\agent\python.exe"
) else (
    set "PYTHON_EXE=python"
)

powershell -NoProfile -ExecutionPolicy Bypass -Command "$ts = Get-Date -Format 'yyyyMMdd_HHmmss'; & '%PYTHON_EXE%' -m RAGKnowledge.run 2>&1 | Tee-Object -FilePath ('.\logs\run_' + $ts + '.log')"

echo.
echo [RAGKnowledge] Finished. You can close this window manually.
cmd /k

