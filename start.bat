@echo off
cd /d "%~dp0"
for %%V in (3.11 3.10 3.9) do (
  py -%%V --version >nul 2>nul && (py -%%V start.py & goto :done)
)
python start.py
:done
pause
