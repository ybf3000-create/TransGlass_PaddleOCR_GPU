@echo off
chcp 65001 >nul
set SCRIPT_DIR=%~dp0
set SCRIPT_DIR=%SCRIPT_DIR:~0,-1%
call "%SCRIPT_DIR%\venv_gpu\Scripts\activate.bat"
set PATH=%SCRIPT_DIR%\venv_gpu\Lib\site-packages\nvidia\cudnn\bin;%PATH%
"%SCRIPT_DIR%\venv_gpu\Scripts\python.exe" "%SCRIPT_DIR%\test_ocr_filter.py"
pause
