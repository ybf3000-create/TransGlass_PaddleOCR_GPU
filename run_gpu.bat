@echo off
chcp 65001 >nul
REM TransGlass PaddleOCR GPU Launcher
echo ================================================
echo    TransGlass PaddleOCR - GPU Version Launcher
echo ================================================
echo.

REM Get script directory
set SCRIPT_DIR=%~dp0
REM Remove trailing backslash
set SCRIPT_DIR=%SCRIPT_DIR:~0,-1%

echo [INFO] Script directory: %SCRIPT_DIR%
echo.

REM Check if virtual environment exists
if not exist "%SCRIPT_DIR%\venv_gpu\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found!
    echo [ERROR] Please make sure venv_gpu folder exists in:
    echo         %SCRIPT_DIR%\venv_gpu\
    pause
    exit /b 1
)

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call "%SCRIPT_DIR%\venv_gpu\Scripts\activate.bat"
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment!
    pause
    exit /b 1
)
echo [INFO] Virtual environment activated successfully.
echo.

REM Set environment variables
echo [INFO] Setting environment variables...
set PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
set CUDA_VISIBLE_DEVICES=0

REM Add all necessary paths to PATH
echo [INFO] Setting up library paths...
set "CUDNN_PATH=%SCRIPT_DIR%\venv_gpu\Lib\site-packages\nvidia\cudnn\bin"
set "CUDA_RUNTIME_PATH=%SCRIPT_DIR%\venv_gpu\Lib\site-packages\nvidia\cuda_runtime\bin"
set "SCRIPTS_PATH=%SCRIPT_DIR%\venv_gpu\Scripts"

REM Add to PATH if they exist
if exist "%CUDNN_PATH%" (
    set "PATH=%CUDNN_PATH%;%PATH%"
    echo [INFO] Added CUDNN to PATH: %CUDNN_PATH%
) else (
    echo [WARNING] CUDNN path not found: %CUDNN_PATH%
)

if exist "%CUDA_RUNTIME_PATH%" (
    set "PATH=%CUDA_RUNTIME_PATH%;%PATH%"
    echo [INFO] Added CUDA Runtime to PATH: %CUDA_RUNTIME_PATH%
) else (
    echo [WARNING] CUDA Runtime path not found: %CUDA_RUNTIME_PATH%
)

set "PATH=%SCRIPTS_PATH%;%PATH%"
echo [INFO] Added Scripts to PATH: %SCRIPTS_PATH%
echo.

REM Verify Python is from virtual environment
echo [INFO] Verifying Python environment...
where python
python --version
echo.

REM Check GPU availability
echo [INFO] Checking GPU availability...
python -c "import paddle; print('PaddlePaddle version:', paddle.__version__); print('CUDA available:', paddle.device.is_compiled_with_cuda()); print('Current device:', paddle.device.get_device())"
if errorlevel 1 (
    echo [ERROR] Failed to verify GPU! Please check PaddlePaddle installation.
    pause
    exit /b 1
)
echo.

REM Start the program
echo [INFO] Starting TransGlass PaddleOCR (GPU version)...
echo [INFO] Using GPU acceleration for faster OCR!
echo [INFO] First run will download models (about 100MB)...
echo ================================================
echo.

REM Run program with full path to ensure using virtual environment Python
"%SCRIPT_DIR%\venv_gpu\Scripts\python.exe" "%SCRIPT_DIR%\TransGlass_PaddleOCR_GPU.py"

REM Check exit code
if errorlevel 1 (
    echo.
    echo ================================================
    echo [ERROR] Program exited abnormally with error code: %errorlevel%
    echo [ERROR] Please check the error messages above.
    echo ================================================
    pause
    exit /b %errorlevel%
) else (
    echo.
    echo ================================================
    echo [INFO] Program exited normally.
    echo ================================================
)
