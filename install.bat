@echo off
setlocal

:: ==============================
:: 配置区
:: ==============================
set "REPO=nihilityer/nihility-gsv"
set "VERSION=v0.2.1"
set "CLI_WIN_ASSET_NAME=nihility-gsv-cli-v0.2.1-x86_64-pc-windows-msvc.exe"
set "CLI_FINAL_EXE_NAME=nihility-gsv-cli.exe"
set "API_WIN_ASSET_NAME=nihility-gsv-api-v0.2.1-x86_64-pc-windows-msvc.exe"
set "API_FINAL_EXE_NAME=nihility-gsv-api.exe"

:: ==============================
:: 参数解析
:: ==============================
set "USE_HF_MIRROR=0"
set "USE_GH_MIRROR=0"

:arg_loop
if "%~1"=="" goto args_done
if /i "%~1"=="-h" set "USE_HF_MIRROR=1"
if /i "%~1"=="--hf-mirror" set "USE_HF_MIRROR=1"
if /i "%~1"=="-g" set "USE_GH_MIRROR=1"
if /i "%~1"=="--gh-mirror" set "USE_GH_MIRROR=1"
shift
goto arg_loop
:args_done

:: ==============================
:: 获取脚本目录
:: ==============================
set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

:: ==============================
:: 创建目录
:: ==============================
for %%d in ("base" "model\default") do (
    if not exist "%SCRIPT_DIR%\%%~d" mkdir "%SCRIPT_DIR%\%%~d"
)

:: ==============================
:: 下载函数
:: ==============================
goto :main

:download_file
set "URL=%~1"
set "OUT=%~2"
if "%URL%"=="" (
    echo [错误] URL 为空
    exit /b 1
)
if "%OUT%"=="" (
    echo [错误] 输出路径为空
    exit /b 1
)
if exist "%OUT%" (
    echo [跳过] 文件已存在: %OUT%
    exit /b 0
)
echo [下载] %URL%
curl -fL -o "%OUT%" "%URL%" 2>nul
if %errorlevel% equ 0 exit /b 0
echo [回退] curl 失败，尝试 PowerShell...
powershell -ExecutionPolicy Bypass -Command "try { (New-Object System.Net.WebClient).DownloadFile('%URL%', '%OUT%') } catch { Write-Error $_; exit 1 }"
if %errorlevel% neq 0 (
    echo [错误] 下载失败: %URL%
    exit /b 1
)
exit /b 0

:: ==============================
:: 主流程
:: ==============================
:main
echo ========================================
echo   nihility-gsv Windows 安装程序
echo ========================================
echo.

:: 步骤 1: LibTorch
echo [步骤 1/3] 下载并处理 LibTorch...
set "LIBTORCH_ZIP=%SCRIPT_DIR%\libtorch.zip"
call :download_file "https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.8.0%%%%2Bcpu.zip" "%LIBTORCH_ZIP%"
if errorlevel 1 exit /b 1

set "TEMP_DIR=%TEMP%\libtorch_extract_%RANDOM%"
mkdir "%TEMP_DIR%" 2>nul
if errorlevel 1 (
    echo [错误] 无法创建临时目录
    exit /b 1
)

:: 解压
if exist "%SystemRoot%\system32\tar.exe" (
    tar -xf "%LIBTORCH_ZIP%" -C "%TEMP_DIR%" >nul
) else (
    powershell -ExecutionPolicy Bypass -Command "Expand-Archive -Path '%LIBTORCH_ZIP%' -DestinationPath '%TEMP_DIR%' -Force"
)
if errorlevel 1 (
    echo [错误] 解压失败
    rmdir /s /q "%TEMP_DIR%" 2>nul
    exit /b 1
)

xcopy /s /e /q "%TEMP_DIR%\libtorch\lib\*" "%SCRIPT_DIR%\" >nul
if errorlevel 1 (
    echo [错误] 复制 lib 文件失败
    rmdir /s /q "%TEMP_DIR%" 2>nul
    exit /b 1
)

del /f "%LIBTORCH_ZIP%" 2>nul
rmdir /s /q "%TEMP_DIR%" 2>nul
echo [完成] LibTorch 已安装
echo.

:: 步骤 2: 模型文件
echo [步骤 2/3] 下载模型文件...

set "HF_BASE=https://huggingface.co/%REPO%/resolve/main/"
if %USE_HF_MIRROR%==1 set "HF_BASE=https://hf-mirror.com/%REPO%/resolve/main/"

for %%f in (base/bert.pt base/g2p-en.pt base/g2p-zh.pt base/ssl.pt) do (
    call :download_file "%HF_BASE%%%f?download=true" "%SCRIPT_DIR%\%%f"
    if errorlevel 1 exit /b 1
)

for %%f in (default/model.pt default/ref.txt default/ref.wav) do (
    call :download_file "%HF_BASE%%%f?download=true" "%SCRIPT_DIR%\model\%%f"
    if errorlevel 1 exit /b 1
)
echo [完成] 模型下载完成
echo.

:: 步骤 3: 应用
echo [步骤 3/3] 下载应用...
set "GH_BASE=https://github.com/%REPO%/releases/download/%VERSION%/"
if %USE_GH_MIRROR%==1 set "GH_BASE=https://ghfast.top/https://github.com/%REPO%/releases/download/%VERSION%/"

set "APP_URL=%GH_BASE%%CLI_WIN_ASSET_NAME%"
set "EXE_PATH=%SCRIPT_DIR%\%CLI_FINAL_EXE_NAME%"

call :download_file "%APP_URL%" "%EXE_PATH%"
if errorlevel 1 (
    echo.
    echo [致命错误] 未找到 Windows 版本CLI应用！
    echo 请确认 Release 中包含: %CLI_WIN_ASSET_NAME%
    pause
    exit /b 1
)

attrib -R "%EXE_PATH%" 2>nul
echo [完成] Cli应用已保存为 %CLI_FINAL_EXE_NAME%
echo.
echo ========================================
echo 安装成功！双击运行 %CLI_FINAL_EXE_NAME% 启动命令行推理程序。
echo ========================================


set "APP_URL=%GH_BASE%%API_WIN_ASSET_NAME%"
set "EXE_PATH=%SCRIPT_DIR%\%API_FINAL_EXE_NAME%"

call :download_file "%APP_URL%" "%EXE_PATH%"
if errorlevel 1 (
    echo.
    echo [致命错误] 未找到 Windows 版本API应用！
    echo 请确认 Release 中包含: %API_WIN_ASSET_NAME%
    pause
    exit /b 1
)

attrib -R "%EXE_PATH%" 2>nul
echo [完成] Api应用已保存为 %API_FINAL_EXE_NAME%
echo.
echo ========================================
echo 安装成功！双击运行 %API_FINAL_EXE_NAME% 启动Api服务程序。
echo ========================================
pause