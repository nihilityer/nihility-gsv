@echo off
chcp 65001
setlocal

:: ==============================
:: ������
:: ==============================
set "REPO=nihilityer/nihility-gsv"
set "VERSION=v0.2.1"
set "CLI_WIN_ASSET_NAME=nihility-gsv-cli-%VERSION%-x86_64-pc-windows-msvc.exe"
set "CLI_FINAL_EXE_NAME=nihility-gsv-cli.exe"
set "API_WIN_ASSET_NAME=nihility-gsv-api-%VERSION%-x86_64-pc-windows-msvc.exe"
set "API_FINAL_EXE_NAME=nihility-gsv-api.exe"

:: ==============================
:: ��������
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
:: ��ȡ�ű�Ŀ¼
:: ==============================
set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

:: ==============================
:: ����Ŀ¼
:: ==============================
for %%d in ("base" "model\default") do (
    if not exist "%SCRIPT_DIR%\%%~d" mkdir "%SCRIPT_DIR%\%%~d"
)

:: ==============================
:: ���غ���
:: ==============================
goto :main

:download_file
set "URL=%~1"
set "OUT=%~2"
if "%URL%"=="" (
    echo [����] URL Ϊ��
    exit /b 1
)
if "%OUT%"=="" (
    echo [����] ���·��Ϊ��
    exit /b 1
)
if exist "%OUT%" (
    echo [����] �ļ��Ѵ���: %OUT%
    exit /b 0
)
echo [����] %URL%
curl -fL -o "%OUT%" "%URL%" 2>nul
if %errorlevel% equ 0 exit /b 0
echo [����] curl ʧ�ܣ����� PowerShell...
powershell -ExecutionPolicy Bypass -Command "try { (New-Object System.Net.WebClient).DownloadFile('%URL%', '%OUT%') } catch { Write-Error $_; exit 1 }"
if %errorlevel% neq 0 (
    echo [����] ����ʧ��: %URL%
    exit /b 1
)
exit /b 0

:: ==============================
:: ������
:: ==============================
:main
echo ========================================
echo   nihility-gsv Windows ��װ����
echo ========================================
echo.

:: ���� 1: LibTorch
echo [���� 1/3] ���ز����� LibTorch...
set "LIBTORCH_ZIP=%SCRIPT_DIR%\libtorch.zip"
call :download_file "https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.8.0%%%%2Bcpu.zip" "%LIBTORCH_ZIP%"
if errorlevel 1 exit /b 1

set "TEMP_DIR=%TEMP%\libtorch_extract_%RANDOM%"
mkdir "%TEMP_DIR%" 2>nul
if errorlevel 1 (
    echo [����] �޷�������ʱĿ¼
    exit /b 1
)

:: ��ѹ
if exist "%SystemRoot%\system32\tar.exe" (
    tar -xf "%LIBTORCH_ZIP%" -C "%TEMP_DIR%" >nul
) else (
    powershell -ExecutionPolicy Bypass -Command "Expand-Archive -Path '%LIBTORCH_ZIP%' -DestinationPath '%TEMP_DIR%' -Force"
)
if errorlevel 1 (
    echo [����] ��ѹʧ��
    rmdir /s /q "%TEMP_DIR%" 2>nul
    exit /b 1
)

xcopy /s /e /q "%TEMP_DIR%\libtorch\lib\*" "%SCRIPT_DIR%\" >nul
if errorlevel 1 (
    echo [����] ���� lib �ļ�ʧ��
    rmdir /s /q "%TEMP_DIR%" 2>nul
    exit /b 1
)

del /f "%LIBTORCH_ZIP%" 2>nul
rmdir /s /q "%TEMP_DIR%" 2>nul
echo [���] LibTorch �Ѱ�װ
echo.

:: ���� 2: ģ���ļ�
echo [���� 2/3] ����ģ���ļ�...

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
echo [���] ģ���������
echo.

:: ���� 3: Ӧ��
echo [���� 3/3] ����Ӧ��...
set "GH_BASE=https://github.com/%REPO%/releases/download/%VERSION%/"
if %USE_GH_MIRROR%==1 set "GH_BASE=https://ghfast.top/https://github.com/%REPO%/releases/download/%VERSION%/"

set "APP_URL=%GH_BASE%%CLI_WIN_ASSET_NAME%"
set "EXE_PATH=%SCRIPT_DIR%\%CLI_FINAL_EXE_NAME%"

call :download_file "%APP_URL%" "%EXE_PATH%"
if errorlevel 1 (
    echo.
    echo [��������] δ�ҵ� Windows �汾CLIӦ�ã�
    echo ��ȷ�� Release �а���: %CLI_WIN_ASSET_NAME%
    pause
    exit /b 1
)

attrib -R "%EXE_PATH%" 2>nul
echo [���] CliӦ���ѱ���Ϊ %CLI_FINAL_EXE_NAME%
echo.
echo ========================================
echo ��װ�ɹ���˫������ %CLI_FINAL_EXE_NAME% �����������������
echo ========================================


set "APP_URL=%GH_BASE%%API_WIN_ASSET_NAME%"
set "EXE_PATH=%SCRIPT_DIR%\%API_FINAL_EXE_NAME%"

call :download_file "%APP_URL%" "%EXE_PATH%"
if errorlevel 1 (
    echo.
    echo [��������] δ�ҵ� Windows �汾APIӦ�ã�
    echo ��ȷ�� Release �а���: %API_WIN_ASSET_NAME%
    pause
    exit /b 1
)

attrib -R "%EXE_PATH%" 2>nul
echo [���] ApiӦ���ѱ���Ϊ %API_FINAL_EXE_NAME%
echo.
echo ========================================
echo ��װ�ɹ���˫������ %API_FINAL_EXE_NAME% ����Api�������
echo ========================================
pause