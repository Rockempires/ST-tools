@echo off
rem 强制更新 ST-tools：所有插件文件以网站版本为准，本地同名文件全部覆盖
cd /d "%~dp0"

git rev-parse --git-dir >nul 2>&1
if errorlevel 1 (
  echo This folder is not a git install.
  echo Please uninstall and reinstall ST-tools in ComfyUI Manager.
  pause
  exit /b 1
)

echo Force updating ST-tools, local changes to plugin files will be overwritten...
git fetch origin
if errorlevel 1 (
  echo Fetch failed, please check your network or proxy.
  pause
  exit /b 1
)
git reset --hard origin/main
echo.
echo Done.
pause
