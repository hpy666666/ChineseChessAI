@echo off
chcp 65001 >nul
title 中国象棋AI训练系统

:menu
cls
echo ╔════════════════════════════════════════╗
echo ║     中国象棋AI训练系统                 ║
echo ║     基于强化学习 (AlphaZero)          ║
echo ╚════════════════════════════════════════╝
echo.
echo 请选择操作:
echo.
echo [1] 开始训练 (train)
echo [2] 观看对局 (watch)
echo [3] 测试系统 (test)
echo [4] 查看帮助 (help)
echo [5] 安装依赖
echo [0] 退出
echo.
set /p choice=请输入选项 (0-5):

if "%choice%"=="1" goto train
if "%choice%"=="2" goto watch
if "%choice%"=="3" goto test
if "%choice%"=="4" goto help
if "%choice%"=="5" goto install
if "%choice%"=="0" goto end
goto menu

:train
cls
echo 启动训练模式...
echo.
python main.py train
pause
goto menu

:watch
cls
echo 启动观看模式...
echo.
python main.py watch
pause
goto menu

:test
cls
echo 启动测试模式...
echo.
python main.py test
pause
goto menu

:help
cls
python main.py help
pause
goto menu

:install
cls
echo 运行安装脚本...
call install.bat
pause
goto menu

:end
echo 感谢使用！
exit
