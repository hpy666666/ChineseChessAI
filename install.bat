@echo off
echo ========================================
echo 中国象棋AI - 依赖安装脚本
echo ========================================
echo.

echo 检查Python版本...
python --version
echo.

echo 开始安装依赖库...
echo.

echo [1/3] 安装 PyTorch (GPU版本，利用RTX 4070)
echo 这一步可能需要几分钟，请耐心等待...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
echo.

echo [2/3] 安装 NumPy
pip install numpy
echo.

echo [3/3] 安装 Pygame
pip install pygame
echo.

echo ========================================
echo 安装完成！
echo ========================================
echo.

echo 验证安装...
python -c "import torch; print('PyTorch版本:', torch.__version__); print('CUDA可用:', torch.cuda.is_available())"
python -c "import numpy; print('NumPy版本:', numpy.__version__)"
python -c "import pygame; print('Pygame版本:', pygame.__version__)"
echo.

echo 如果上面显示 CUDA可用: True，说明GPU支持正常
echo.

echo 接下来可以运行:
echo   python main.py test    # 测试系统
echo   python main.py train   # 开始训练
echo.

pause
