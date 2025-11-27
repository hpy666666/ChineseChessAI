"""
配置文件 - 所有训练参数都在这里
你可以根据需要调整这些参数
"""

# ============ 训练参数 ============
# 自我对弈参数
SELF_PLAY_GAMES = 100           # 每轮训练前进行多少局自我对弈
MAX_MOVES = 70                  # 每局最多走多少步（降低步数引导快速结束，后期可改回100）
MCTS_SIMULATIONS = 50           # 每步思考模拟次数（越大越聪明但越慢，推荐50-200）

# 动态MCTS模拟次数（根据训练进度自动调整）
def get_dynamic_mcts_simulations(total_games):
    """
    根据训练进度返回合适的MCTS模拟次数

    【优化v4】平衡速度和效果，避免过度模拟
    """
    if total_games < 1000:
        return 30   # 初期：30次模拟，比旧版25次稍强
    elif total_games < 3000:
        return 35   # 早期：35次模拟，平衡速度和质量
    elif total_games < 8000:
        return 60   # 中期：60次模拟，能看到4-5步深度
    elif total_games < 15000:
        return 100  # 中后期：100次模拟，较强战术视野
    else:
        return 150  # 后期：150次模拟，高质量决策

# 动态学习率（根据训练进度自动调整）
def get_dynamic_learning_rate(total_games):
    """
    根据训练进度返回合适的学习率

    原理：
    - 初期：高学习率快速学习基础知识
    - 中期：降低学习率避免震荡
    - 后期：低学习率精细调优
    """
    if total_games < 5000:
        return 0.001   # 初期：快速学习
    elif total_games < 15000:
        return 0.0005  # 中期：稳定提升
    else:
        return 0.0002  # 后期：精细调优

# 多进程配置
NUM_WORKERS = 4                 # 并行对弈的进程数（建议=CPU核心数）
USE_MULTIPROCESSING = True     # 是否使用多进程（True=4倍快但Ctrl+C较慢, False=稳定可随时退出）

# 神经网络训练参数
BATCH_SIZE = 64                 # 每次训练使用多少局数据
EPOCHS = 5                      # 每轮训练遍历数据集次数
LEARNING_RATE = 0.001           # 学习率
BUFFER_SIZE = 10000             # 最多保存多少局历史数据

# 模型保存
SAVE_INTERVAL = 10              # 每训练多少轮保存一次模型
EVALUATE_INTERVAL = 5           # 每多少轮评估一次棋力

# ============ 棋盘设置 ============
BOARD_SIZE = 10                 # 棋盘大小（10x9）
BOARD_WIDTH = 9

# 棋子类型编码
PIECES = {
    'EMPTY': 0,
    # 红方（1-7）
    'R_KING': 1,    'R_ADVISOR': 2, 'R_BISHOP': 3, 'R_KNIGHT': 4,
    'R_ROOK': 5,    'R_CANNON': 6,  'R_PAWN': 7,
    # 黑方（-1到-7）
    'B_KING': -1,   'B_ADVISOR': -2, 'B_BISHOP': -3, 'B_KNIGHT': -4,
    'B_ROOK': -5,   'B_CANNON': -6,  'B_PAWN': -7,
}

# ============ 显示设置 ============
CELL_SIZE = 60                  # 棋盘格子大小（像素）
WINDOW_WIDTH = BOARD_WIDTH * CELL_SIZE + 100
WINDOW_HEIGHT = BOARD_SIZE * CELL_SIZE + 100
FPS = 30                        # 画面刷新率

# 颜色定义（RGB）
COLOR_BG = (220, 179, 92)       # 棋盘背景色
COLOR_LINE = (0, 0, 0)          # 棋盘线条色
COLOR_RED = (200, 0, 0)         # 红方棋子
COLOR_BLACK = (0, 0, 0)         # 黑方棋子
COLOR_HIGHLIGHT = (255, 255, 0) # 高亮选中

# ============ 路径设置 ============
DATA_DIR = "data"
MODEL_DIR = "models"
LOG_DIR = "logs"
LATEST_MODEL = f"{MODEL_DIR}/latest.pt"

# ============ 设备设置 ============
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU型号: {torch.cuda.get_device_name(0)}")
