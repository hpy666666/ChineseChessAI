"""
主程序 - 中国象棋AI训练系统
使用方法:
    python main.py train     # 开始训练
    python main.py watch     # 观看AI对局
    python main.py test      # 测试各模块
"""
import sys
import os

def train():
    """训练模式"""
    from trainer import Trainer

    print("\n" + "="*60)
    print("  中国象棋AI训练系统")
    print("  基于强化学习 (AlphaZero简化版)")
    print("="*60)

    trainer = Trainer()

    print("\n训练提示:")
    print("- 训练会自动保存到 models/latest.pt")
    print("- 可以随时按 Ctrl+C 停止训练")
    print("- 重新运行会自动加载上次的模型继续训练")
    print("- 日志保存在 logs/training.log")
    print("\n开始训练...\n")

    try:
        # 训练100轮（可以根据需要调整）
        trainer.train_loop(num_iterations=100)
    except KeyboardInterrupt:
        print("\n\n训练被用户中断")
        # 模型在trainer.py中已经保存，这里不需要重复保存
        print("训练已停止")

def play():
    """人机对战模式"""
    import pygame
    from visualizer import GameVisualizer
    from neural_network import ChessNet
    from config import DEVICE, LATEST_MODEL
    import torch

    print("\n" + "="*60)
    print("  人机对战模式")
    print("="*60)

    print("\n加载模型...")

    network = ChessNet().to(DEVICE)

    if os.path.exists(LATEST_MODEL):
        checkpoint = torch.load(LATEST_MODEL, map_location=DEVICE)
        network.load_state_dict(checkpoint['model_state_dict'])
        total_games = checkpoint.get('total_games', 0)
        print(f"[OK] 已加载模型: {LATEST_MODEL}")
        print(f"[OK] 该模型已训练 {total_games} 局")

        # 根据训练局数估算AI实力
        if total_games < 500:
            skill = "新手(完全随机)"
        elif total_games < 2000:
            skill = "入门级"
        elif total_games < 5000:
            skill = "业余初级"
        elif total_games < 10000:
            skill = "业余中级"
        else:
            skill = "业余高级+"

        print(f"[OK] 预估AI实力: {skill}")
    else:
        print("[!] 未找到训练好的模型，使用随机初始化网络")
        print("  AI将完全随机走棋(建议先训练)")
        total_games = 0

    network.eval()

    # 选择执哪方
    print("\n请选择您要执哪方:")
    print("  1. 红方(先手)")
    print("  2. 黑方(后手)")

    choice = input("请输入 1 或 2 (默认1): ").strip()
    human_color = 'black' if choice == '2' else 'red'

    print(f"\n您选择执: {'红方' if human_color == 'red' else '黑方'}")
    print("\n启动图形界面...")
    print("控制说明:")
    print("  - 鼠标点击选择您的棋子")
    print("  - 再次点击目标位置移动棋子")
    print("  - 绿色圆圈表示可移动的位置")
    print("  - Q键退出对局")
    print("\n祝您下棋愉快！")

    visualizer = GameVisualizer()

    # 开始人机对战
    visualizer.human_vs_ai(network, human_color=human_color)

    pygame.quit()
    print("\n对战结束")


def watch():
    """观看模式"""
    import pygame
    from visualizer import GameVisualizer
    from neural_network import ChessNet
    from config import DEVICE, LATEST_MODEL
    import torch

    print("\n加载模型...")

    network = ChessNet().to(DEVICE)

    if os.path.exists(LATEST_MODEL):
        checkpoint = torch.load(LATEST_MODEL, map_location=DEVICE)
        network.load_state_dict(checkpoint['model_state_dict'])
        total_games = checkpoint.get('total_games', 0)
        training_steps = checkpoint.get('training_steps', 0)
        print(f"[OK] 已加载模型: {LATEST_MODEL}")
        print(f"[OK] 该模型已训练 {total_games} 局")
        print(f"[OK] 训练步数: {training_steps}")

        # 估算实力等级
        if total_games < 100:
            skill = "完全随机"
        elif total_games < 500:
            skill = "初识规则"
        elif total_games < 1000:
            skill = "入门级"
        elif total_games < 5000:
            skill = "业余初级"
        elif total_games < 10000:
            skill = "业余中级"
        elif total_games < 50000:
            skill = "业余高级"
        else:
            skill = "专业水平"

        print(f"[OK] 预估实力: {skill}")
    else:
        print("[!] 未找到训练好的模型，使用随机初始化网络")
        print("  建议先运行 'python main.py train' 进行训练")

    network.eval()

    print("\n启动图形界面...")
    print("- 窗口会显示AI自我对弈过程")
    print("- 关闭窗口即可退出")

    visualizer = GameVisualizer()

    # 观看5局对局
    visualizer.watch_game(network, num_games=5)

    pygame.quit()
    print("\n观看结束")

def test():
    """测试模式 - 测试各个模块"""
    print("\n=== 测试各模块 ===\n")

    # 测试1: 棋盘环境
    print("1. 测试棋盘环境...")
    from chess_env import ChineseChess
    env = ChineseChess()
    env.reset()
    legal_moves = env.get_legal_moves()
    print(f"   [OK] 初始局面合法走法数: {len(legal_moves)}")
    env.render()

    # 测试2: 神经网络
    print("\n2. 测试神经网络...")
    from neural_network import test_network
    test_network()

    # 测试3: 自我对弈
    print("\n3. 测试自我对弈...")
    from self_play import test_self_play
    test_self_play()

    print("\n=== 所有测试完成 ===")

def evaluate():
    """快速评估模式"""
    print("\n启动快速评估...")
    import evaluate as eval_module
    # 清理sys.argv，避免argparse解析错误
    original_argv = sys.argv.copy()
    sys.argv = [sys.argv[0]]  # 只保留脚本名
    try:
        eval_module.main()
    finally:
        sys.argv = original_argv  # 恢复原始参数


def show_help():
    """显示帮助"""
    help_text = """
中国象棋AI训练系统 - 使用说明

命令:
    python main.py train       开始训练AI（会自动保存和加载模型）
    python main.py play        与AI对战（人机对战模式）
    python main.py watch       观看AI自我对局
    python main.py evaluate    快速评估当前模型实力
    python main.py test        测试各模块是否正常工作
    python main.py help        显示此帮助信息

训练流程:
    1. 首次运行: python main.py train
       - AI会从零开始学习
       - 前期完全随机走棋
       - 模型自动保存到 models/latest.pt

    2. 评估进度: python main.py evaluate
       - 快速测试当前模型实力
       - 显示训练局数和预估等级
       - 无需等待完整训练轮

    3. 人机对战: python main.py play
       - 与AI对战验收成果
       - 鼠标点击操作
       - 可选红方或黑方

    4. 观看对局: python main.py watch
       - 图形界面显示AI对局
       - 可以看到AI的走棋过程
       - 支持速度调节

    5. 继续训练: 再次运行 python main.py train
       - 会自动加载上次的模型
       - 继续训练提升棋力

配置文件:
    config.py - 可以调整训练参数
    - MCTS_SIMULATIONS: 每步思考次数（影响棋力和速度）
    - SELF_PLAY_GAMES: 每轮训练对局数
    - LEARNING_RATE: 学习率

文件说明:
    chess_env.py       - 象棋规则引擎
    neural_network.py  - AI大脑（神经网络）
    self_play.py       - 自我对弈系统
    trainer.py         - 训练管理器
    visualizer.py      - 图形界面
    config.py          - 配置参数

数据存储:
    data/              - 对局数据（自动管理）
    models/            - AI模型（latest.pt是最新版本）
    logs/              - 训练日志

预期进步时间线:
    - 100局后: 学会基本走法规则
    - 500局后: 知道吃子
    - 2000局后: 有简单战术意识
    - 5000局后: 业余初级水平
    - 10000局后: 能完成完整对局

注意:
    - 首次运行需要安装依赖: pip install torch pygame numpy
    - 有显卡(RTX 4070)训练速度快很多
    - 可以随时按Ctrl+C停止，模型会自动保存
"""
    print(help_text)

def main():
    if len(sys.argv) < 2:
        print("错误: 缺少命令参数")
        print("使用 'python main.py help' 查看帮助")
        return

    command = sys.argv[1].lower()

    if command == "train":
        train()
    elif command == "play":
        play()
    elif command == "watch":
        watch()
    elif command == "evaluate" or command == "eval":
        evaluate()
    elif command == "test":
        test()
    elif command == "help":
        show_help()
    else:
        print(f"错误: 未知命令 '{command}'")
        print("使用 'python main.py help' 查看帮助")

if __name__ == "__main__":
    import os
    import multiprocessing

    # 禁用Intel MKL的Ctrl+C处理（解决Windows多进程forrtl错误）
    os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"

    multiprocessing.freeze_support()  # Windows多进程必需
    main()
