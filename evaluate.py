"""
快速评估脚本 - 测试当前模型的实力
不需要等待完整训练轮，可以随时评估模型
"""
import torch
import os
import numpy as np
from datetime import datetime
from neural_network import ChessNet
from self_play import self_play_game
from config import DEVICE, LATEST_MODEL

def evaluate_model(model_path=LATEST_MODEL, num_games=10, verbose=True):
    """
    评估模型实力

    参数:
        model_path: 模型文件路径
        num_games: 评估对局数（推荐10-20局）
        verbose: 是否显示详细信息

    返回:
        dict: 评估结果统计
    """
    if verbose:
        print("=" * 60)
        print("快速评估模型实力".center(60))
        print("=" * 60)

    # 加载模型
    if not os.path.exists(model_path):
        print(f"\n[错误] 模型文件不存在: {model_path}")
        print("请先运行 'python main.py train' 进行训练")
        return None

    network = ChessNet().to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    network.load_state_dict(checkpoint['model_state_dict'])
    network.eval()

    total_games_trained = checkpoint.get('total_games', 0)
    training_steps = checkpoint.get('training_steps', 0)

    if verbose:
        print(f"\n模型信息:")
        print(f"  - 文件: {model_path}")
        print(f"  - 已训练局数: {total_games_trained}")
        print(f"  - 训练步数: {training_steps}")
        print(f"  - 设备: {DEVICE}")

    # 评估对局
    if verbose:
        print(f"\n开始评估 ({num_games}局)...")
        print("-" * 60)

    red_wins = 0
    black_wins = 0
    draws = 0
    total_moves = 0
    move_counts = []

    for i in range(num_games):
        if verbose:
            print(f"  对局 {i+1}/{num_games}...", end='', flush=True)

        # 使用低温度（0.1）让AI更确定性地走棋
        game_data, winner = self_play_game(network, temperature=0.1, render=False)

        moves = len(game_data)
        total_moves += moves
        move_counts.append(moves)

        if winner == 1:
            red_wins += 1
            result = "红胜"
        elif winner == -1:
            black_wins += 1
            result = "黑胜"
        else:
            draws += 1
            result = "和局"

        if verbose:
            print(f" {result} ({moves}步)")

    # 统计分析
    avg_moves = total_moves / num_games
    min_moves = min(move_counts)
    max_moves = max(move_counts)

    red_rate = red_wins / num_games * 100
    black_rate = black_wins / num_games * 100
    draw_rate = draws / num_games * 100

    # 实力等级评估
    skill_level = estimate_skill_level(total_games_trained, avg_moves, draw_rate, red_rate, black_rate)

    if verbose:
        print("-" * 60)
        print("\n评估结果:")
        print("=" * 60)
        print(f"\n对局统计:")
        print(f"  红方获胜: {red_wins}/{num_games} ({red_rate:.1f}%)")
        print(f"  黑方获胜: {black_wins}/{num_games} ({black_rate:.1f}%)")
        print(f"  和局:     {draws}/{num_games} ({draw_rate:.1f}%)")

        print(f"\n步数统计:")
        print(f"  平均步数: {avg_moves:.1f}")
        print(f"  最少步数: {min_moves}")
        print(f"  最多步数: {max_moves}")

        print(f"\n预估实力等级: {skill_level}")
        print("\n" + "=" * 60)

    # 返回结果
    result = {
        'total_games_trained': total_games_trained,
        'red_wins': red_wins,
        'black_wins': black_wins,
        'draws': draws,
        'avg_moves': avg_moves,
        'min_moves': min_moves,
        'max_moves': max_moves,
        'skill_level': skill_level,
        'red_rate': red_rate,
        'black_rate': black_rate,
        'draw_rate': draw_rate
    }

    return result


def estimate_skill_level(games_trained, avg_moves, draw_rate, red_rate, black_rate=0):
    """
    根据训练局数和表现估算实力等级

    返回: str 实力等级描述
    """
    # 基于训练局数的基础等级
    if games_trained < 100:
        base_level = "完全随机"
    elif games_trained < 500:
        base_level = "初识规则"
    elif games_trained < 1000:
        base_level = "入门级"
    elif games_trained < 5000:
        base_level = "业余初级"
    elif games_trained < 10000:
        base_level = "业余初级+"
    elif games_trained < 20000:
        base_level = "业余中级"
    elif games_trained < 50000:
        base_level = "业余高级"
    elif games_trained < 100000:
        base_level = "专业入门"
    else:
        base_level = "专业水平"

    # 基于表现的修正
    indicators = []

    # 和局率低于90%说明开始有胜负
    if draw_rate < 90:
        indicators.append("已有胜负分化")

    # 红方胜率显著高于黑方说明理解先手优势
    if red_rate > black_rate + 10:
        indicators.append("理解先手优势")

    # 平均步数少于80步说明知道进攻
    if avg_moves < 80:
        indicators.append("懂得进攻")

    # 平均步数大于80步说明还在和棋
    if avg_moves > 90:
        indicators.append("倾向和棋")

    if indicators:
        return f"{base_level} ({', '.join(indicators)})"
    else:
        return base_level


def compare_with_history(current_result):
    """
    与历史评估结果对比（如果有的话）
    """
    history_file = "logs/evaluation_history.txt"

    if not os.path.exists(history_file):
        # 创建历史记录
        os.makedirs("logs", exist_ok=True)
        with open(history_file, "w", encoding="utf-8") as f:
            f.write("评估历史记录\n")
            f.write("=" * 60 + "\n\n")

    # 添加当前记录
    with open(history_file, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"训练局数: {current_result['total_games_trained']}\n")
        f.write(f"红胜率: {current_result['red_rate']:.1f}% | "
                f"黑胜率: {current_result['black_rate']:.1f}% | "
                f"和局率: {current_result['draw_rate']:.1f}%\n")
        f.write(f"平均步数: {current_result['avg_moves']:.1f}\n")
        f.write(f"实力等级: {current_result['skill_level']}\n")
        f.write("-" * 60 + "\n\n")

    print(f"评估记录已保存到: {history_file}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='快速评估象棋AI模型')
    parser.add_argument('--games', type=int, default=10,
                      help='评估对局数 (默认: 10)')
    parser.add_argument('--model', type=str, default=LATEST_MODEL,
                      help='模型文件路径')

    args = parser.parse_args()

    result = evaluate_model(
        model_path=args.model,
        num_games=args.games,
        verbose=True
    )

    if result:
        compare_with_history(result)

        print("\n提示:")
        print("  - 可以运行 'python main.py watch' 观看AI对局")
        print("  - 可以运行 'python main.py play' 与AI对战")
        print("  - 继续训练: 'python main.py train'")


if __name__ == "__main__":
    main()
