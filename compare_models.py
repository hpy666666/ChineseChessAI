"""
模型对比工具 - 让两个不同版本的模型对战
用于验证训练是否真的让AI变强了
"""
import torch
import os
import numpy as np
from neural_network import ChessNet
from chess_env import ChineseChess
from self_play import MCTS
from config import DEVICE, MODEL_DIR, LATEST_MODEL

def play_match(network1, network2, num_games=20, verbose=True):
    """
    让两个模型对战

    参数:
        network1: 模型1(执红方)
        network2: 模型2(执黑方)
        num_games: 对战局数
        verbose: 是否显示详细信息

    返回:
        dict: 对战结果统计
    """
    mcts1 = MCTS(network1)
    mcts2 = MCTS(network2)

    model1_wins = 0  # 模型1(红方)获胜
    model2_wins = 0  # 模型2(黑方)获胜
    draws = 0
    total_moves = 0

    for game_num in range(num_games):
        env = ChineseChess()

        if verbose:
            print(f"  对局 {game_num+1}/{num_games}...", end='', flush=True)

        for move_count in range(100):  # 最多100步
            legal_moves = env.get_legal_moves()
            if len(legal_moves) == 0 or env.winner is not None:
                break

            # 根据当前玩家选择MCTS
            current_mcts = mcts1 if env.current_player == 1 else mcts2

            # 搜索
            visit_counts = current_mcts.search(env)

            if len(visit_counts) == 0:
                break

            # 选择最佳走法(使用低温度让对局更确定)
            moves = list(visit_counts.keys())
            counts = np.array(list(visit_counts.values()))
            temperature = 0.3
            counts = counts ** (1.0 / temperature)
            move_probs = counts / counts.sum()
            move_idx = np.random.choice(len(moves), p=move_probs)
            best_move = moves[move_idx]

            # 执行走法
            env.make_move(best_move)

        # 统计结果
        total_moves += env.move_count

        if env.winner == 1:
            model1_wins += 1
            result = "模型1胜"
        elif env.winner == -1:
            model2_wins += 1
            result = "模型2胜"
        else:
            draws += 1
            result = "和局"

        if verbose:
            print(f" {result} ({env.move_count}步)")

    avg_moves = total_moves / num_games

    return {
        'model1_wins': model1_wins,
        'model2_wins': model2_wins,
        'draws': draws,
        'avg_moves': avg_moves,
        'model1_winrate': model1_wins / num_games * 100,
        'model2_winrate': model2_wins / num_games * 100,
        'draw_rate': draws / num_games * 100
    }


def compare_two_models(model1_path, model2_path, num_games=20):
    """
    对比两个模型

    参数:
        model1_path: 模型1路径
        model2_path: 模型2路径
        num_games: 对战局数(会对战两轮,双方各执红一次)
    """
    print("=" * 60)
    print("模型对比系统".center(60))
    print("=" * 60)

    # 加载模型1
    print(f"\n加载模型1: {model1_path}")
    if not os.path.exists(model1_path):
        print(f"[错误] 模型文件不存在!")
        return

    network1 = ChessNet().to(DEVICE)
    checkpoint1 = torch.load(model1_path, map_location=DEVICE)
    network1.load_state_dict(checkpoint1['model_state_dict'])
    network1.eval()

    games1 = checkpoint1.get('total_games', 0)
    print(f"[OK] 模型1已训练 {games1} 局")

    # 加载模型2
    print(f"\n加载模型2: {model2_path}")
    if not os.path.exists(model2_path):
        print(f"[错误] 模型文件不存在!")
        return

    network2 = ChessNet().to(DEVICE)
    checkpoint2 = torch.load(model2_path, map_location=DEVICE)
    network2.load_state_dict(checkpoint2['model_state_dict'])
    network2.eval()

    games2 = checkpoint2.get('total_games', 0)
    print(f"[OK] 模型2已训练 {games2} 局")

    # 第一轮对战: 模型1执红，模型2执黑
    print(f"\n第一轮对战 (模型1执红, 模型2执黑, {num_games}局):")
    print("-" * 60)
    round1 = play_match(network1, network2, num_games=num_games, verbose=True)

    # 第二轮对战: 模型2执红，模型1执黑
    print(f"\n第二轮对战 (模型2执红, 模型1执黑, {num_games}局):")
    print("-" * 60)
    round2 = play_match(network2, network1, num_games=num_games, verbose=True)

    # 汇总统计
    print("\n" + "=" * 60)
    print("对比结果汇总".center(60))
    print("=" * 60)

    # 模型1总战绩: 第一轮红方获胜 + 第二轮黑方获胜
    model1_total_wins = round1['model1_wins'] + round2['model2_wins']
    model2_total_wins = round1['model2_wins'] + round2['model1_wins']
    total_draws = round1['draws'] + round2['draws']
    total_games = num_games * 2

    model1_total_winrate = model1_total_wins / total_games * 100
    model2_total_winrate = model2_total_wins / total_games * 100

    print(f"\n模型1 (训练{games1}局):")
    print(f"  总战绩: {model1_total_wins}胜 {model2_total_wins}负 {total_draws}和")
    print(f"  胜率: {model1_total_winrate:.1f}%")
    print(f"  执红: {round1['model1_wins']}胜 / 执黑: {round2['model2_wins']}胜")

    print(f"\n模型2 (训练{games2}局):")
    print(f"  总战绩: {model2_total_wins}胜 {model1_total_wins}负 {total_draws}和")
    print(f"  胜率: {model2_total_winrate:.1f}%")
    print(f"  执红: {round2['model1_wins']}胜 / 执黑: {round1['model2_wins']}胜")

    # 判断结论
    print(f"\n对比结论:")
    if model1_total_winrate > model2_total_winrate + 10:
        print(f"  >>> 模型1明显更强 (胜率高{model1_total_winrate - model2_total_winrate:.1f}%)")
    elif model2_total_winrate > model1_total_winrate + 10:
        print(f"  >>> 模型2明显更强 (胜率高{model2_total_winrate - model1_total_winrate:.1f}%)")
    elif abs(model1_total_winrate - model2_total_winrate) <= 10:
        print(f"  >>> 两个模型实力接近")

    # 判断训练是否有效
    if games2 > games1:
        newer_model = "模型2"
        newer_winrate = model2_total_winrate
    else:
        newer_model = "模型1"
        newer_winrate = model1_total_winrate

    if newer_winrate > 55:
        print(f"  ✓ {newer_model}(更新版本)表现更好，训练有效！")
    elif newer_winrate < 45:
        print(f"  ✗ {newer_model}(更新版本)表现更差，可能过拟合或配置问题")
    else:
        print(f"  - 两个模型水平接近，继续训练可能会拉开差距")

    print("\n" + "=" * 60)


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='对比两个象棋AI模型')
    parser.add_argument('--model1', type=str, default=LATEST_MODEL,
                      help='模型1路径 (默认: latest.pt)')
    parser.add_argument('--model2', type=str, required=False,
                      help='模型2路径 (必需)')
    parser.add_argument('--games', type=int, default=10,
                      help='每轮对战局数 (总共2轮, 默认: 10)')

    args = parser.parse_args()

    if not args.model2:
        print("错误: 需要指定第二个模型路径")
        print("\n示例用法:")
        print("  python compare_models.py --model2 models/model_1000.pt")
        print("  python compare_models.py --model1 models/model_5000.pt --model2 models/model_10000.pt --games 20")
        print("\n可用的模型文件:")

        if os.path.exists(MODEL_DIR):
            models = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pt')]
            if models:
                for model in sorted(models):
                    full_path = os.path.join(MODEL_DIR, model)
                    try:
                        checkpoint = torch.load(full_path, map_location='cpu')
                        games = checkpoint.get('total_games', '?')
                        print(f"  - {model} (训练{games}局)")
                    except:
                        print(f"  - {model}")
            else:
                print("  (暂无备份模型)")
        return

    compare_two_models(args.model1, args.model2, num_games=args.games)


if __name__ == "__main__":
    main()
