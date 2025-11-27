"""
性能测试 - 检查规则检查是否太慢
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import time
from chess_env import ChineseChess
from neural_network import ChessNet
from self_play import MCTS
import torch

def test_single_game_speed():
    """测试单局对弈速度"""
    print("=" * 60)
    print("性能测试 - 单局对弈速度")
    print("=" * 60)

    network = ChessNet().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    network.eval()

    env = ChineseChess()
    mcts = MCTS(network)

    print("\n开始测试...")
    start_time = time.time()

    move_count = 0
    max_moves = 20  # 只测试前20步

    for i in range(max_moves):
        legal_moves = env.get_legal_moves()

        if len(legal_moves) == 0:
            print(f"  第{i+1}步: 无合法走法,游戏结束")
            break

        step_start = time.time()

        # MCTS搜索
        visit_counts = mcts.search(env)

        if len(visit_counts) == 0:
            print(f"  第{i+1}步: MCTS无结果")
            break

        step_time = time.time() - step_start

        # 选择最佳走法
        best_move = max(visit_counts, key=visit_counts.get)

        # 执行走法
        state, reward, done = env.make_move(best_move)

        move_count += 1

        print(f"  第{i+1}步: {step_time:.2f}秒 | 合法走法:{len(legal_moves)} | 奖励:{reward:.2f}")

        if done:
            print(f"  游戏结束: 胜者={env.winner}")
            break

    total_time = time.time() - start_time

    print(f"\n总结:")
    print(f"  完成步数: {move_count}")
    print(f"  总耗时: {total_time:.2f}秒")
    print(f"  平均每步: {total_time/move_count:.2f}秒")

    if total_time / move_count > 10:
        print(f"\n⚠️  警告: 每步超过10秒,训练会非常慢!")
        print(f"  可能原因:")
        print(f"    1. MCTS模拟次数太多({import_config().MCTS_SIMULATIONS})")
        print(f"    2. 规则检查太慢(对脸/送将检测)")
        print(f"    3. GPU未正确使用")
    elif total_time / move_count > 3:
        print(f"\n[!] 每步3-10秒,训练较慢")
    else:
        print(f"\n[OK] 速度正常")

def test_legal_moves_speed():
    """测试合法走法生成速度"""
    print("\n" + "=" * 60)
    print("性能测试 - 合法走法生成速度")
    print("=" * 60)

    env = ChineseChess()

    print("\n测试初始局面...")
    start = time.time()

    for i in range(100):
        legal_moves = env.get_legal_moves()

    elapsed = time.time() - start

    print(f"  100次调用耗时: {elapsed:.3f}秒")
    print(f"  平均每次: {elapsed/100*1000:.1f}毫秒")
    print(f"  初始局面合法走法数: {len(legal_moves)}")

    if elapsed / 100 > 0.1:
        print(f"\n[!] 警告: 合法走法生成太慢(>{elapsed/100*1000:.0f}ms)")
        print(f"  可能是对脸/送将检测导致的")
    else:
        print(f"\n[OK] 合法走法生成速度正常")

def import_config():
    """动态导入config避免循环"""
    import config
    return config

if __name__ == "__main__":
    test_legal_moves_speed()
    print("\n" + "=" * 60)
    test_single_game_speed()
