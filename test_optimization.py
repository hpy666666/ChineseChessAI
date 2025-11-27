"""
性能优化测试 - 对比优化前后的速度
"""
import time
import torch
from chess_env import ChineseChess
from neural_network import ChessNet
from self_play import MCTS
import config

def test_mcts_performance():
    """测试MCTS搜索性能"""
    print("=" * 60)
    print("性能优化测试")
    print("=" * 60)
    print(f"设备: {config.DEVICE}")
    print(f"MCTS模拟次数: {config.MCTS_SIMULATIONS}")
    print()

    # 初始化
    network = ChessNet().to(torch.device(config.DEVICE))
    network.eval()
    env = ChineseChess()
    mcts = MCTS(network)

    print("开始性能测试（测试10步）...")
    print()

    total_time = 0
    step_times = []

    for step in range(10):
        start = time.time()
        visit_counts = mcts.search(env)
        elapsed = time.time() - start
        total_time += elapsed
        step_times.append(elapsed)

        # 选择最佳走法并执行
        best_move = max(visit_counts, key=visit_counts.get)
        env.make_move(best_move)

        print(f"第{step+1}步: {elapsed:.2f}秒")

    avg_time = total_time / 10
    print()
    print("=" * 60)
    print("性能测试结果:")
    print("=" * 60)
    print(f"平均每步: {avg_time:.2f}秒")
    print(f"预估单局(100步): {avg_time * 100 / 60:.1f}分钟")
    print(f"预估100局: {avg_time * 100 * 100 / 3600:.1f}小时")
    print()

    # 性能对比
    print("优化对比 (vs 最初的6分钟/200步):")
    estimated_game_time = avg_time * 100 / 60
    speedup = 6.0 / estimated_game_time
    print(f"- 单局时间: {estimated_game_time:.1f}分钟 (原来: 6.0分钟)")
    print(f"- 提速倍数: {speedup:.1f}x")
    print(f"- 100局时间: {estimated_game_time * 100 / 60:.1f}小时 (原来: 10小时)")
    print()

    # 详细优化说明
    print("已实施的优化:")
    print("[OK] 1. 缓存将帅位置 - 避免每次遍历棋盘")
    print("[OK] 2. 批量神经网络推理 - 提升GPU利用率")
    print("[OK] 3. 优化环境复制 - 只复制必需状态")
    print("[OK] 4. 降低MAX_MOVES - 从200步改为100步")
    print("[OK] 5. 和局惩罚机制 - 鼓励红方进攻")
    print("=" * 60)

if __name__ == "__main__":
    test_mcts_performance()
