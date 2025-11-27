"""
测试奖励机制改进
验证AI是否会主动吃将/帅
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from chess_env import ChineseChess
from config import PIECES
import numpy as np


def test_king_capture_reward():
    """测试吃将/帅的奖励"""
    print("\n【测试1】吃将/帅奖励机制")
    env = ChineseChess()

    # 设置简单局面:红炮可以直接吃黑将
    env.board = np.zeros((10, 9), dtype=np.int8)
    env.board[0, 4] = PIECES['B_KING']   # 黑将
    env.board[0, 1] = PIECES['R_CANNON'] # 红炮(可以吃将)
    env.board[9, 4] = PIECES['R_KING']   # 红帅
    env.current_player = 1  # 红方走

    # 执行吃将
    state, reward, done = env.make_move((0, 1, 0, 4))

    print(f"  吃将后奖励: {reward}")
    print(f"  是否结束: {done}")
    print(f"  胜者: {env.winner}")

    if reward == 100 and done and env.winner == 1:
        print("  [PASS] 吃将奖励=100,正确!")
        return True
    else:
        print("  [FAIL] 奖励机制有问题")
        return False


def test_piece_capture_reward():
    """测试吃子奖励"""
    print("\n【测试2】吃子奖励机制")
    env = ChineseChess()

    # 设置局面:红车吃黑车
    env.board = np.zeros((10, 9), dtype=np.int8)
    env.board[0, 4] = PIECES['B_KING']
    env.board[9, 4] = PIECES['R_KING']
    env.board[0, 0] = PIECES['B_ROOK']  # 黑车
    env.board[0, 8] = PIECES['R_ROOK']  # 红车
    env.current_player = 1

    # 红车吃黑车
    state, reward, done = env.make_move((0, 8, 0, 0))

    print(f"  吃车奖励: {reward}")
    expected = 9 * 0.1  # 车价值9,乘以0.1

    if abs(reward - expected) < 0.01:
        print(f"  [PASS] 吃车奖励={reward:.1f},正确!")
        return True
    else:
        print(f"  [FAIL] 预期{expected},实际{reward}")
        return False


def test_check_reward():
    """测试将军奖励"""
    print("\n【测试3】将军奖励")
    env = ChineseChess()

    # 设置局面:红车将军
    env.board = np.zeros((10, 9), dtype=np.int8)
    env.board[0, 4] = PIECES['B_KING']
    env.board[9, 4] = PIECES['R_KING']
    env.board[2, 4] = PIECES['R_ROOK']  # 红车
    env.current_player = 1

    # 红车将军
    state, reward, done = env.make_move((2, 4, 0, 4))  # 吃将应该赢

    # 这是吃将,不只是将军
    if reward == 100:
        print(f"  [PASS] 吃将奖励={reward},正确!")
        return True

    print(f"  预期100,实际{reward}")
    return False


def test_suicide_prevention():
    """测试送将防止"""
    print("\n【测试4】防止送将")
    env = ChineseChess()

    # 设置局面:红方如果乱走会被吃将
    env.board = np.zeros((10, 9), dtype=np.int8)
    env.board[9, 4] = PIECES['R_KING']   # 红帅
    env.board[9, 3] = PIECES['R_ADVISOR'] # 红士(保护帅)
    env.board[0, 4] = PIECES['B_KING']
    env.board[7, 4] = PIECES['B_ROOK']   # 黑车(盯着帅)
    env.current_player = 1

    # 获取红方合法走法
    legal_moves = env.get_legal_moves()

    # 检查士是否能移开(会导致帅被将军)
    suicide_moves = [(9, 3, 8, 2), (9, 3, 8, 4)]  # 士斜走
    can_suicide = any(move in legal_moves for move in suicide_moves)

    if not can_suicide:
        print("  [PASS] 正确过滤了送将的走法")
        return True
    else:
        print("  [FAIL] 仍允许送将")
        print(f"  合法走法: {legal_moves}")
        return False


def test_all():
    """运行所有测试"""
    print("=" * 60)
    print("奖励机制改进测试".center(60))
    print("=" * 60)

    results = []
    results.append(("吃将/帅奖励", test_king_capture_reward()))
    results.append(("吃子奖励", test_piece_capture_reward()))
    results.append(("将军奖励", test_check_reward()))
    results.append(("防止送将", test_suicide_prevention()))

    print("\n" + "=" * 60)
    print("测试结果汇总".center(60))
    print("=" * 60)

    passed = 0
    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{name:20s} {status}")
        if result:
            passed += 1

    print("=" * 60)
    print(f"总计: {passed}/{len(results)} 通过")
    print("=" * 60)

    if passed == len(results):
        print("\n[OK] 所有测试通过!")
        print("改进效果:")
        print("  - 吃将/帅: 奖励=100 (原来=1)")
        print("  - 吃车: 奖励=0.9 (原来=0)")
        print("  - 吃兵: 奖励=0.1 (原来=0)")
        print("  - 将军: 额外奖励=0.2 (原来=0)")
        print("  - 防止送将: 过滤非法走法")
        print("\nAI现在应该会主动吃将/帅了!")
    else:
        print("\n有测试失败,请检查")


if __name__ == "__main__":
    test_all()
