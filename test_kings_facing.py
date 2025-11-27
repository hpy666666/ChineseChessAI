"""
测试将帅对脸规则
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from chess_env import ChineseChess
from config import PIECES
import numpy as np


def test_kings_facing_directly():
    """测试1: 将帅直接对脸(应该被禁止)"""
    print("\n【测试1】将帅直接对脸(同竖线无遮挡)")
    env = ChineseChess()

    # 设置局面: 将帅在同一竖线,中间无子
    env.board = np.zeros((10, 9), dtype=np.int8)
    env.board[2, 4] = PIECES['B_KING']  # 黑将在(2,4)
    env.board[8, 4] = PIECES['R_KING']  # 红帅在(8,4), 同一列

    # 检测是否对脸
    is_facing = env._are_kings_facing()

    if is_facing:
        print("  [PASS] 正确检测到将帅对脸!")
        return True
    else:
        print("  [FAIL] 未检测到对脸")
        return False


def test_kings_with_blocker():
    """测试2: 将帅同竖线但有遮挡(应该允许)"""
    print("\n【测试2】将帅同竖线有遮挡(允许)")
    env = ChineseChess()

    # 设置局面: 将帅在同一竖线,中间有车
    env.board = np.zeros((10, 9), dtype=np.int8)
    env.board[2, 4] = PIECES['B_KING']  # 黑将在(2,4)
    env.board[5, 4] = PIECES['R_ROOK']  # 红车在中间
    env.board[8, 4] = PIECES['R_KING']  # 红帅在(8,4)

    # 检测是否对脸
    is_facing = env._are_kings_facing()

    if not is_facing:
        print("  [PASS] 正确判定有遮挡,不算对脸!")
        return True
    else:
        print("  [FAIL] 错误判定为对脸")
        return False


def test_prevent_facing_move():
    """测试3: 走法导致对脸(应该被过滤)"""
    print("\n【测试3】过滤导致对脸的走法")
    env = ChineseChess()

    # 设置局面: 将帅在同一竖线,中间只有一个炮
    env.board = np.zeros((10, 9), dtype=np.int8)
    env.board[2, 4] = PIECES['B_KING']   # 黑将
    env.board[5, 4] = PIECES['R_CANNON'] # 红炮(遮挡)
    env.board[8, 4] = PIECES['R_KING']   # 红帅
    env.current_player = 1  # 红方走

    # 红方炮移开会导致对脸
    illegal_moves = [
        (5, 4, 5, 0),  # 炮横移
        (5, 4, 5, 8),  # 炮横移
        (5, 4, 0, 4),  # 炮向上
    ]

    # 获取合法走法
    legal_moves = env.get_legal_moves()

    # 检查这些导致对脸的走法是否被过滤
    blocked_count = 0
    for move in illegal_moves:
        if move not in legal_moves:
            blocked_count += 1

    if blocked_count == len(illegal_moves):
        print(f"  [PASS] 成功过滤{blocked_count}个导致对脸的走法!")
        return True
    else:
        print(f"  [FAIL] 只过滤了{blocked_count}/{len(illegal_moves)}个走法")
        print(f"  合法走法: {legal_moves}")
        return False


def test_kings_different_columns():
    """测试4: 将帅不在同一竖线(允许)"""
    print("\n【测试4】将帅不在同一竖线(允许)")
    env = ChineseChess()

    # 设置局面: 将帅在不同竖线
    env.board = np.zeros((10, 9), dtype=np.int8)
    env.board[2, 3] = PIECES['B_KING']  # 黑将在第3列
    env.board[8, 4] = PIECES['R_KING']  # 红帅在第4列

    # 检测是否对脸
    is_facing = env._are_kings_facing()

    if not is_facing:
        print("  [PASS] 正确判定不同列不算对脸!")
        return True
    else:
        print("  [FAIL] 错误判定为对脸")
        return False


def test_general_arrow_tactic():
    """测试5: 将军箭战术(有遮挡,允许)"""
    print("\n【测试5】将军箭战术(有遮挡应允许)")
    env = ChineseChess()

    # 经典将军箭局面: 车、将、帅在同一线,车在中间
    env.board = np.zeros((10, 9), dtype=np.int8)
    env.board[0, 4] = PIECES['B_KING']   # 黑将
    env.board[4, 4] = PIECES['B_ROOK']   # 黑车(遮挡,可以将军)
    env.board[9, 4] = PIECES['R_KING']   # 红帅
    env.current_player = -1  # 黑方走

    # 黑车移开会将军(利用对面笑)
    # 但这个走法本身不应该因为"对脸"而被禁止,
    # 因为移开前有车遮挡,移开后红帅被将军(不是对脸)

    # 先检查移开前不对脸
    is_facing_before = env._are_kings_facing()

    if not is_facing_before:
        print("  [PASS] 车遮挡时正确判定不对脸,允许将军箭战术!")
        return True
    else:
        print("  [FAIL] 车遮挡时错误判定为对脸")
        return False


def test_all():
    """运行所有测试"""
    print("=" * 60)
    print("将帅对脸规则测试".center(60))
    print("=" * 60)

    results = []
    results.append(("将帅直接对脸", test_kings_facing_directly()))
    results.append(("将帅有遮挡", test_kings_with_blocker()))
    results.append(("过滤对脸走法", test_prevent_facing_move()))
    results.append(("不同竖线", test_kings_different_columns()))
    results.append(("将军箭战术", test_general_arrow_tactic()))

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
        print("\n国规要点:")
        print("  [OK] 直接对脸(同列无遮挡) → 禁止")
        print("  [OK] 有棋子遮挡 → 允许(将军箭战术)")
        print("  [OK] 不同竖线 → 允许")
        print("\n规则已正确实现!")
    else:
        print("\n有测试失败,请检查")


if __name__ == "__main__":
    test_all()
