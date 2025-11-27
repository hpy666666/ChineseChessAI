"""
测试所有修复
"""
import sys
sys.path.insert(0, 'D:\\ChineseChessAI')

from chess_env import ChineseChess
import re

def test_checkmate_detection():
    """测试将死判断"""
    print("=" * 60)
    print("测试1: 将死判断")
    print("=" * 60)

    env = ChineseChess()

    # 设置一个简单的将死局面
    # 黑将在(0,4)，红车在(0,5)和(1,4)
    env.board.fill(0)
    env.board[0, 4] = -1  # 黑将
    env.board[0, 5] = 5   # 红车
    env.board[1, 4] = 5   # 红车
    env.board[9, 4] = 1   # 红帅(避免对面错误)

    env.black_king_pos = (0, 4)
    env.red_king_pos = (9, 4)
    env.current_player = -1  # 黑方行棋

    # 黑方没有合法走法且被将军 = 将死
    print(f"黑方合法走法数: {len(env.get_legal_moves())}")
    print(f"黑方是否被将军: {env._is_in_check(-1)}")
    print(f"是否将死: {env._check_checkmate()}")

    if env._check_checkmate():
        print("✓ 将死判断正确!")
    else:
        print("✗ 将死判断失败!")

    print()


def test_stalemate_detection():
    """测试困毙判断"""
    print("=" * 60)
    print("测试2: 困毙判断")
    print("=" * 60)

    env = ChineseChess()

    # 设置一个困毙局面（无合法走法但未被将军）
    env.board.fill(0)
    env.board[0, 0] = -1  # 黑将在角落
    env.board[1, 1] = 5   # 红车控制
    env.board[2, 0] = 5   # 红车控制
    env.board[9, 4] = 1   # 红帅

    env.black_king_pos = (0, 0)
    env.red_king_pos = (9, 4)
    env.current_player = -1  # 黑方行棋

    print(f"黑方合法走法数: {len(env.get_legal_moves())}")
    print(f"黑方是否被将军: {env._is_in_check(-1)}")
    print(f"是否困毙: {env._check_stalemate()}")

    if env._check_stalemate():
        print("✓ 困毙判断正确!")
    else:
        print("✗ 困毙判断失败!")

    print()


def test_log_format():
    """测试日志格式"""
    print("=" * 60)
    print("测试3: 训练日志格式")
    print("=" * 60)

    # 测试新日志格式的正则匹配
    test_log = "2025-11-26 20:31:28.546135 | 轮次:1 | 总局数:100 | 红胜:45 黑胜:40 和:15 | 平均步数:52.3 | 缓冲区:9967 | 类型:训练"

    pattern = r'轮次:(\d+).*?总局数:(\d+).*?红胜:(\d+)\s+黑胜:(\d+)\s+和:(\d+).*?平均步数:([\d.]+)'
    match = re.search(pattern, test_log)

    if match:
        print("日志解析成功:")
        print(f"  轮次: {match.group(1)}")
        print(f"  总局数: {match.group(2)}")
        print(f"  红胜: {match.group(3)}")
        print(f"  黑胜: {match.group(4)}")
        print(f"  和局: {match.group(5)}")
        print(f"  平均步数: {match.group(6)}")
        print("✓ 日志格式解析正确!")
    else:
        print("✗ 日志格式解析失败!")

    print()


def main():
    print("\n开始测试所有修复...\n")

    test_checkmate_detection()
    test_stalemate_detection()
    test_log_format()

    print("=" * 60)
    print("所有测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
