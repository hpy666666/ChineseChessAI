"""
测试长将/长捉检测机制
"""
from chess_env import ChineseChess
from config import PIECES

def test_perpetual_check():
    """测试长将检测（宽松版）"""
    print("=" * 60)
    print("测试长将检测")
    print("=" * 60)

    env = ChineseChess()

    # 模拟连续将军的情况
    print("\n模拟场景: 连续将军12步，其中10步是将军")

    # 手动设置check_history来模拟
    # True = 将军, False = 不是将军
    env.check_history = [
        True,   # 第1步: 将军
        False,  # 第2步: 应将(不算将军)
        True,   # 第3步: 将军
        False,  # 第4步: 应将
        True,   # 第5步: 将军
        False,  # 第6步: 应将
        True,   # 第7步: 将军
        False,  # 第8步: 应将
        True,   # 第9步: 将军
        False,  # 第10步: 应将
        True,   # 第11步: 将军
        False,  # 第12步: 应将
    ]

    result = env._check_perpetual_check()
    print(f"12步中6步将军: {result}")
    print(f"预期: False (不到10步)")
    print(f"结果: {'[OK] 通过' if not result else '[!] 失败'}")

    # 测试真正的长将
    print("\n模拟场景: 连续将军12步，其中11步是将军")
    env.check_history = [
        True, True, True, True, True, True,
        True, True, True, True, True, False
    ]

    result = env._check_perpetual_check()
    print(f"12步中11步将军: {result}")
    print(f"预期: True (长将判负)")
    print(f"结果: {'[OK] 通过' if result else '[!] 失败'}")

def test_perpetual_chase():
    """测试长捉检测（宽松版）"""
    print("\n" + "=" * 60)
    print("测试长捉检测")
    print("=" * 60)

    env = ChineseChess()

    # 模拟连续捉子
    print("\n模拟场景: 连续捉子12步，其中6步有捉子")

    # 手动设置chase_history
    # [] = 没有捉子, [((r,c), (tr,tc))] = 有捉子
    env.chase_history = [
        [((5,5), (3,5))],  # 第1步: 捉子
        [],                # 第2步: 无捉子
        [((4,5), (3,5))],  # 第3步: 捉子
        [],                # 第4步: 无捉子
        [((5,5), (3,5))],  # 第5步: 捉子
        [],                # 第6步: 无捉子
        [((4,5), (3,5))],  # 第7步: 捉子
        [],                # 第8步: 无捉子
        [((5,5), (3,5))],  # 第9步: 捉子
        [],                # 第10步: 无捉子
        [((4,5), (3,5))],  # 第11步: 捉子
        [],                # 第12步: 无捉子
    ]

    result = env._check_perpetual_chase()
    print(f"12步中6步捉子: {result}")
    print(f"预期: False (不到10步)")
    print(f"结果: {'[OK] 通过' if not result else '[!] 失败'}")

    # 测试真正的长捉
    print("\n模拟场景: 连续捉子12步，其中11步有捉子")
    env.chase_history = [
        [((5,5), (3,5))],
        [((4,5), (3,5))],
        [((5,5), (3,5))],
        [((4,5), (3,5))],
        [((5,5), (3,5))],
        [((4,5), (3,5))],
        [((5,5), (3,5))],
        [((4,5), (3,5))],
        [((5,5), (3,5))],
        [((4,5), (3,5))],
        [((5,5), (3,5))],
        [],  # 只有最后一步不捉
    ]

    result = env._check_perpetual_chase()
    print(f"12步中11步捉子: {result}")
    print(f"预期: True (长捉判负)")
    print(f"结果: {'[OK] 通过' if result else '[!] 失败'}")

def test_summary():
    """总结规则"""
    print("\n" + "=" * 60)
    print("宽松版长将/长捉规则总结")
    print("=" * 60)
    print("\n长将检测:")
    print("  - 检查最近12步")
    print("  - 如果其中10步以上在将军 → 判负(-10分)")
    print("  - 目的: 禁止明显的滥用，但允许战术性将军")
    print("\n长捉检测:")
    print("  - 检查最近12步")
    print("  - 如果其中10步以上在捉子 → 判负(-10分)")
    print("  - 目的: 禁止明显的滥用，但允许战术性追捉")
    print("\n宽松程度:")
    print("  - 允许偶尔的将军/捉子")
    print("  - 只惩罚持续、明显的违规行为")
    print("  - 适合训练初期，不会过度限制AI探索")
    print("=" * 60)

if __name__ == "__main__":
    test_perpetual_check()
    test_perpetual_chase()
    test_summary()
