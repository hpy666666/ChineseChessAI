"""
快速训练测试 - 测试3局看是否正常工作
"""
import sys
import torch
from trainer import Trainer
from config import SELF_PLAY_GAMES

# 临时修改配置
import config
original_games = config.SELF_PLAY_GAMES
config.SELF_PLAY_GAMES = 3  # 只训练3局

print("=" * 60)
print("快速训练测试(3局)")
print("=" * 60)

trainer = Trainer()

print("\n开始测试训练流程...")
try:
    # 只训练1轮
    trainer.collect_self_play_data(3)

    print("\n[OK] 训练流程正常!")
    print(f"  完成3局对弈")
    print(f"  缓冲区大小: {len(trainer.replay_buffer)}")

    # 恢复配置
    config.SELF_PLAY_GAMES = original_games

    print("\n提示:")
    print("  训练流程已验证正常,现在可以:")
    print("  1. 运行 python main.py train 开始完整训练")
    print("  2. 每局都会实时显示进度")
    print("  3. 100局大约需要2-3小时")

except KeyboardInterrupt:
    print("\n用户中断")
except Exception as e:
    print(f"\n错误: {e}")
    import traceback
    traceback.print_exc()
