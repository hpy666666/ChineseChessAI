"""
完整训练测试 - 测试1轮完整流程(包括评估和日志)
"""
import sys
from trainer import Trainer

print("=" * 60)
print("完整训练流程测试(1轮)")
print("=" * 60)

# 临时修改配置,只训练3局
import config
config.SELF_PLAY_GAMES = 3
config.EVALUATE_INTERVAL = 1  # 第1轮就评估

print(f"\n配置: 每轮{config.SELF_PLAY_GAMES}局, 每{config.EVALUATE_INTERVAL}轮评估")

# 保存原始配置
original_games = 100
original_eval = 5

trainer = Trainer()

print("\n开始完整训练流程...")
try:
    # 训练1轮
    trainer.train_loop(num_iterations=1)

    print("\n" + "=" * 60)
    print("[成功] 完整训练流程测试通过!")
    print("=" * 60)

    print(f"\n检查结果:")
    print(f"  总对局数: {trainer.total_games}")
    print(f"  缓冲区大小: {len(trainer.replay_buffer)}")

    # 检查日志
    import os
    from config import LOG_DIR
    log_file = f"{LOG_DIR}/training.log"

    if os.path.exists(log_file):
        print(f"  [OK] 日志文件已生成: {log_file}")
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print(f"  日志行数: {len(lines)}")
            if lines:
                print(f"  最后一行: {lines[-1].strip()}")
    else:
        print(f"  [警告] 日志文件未生成")

    # 恢复配置
    config.SELF_PLAY_GAMES = original_games
    config.EVALUATE_INTERVAL = original_eval

    print("\n下一步:")
    print("  1. 运行 python main.py train 开始正式训练")
    print("  2. 运行 python plot_progress.py 查看进步曲线")
    print("  3. 运行 python main.py watch 观看AI对局")

except KeyboardInterrupt:
    print("\n用户中断")
except Exception as e:
    print(f"\n[错误] {e}")
    import traceback
    traceback.print_exc()
