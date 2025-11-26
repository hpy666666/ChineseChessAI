"""
进步曲线可视化工具
读取训练日志,绘制AI的进步曲线
"""
import re
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
from config import LOG_DIR

# 设置中文字体
rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
rcParams['axes.unicode_minus'] = False


def parse_training_log(log_file):
    """
    解析训练日志
    返回: {
        'rounds': [1, 2, 3, ...],
        'avg_moves': [45, 42, 38, ...],
        'red_wins': [50, 52, 55, ...],
        'black_wins': [45, 43, 40, ...],
        'draws': [5, 5, 5, ...],
        'total_games': [100, 200, 300, ...]
    }
    """
    if not os.path.exists(log_file):
        print(f"日志文件不存在: {log_file}")
        return None

    data = {
        'rounds': [],
        'avg_moves': [],
        'red_wins': [],
        'black_wins': [],
        'draws': [],
        'total_games': []
    }

    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 正则匹配训练轮次
    rounds = re.findall(r'【第 (\d+)/\d+ 轮训练】', content)
    data['rounds'] = [int(r) for r in rounds]

    # 匹配每轮的统计数据
    # 例如: "红胜:50 黑胜:45 和:5 | 平均步数: 42"
    stats = re.findall(r'红胜:(\d+)\s+黑胜:(\d+)\s+和:(\d+)\s+\|\s+平均步数:\s+(\d+)', content)

    for red, black, draw, avg_moves in stats:
        data['red_wins'].append(int(red))
        data['black_wins'].append(int(black))
        data['draws'].append(int(draw))
        data['avg_moves'].append(int(avg_moves))

    # 计算累计对局数
    if data['rounds']:
        games_per_round = data['red_wins'][0] + data['black_wins'][0] + data['draws'][0]
        data['total_games'] = [r * games_per_round for r in data['rounds']]

    return data


def plot_progress(data):
    """
    绘制进步曲线
    """
    if not data or not data['rounds']:
        print("没有数据可绘制!")
        return

    # 创建2x2的子图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('中国象棋AI训练进度', fontsize=16, fontweight='bold')

    # 1. 胜率变化
    ax1 = axes[0, 0]
    total_games = [r + b + d for r, b, d in zip(data['red_wins'], data['black_wins'], data['draws'])]
    red_rate = [r / t * 100 for r, t in zip(data['red_wins'], total_games)]
    black_rate = [b / t * 100 for b, t in zip(data['black_wins'], total_games)]
    draw_rate = [d / t * 100 for d, t in zip(data['draws'], total_games)]

    ax1.plot(data['rounds'], red_rate, 'r-o', label='红方胜率', linewidth=2)
    ax1.plot(data['rounds'], black_rate, 'b-s', label='黑方胜率', linewidth=2)
    ax1.plot(data['rounds'], draw_rate, 'g-^', label='和局率', linewidth=2)
    ax1.set_xlabel('训练轮次', fontsize=12)
    ax1.set_ylabel('胜率 (%)', fontsize=12)
    ax1.set_title('胜率变化趋势', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 平均步数变化
    ax2 = axes[0, 1]
    ax2.plot(data['rounds'], data['avg_moves'], 'm-o', linewidth=2)
    ax2.set_xlabel('训练轮次', fontsize=12)
    ax2.set_ylabel('平均步数', fontsize=12)
    ax2.set_title('对局平均步数', fontsize=14)
    ax2.grid(True, alpha=0.3)

    # 3. 累计对局数
    ax3 = axes[1, 0]
    ax3.plot(data['rounds'], data['total_games'], 'c-o', linewidth=2)
    ax3.set_xlabel('训练轮次', fontsize=12)
    ax3.set_ylabel('累计对局数', fontsize=12)
    ax3.set_title('训练进度', fontsize=14)
    ax3.grid(True, alpha=0.3)

    # 4. 胜负分布(最近一轮)
    ax4 = axes[1, 1]
    if data['red_wins']:
        labels = ['红方胜', '黑方胜', '和局']
        sizes = [data['red_wins'][-1], data['black_wins'][-1], data['draws'][-1]]
        colors = ['#ff6b6b', '#4ecdc4', '#95e1d3']
        explode = (0.1, 0, 0)

        ax4.pie(sizes, explode=explode, labels=labels, colors=colors,
                autopct='%1.1f%%', shadow=True, startangle=90)
        ax4.set_title(f'第{data["rounds"][-1]}轮胜负分布', fontsize=14)

    plt.tight_layout()

    # 保存图片
    output_file = os.path.join(LOG_DIR, 'training_progress.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"[OK] 进步曲线已保存到: {output_file}")

    # 显示图形
    plt.show()


def print_statistics(data):
    """打印训练统计摘要"""
    if not data or not data['rounds']:
        return

    print("\n" + "=" * 60)
    print("训练统计摘要".center(60))
    print("=" * 60)

    total_rounds = len(data['rounds'])
    total_games = data['total_games'][-1] if data['total_games'] else 0

    print(f"\n总训练轮次: {total_rounds}")
    print(f"累计对局数: {total_games}")

    if data['avg_moves']:
        print(f"\n平均步数:")
        print(f"  初始: {data['avg_moves'][0]} 步")
        print(f"  当前: {data['avg_moves'][-1]} 步")
        print(f"  变化: {data['avg_moves'][-1] - data['avg_moves'][0]:+d} 步")

    if data['red_wins']:
        total = data['red_wins'][-1] + data['black_wins'][-1] + data['draws'][-1]
        print(f"\n最近一轮胜负:")
        print(f"  红方胜: {data['red_wins'][-1]} ({data['red_wins'][-1]/total*100:.1f}%)")
        print(f"  黑方胜: {data['black_wins'][-1]} ({data['black_wins'][-1]/total*100:.1f}%)")
        print(f"  和局:   {data['draws'][-1]} ({data['draws'][-1]/total*100:.1f}%)")

    print("\n" + "=" * 60)


def main():
    """主函数"""
    log_file = os.path.join(LOG_DIR, 'training.log')

    print("正在读取训练日志...")
    data = parse_training_log(log_file)

    if data:
        print_statistics(data)
        print("\n正在绘制进步曲线...")
        plot_progress(data)
    else:
        print("无法读取训练数据。请先运行训练程序生成日志。")


if __name__ == "__main__":
    main()
