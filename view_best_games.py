"""
查看精彩对局 - 可视化已保存的精彩对局
使用方法:
    python view_best_games.py           # 查看所有精彩对局列表
    python view_best_games.py --index 0 # 播放第0局
    python view_best_games.py --latest 5 # 播放最近5局
"""
import pickle
import os
import sys
import pygame
from config import DATA_DIR, CELL_SIZE, COLOR_BG, COLOR_LINE, COLOR_RED, COLOR_BLACK
from chess_env import ChineseChess

def load_best_games():
    """加载精彩对局"""
    games_file = f"{DATA_DIR}/best_games.pkl"

    if not os.path.exists(games_file):
        print("没有找到精彩对局文件")
        print(f"文件路径: {games_file}")
        print("请先训练一段时间，等AI下出有胜负的对局后再查看")
        return []

    try:
        with open(games_file, 'rb') as f:
            games = pickle.load(f)
        return games
    except Exception as e:
        print(f"加载失败: {e}")
        return []


def list_best_games(games):
    """列出所有精彩对局"""
    if not games:
        print("\n暂无精彩对局")
        return

    print("\n" + "=" * 80)
    print("精彩对局列表".center(80))
    print("=" * 80)

    # 按类型分组统计
    wins = [g for g in games if g['winner'] != 0]
    draws = [g for g in games if g['winner'] == 0]

    print(f"\n总计: {len(games)} 局")
    print(f"  - 有胜负: {len(wins)} 局")
    print(f"  - 快速和局: {len(draws)} 局")
    print("\n" + "-" * 80)

    # 显示最近10局
    print("\n最近的精彩对局:")
    print(f"{'序号':<6} {'时间':<20} {'训练局数':<10} {'结果':<10} {'步数':<6} {'类型':<10}")
    print("-" * 80)

    for i, game in enumerate(reversed(games[-10:])):
        index = len(games) - 1 - i
        timestamp = game['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        total_games = game['total_games']

        # 结果
        if game['winner'] == 1:
            result = "红胜"
        elif game['winner'] == -1:
            result = "黑胜"
        else:
            result = "和局"

        moves = game['moves']
        game_type = game['type']

        print(f"{index:<6} {timestamp:<20} {total_games:<10} {result:<10} {moves:<6} {game_type:<10}")

    print("\n" + "=" * 80)
    print(f"查看指定对局: python view_best_games.py --index <序号>")
    print(f"播放最近N局: python view_best_games.py --latest <数量>")
    print("=" * 80 + "\n")


class BestGameViewer:
    """精彩对局播放器"""

    def __init__(self):
        pygame.init()

        # 窗口设置
        self.board_width = 9 * CELL_SIZE
        self.board_height = 10 * CELL_SIZE
        self.margin = 50
        self.info_height = 100

        self.screen_width = self.board_width + 2 * self.margin
        self.screen_height = self.board_height + 2 * self.margin + self.info_height

        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption('精彩对局回放')

        self.clock = pygame.time.Clock()

        # 字体
        self.font_large = pygame.font.SysFont('microsoftyahei', 28)
        self.font_medium = pygame.font.SysFont('microsoftyahei', 20)
        self.font_small = pygame.font.SysFont('microsoftyahei', 16)

    def draw_board(self, board):
        """绘制棋盘"""
        self.screen.fill((240, 240, 240))

        # 绘制棋盘背景
        board_rect = pygame.Rect(self.margin, self.margin, self.board_width, self.board_height)
        pygame.draw.rect(self.screen, COLOR_BG, board_rect)

        # 绘制线条
        for i in range(10):
            y = self.margin + i * CELL_SIZE
            pygame.draw.line(self.screen, COLOR_LINE,
                           (self.margin, y),
                           (self.margin + 8 * CELL_SIZE, y), 2)

        for i in range(9):
            x = self.margin + i * CELL_SIZE
            # 上半部分
            pygame.draw.line(self.screen, COLOR_LINE,
                           (x, self.margin),
                           (x, self.margin + 4 * CELL_SIZE), 2)
            # 下半部分
            pygame.draw.line(self.screen, COLOR_LINE,
                           (x, self.margin + 5 * CELL_SIZE),
                           (x, self.margin + 9 * CELL_SIZE), 2)

        # 绘制棋子
        piece_names = {
            0: '',
            1: '帅', 2: '仕', 3: '相', 4: '马', 5: '车', 6: '炮', 7: '兵',
            -1: '将', -2: '士', -3: '象', -4: '马', -5: '车', -6: '炮', -7: '卒'
        }

        for r in range(10):
            for c in range(9):
                piece = board[r, c]
                if piece != 0:
                    x = self.margin + c * CELL_SIZE
                    y = self.margin + r * CELL_SIZE

                    # 棋子圆圈
                    color = COLOR_RED if piece > 0 else COLOR_BLACK
                    pygame.draw.circle(self.screen, (255, 255, 200), (x, y), 22)
                    pygame.draw.circle(self.screen, color, (x, y), 22, 3)

                    # 棋子文字
                    text = self.font_large.render(piece_names[piece], True, color)
                    text_rect = text.get_rect(center=(x, y))
                    self.screen.blit(text, text_rect)

    def draw_info(self, step, total_steps, game_info, paused):
        """绘制信息栏"""
        info_y = self.board_height + 2 * self.margin + 10

        # 对局信息
        timestamp = game_info['timestamp'].strftime('%Y-%m-%d %H:%M')
        total_games = game_info['total_games']

        if game_info['winner'] == 1:
            result = "红方胜利"
            result_color = COLOR_RED
        elif game_info['winner'] == -1:
            result = "黑方胜利"
            result_color = COLOR_BLACK
        else:
            result = "和局"
            result_color = (100, 100, 100)

        # 第一行：对局结果
        text = self.font_large.render(f"{result} - {game_info['moves']}步", True, result_color)
        self.screen.blit(text, (self.margin, info_y))

        # 第二行：进度和训练信息
        info_text = f"步骤: {step}/{total_steps} | 训练局数: {total_games} | {timestamp}"
        text = self.font_small.render(info_text, True, (60, 60, 60))
        self.screen.blit(text, (self.margin, info_y + 35))

        # 第三行：操作提示
        if paused:
            status = "【已暂停】空格=继续 | ←→=前后 | Q=退出"
        else:
            status = "空格=暂停 | ←→=前后 | 1-5=速度 | Q=退出"

        text = self.font_small.render(status, True, (100, 100, 100))
        self.screen.blit(text, (self.margin, info_y + 60))

    def replay_game(self, game_info):
        """回放对局"""
        game_data = game_info['game_data']

        if not game_data:
            print("对局数据为空")
            return

        # 重建对局过程
        env = ChineseChess()
        env.reset()

        boards = [env.board.copy()]  # 初始局面

        # 执行所有走法，记录每步棋盘
        for board_state, move_probs, player in game_data:
            # 从move_probs中选择概率最高的走法
            if move_probs:
                best_move = max(move_probs.items(), key=lambda x: x[1])[0]
                env.make_move(best_move)
                boards.append(env.board.copy())

        # 播放控制
        current_step = 0
        speed = 2  # 1-5, 数字越大越快
        paused = False
        frame_delay = {1: 60, 2: 30, 3: 15, 4: 8, 5: 4}
        frame_counter = 0

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False

                    elif event.key == pygame.K_SPACE:
                        paused = not paused

                    elif event.key == pygame.K_LEFT:
                        current_step = max(0, current_step - 1)

                    elif event.key == pygame.K_RIGHT:
                        current_step = min(len(boards) - 1, current_step + 1)

                    elif event.key in [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5]:
                        speed = int(event.unicode)

            # 自动播放
            if not paused:
                frame_counter += 1
                if frame_counter >= frame_delay[speed]:
                    frame_counter = 0
                    current_step += 1
                    if current_step >= len(boards):
                        current_step = 0  # 循环播放

            # 绘制
            self.draw_board(boards[current_step])
            self.draw_info(current_step, len(boards) - 1, game_info, paused)

            pygame.display.flip()
            self.clock.tick(30)

        pygame.quit()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='查看精彩对局')
    parser.add_argument('--index', type=int, help='播放指定序号的对局')
    parser.add_argument('--latest', type=int, help='播放最近N局')

    args = parser.parse_args()

    # 加载对局
    games = load_best_games()

    if not games:
        return

    # 如果没有指定参数，显示列表
    if args.index is None and args.latest is None:
        list_best_games(games)
        return

    # 播放指定对局
    viewer = BestGameViewer()

    if args.index is not None:
        if 0 <= args.index < len(games):
            print(f"\n播放第 {args.index} 局对局...")
            viewer.replay_game(games[args.index])
        else:
            print(f"错误: 序号 {args.index} 超出范围 (0-{len(games)-1})")

    elif args.latest is not None:
        print(f"\n连续播放最近 {args.latest} 局...")
        for i in range(min(args.latest, len(games))):
            index = len(games) - args.latest + i
            if index >= 0:
                print(f"\n[{i+1}/{args.latest}] 播放第 {index} 局...")
                viewer.replay_game(games[index])


if __name__ == "__main__":
    main()
