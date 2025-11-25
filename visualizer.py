"""
可视化界面 - 使用Pygame显示棋盘和对局过程
让你能看到AI是怎么下棋的
"""
import pygame
import sys
from chess_env import ChineseChess
from neural_network import ChessNet
from self_play import MCTS
from config import *

# 初始化Pygame
pygame.init()

# 字体
FONT_LARGE = pygame.font.SysFont('microsoftyahei', 32)
FONT_MEDIUM = pygame.font.SysFont('microsoftyahei', 24)
FONT_SMALL = pygame.font.SysFont('microsoftyahei', 18)

class GameVisualizer:
    """游戏可视化器"""
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption('中国象棋AI训练 - 实时对局')
        self.clock = pygame.time.Clock()

        # 棋盘偏移（留出边距）
        self.offset_x = 50
        self.offset_y = 50

        # 棋子名称映射
        self.piece_names = {
            0: '',
            1: '帅', 2: '士', 3: '相', 4: '马', 5: '车', 6: '炮', 7: '兵',
            -1: '将', -2: '士', -3: '象', -4: '马', -5: '车', -6: '炮', -7: '卒'
        }

    def draw_board(self):
        """绘制棋盘"""
        self.screen.fill(COLOR_BG)

        # 绘制网格线
        for i in range(BOARD_SIZE):
            # 横线
            start_x = self.offset_x
            end_x = self.offset_x + (BOARD_WIDTH - 1) * CELL_SIZE
            y = self.offset_y + i * CELL_SIZE
            pygame.draw.line(self.screen, COLOR_LINE, (start_x, y), (end_x, y), 2)

        for i in range(BOARD_WIDTH):
            # 纵线（楚河汉界处断开）
            x = self.offset_x + i * CELL_SIZE
            # 上半部分
            pygame.draw.line(self.screen, COLOR_LINE,
                           (x, self.offset_y),
                           (x, self.offset_y + 4 * CELL_SIZE), 2)
            # 下半部分
            pygame.draw.line(self.screen, COLOR_LINE,
                           (x, self.offset_y + 5 * CELL_SIZE),
                           (x, self.offset_y + 9 * CELL_SIZE), 2)

        # 绘制九宫格斜线
        # 上方（黑方）
        pygame.draw.line(self.screen, COLOR_LINE,
                        (self.offset_x + 3 * CELL_SIZE, self.offset_y),
                        (self.offset_x + 5 * CELL_SIZE, self.offset_y + 2 * CELL_SIZE), 2)
        pygame.draw.line(self.screen, COLOR_LINE,
                        (self.offset_x + 5 * CELL_SIZE, self.offset_y),
                        (self.offset_x + 3 * CELL_SIZE, self.offset_y + 2 * CELL_SIZE), 2)

        # 下方（红方）
        pygame.draw.line(self.screen, COLOR_LINE,
                        (self.offset_x + 3 * CELL_SIZE, self.offset_y + 7 * CELL_SIZE),
                        (self.offset_x + 5 * CELL_SIZE, self.offset_y + 9 * CELL_SIZE), 2)
        pygame.draw.line(self.screen, COLOR_LINE,
                        (self.offset_x + 5 * CELL_SIZE, self.offset_y + 7 * CELL_SIZE),
                        (self.offset_x + 3 * CELL_SIZE, self.offset_y + 9 * CELL_SIZE), 2)

        # 绘制"楚河汉界"文字
        river_text = FONT_MEDIUM.render("楚河", True, COLOR_LINE)
        self.screen.blit(river_text, (self.offset_x + CELL_SIZE, self.offset_y + 4.3 * CELL_SIZE))
        river_text2 = FONT_MEDIUM.render("汉界", True, COLOR_LINE)
        self.screen.blit(river_text2, (self.offset_x + 6 * CELL_SIZE, self.offset_y + 4.3 * CELL_SIZE))

    def draw_piece(self, row, col, piece, highlight=False):
        """绘制棋子"""
        if piece == 0:
            return

        x = self.offset_x + col * CELL_SIZE
        y = self.offset_y + row * CELL_SIZE

        # 棋子圆圈
        color = COLOR_RED if piece > 0 else COLOR_BLACK
        radius = 25

        if highlight:
            pygame.draw.circle(self.screen, COLOR_HIGHLIGHT, (x, y), radius + 3, 3)

        pygame.draw.circle(self.screen, COLOR_BG, (x, y), radius)
        pygame.draw.circle(self.screen, color, (x, y), radius, 3)

        # 棋子文字
        text = FONT_LARGE.render(self.piece_names[piece], True, color)
        text_rect = text.get_rect(center=(x, y))
        self.screen.blit(text, text_rect)

    def draw_game_state(self, env, info_text=""):
        """绘制游戏状态"""
        self.draw_board()

        # 绘制所有棋子
        for r in range(BOARD_SIZE):
            for c in range(BOARD_WIDTH):
                self.draw_piece(r, c, env.board[r, c])

        # 绘制信息
        player_text = "红方走棋" if env.current_player == 1 else "黑方走棋"
        text = FONT_MEDIUM.render(player_text, True, COLOR_LINE)
        self.screen.blit(text, (10, 10))

        move_text = FONT_SMALL.render(f"步数: {env.move_count}", True, COLOR_LINE)
        self.screen.blit(text, (10, 40))

        if info_text:
            info = FONT_SMALL.render(info_text, True, COLOR_LINE)
            self.screen.blit(info, (10, WINDOW_HEIGHT - 30))

        pygame.display.flip()

    def watch_game(self, network, num_games=1):
        """
        观看AI自我对弈
        network: 训练好的网络
        num_games: 观看几局
        """
        for game_num in range(num_games):
            env = ChineseChess()
            mcts = MCTS(network)

            print(f"\n=== 第 {game_num + 1}/{num_games} 局 ===")

            for move_count in range(MAX_MOVES):
                # 处理退出事件
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()

                board, current_player = env.get_state()
                legal_moves = env.get_legal_moves()

                if len(legal_moves) == 0:
                    break

                # 显示当前局面
                self.draw_game_state(env, f"正在思考... (MCTS {MCTS_SIMULATIONS}次模拟)")
                self.clock.tick(FPS)

                # MCTS搜索
                visit_counts = mcts.search(env)

                if len(visit_counts) == 0:
                    break

                # 选择最佳走法
                best_move = max(visit_counts, key=visit_counts.get)
                print(f"第{move_count+1}步: {best_move}, 访问次数: {visit_counts[best_move]}")

                # 执行走法
                state, reward, done = env.make_move(best_move)

                # 显示走法后的局面
                self.draw_game_state(env, f"上一步: {best_move}")
                pygame.time.wait(500)  # 暂停0.5秒让你看清楚

                if done:
                    break

            # 游戏结束
            winner_text = ""
            if env.winner == 1:
                winner_text = "红方获胜！"
            elif env.winner == -1:
                winner_text = "黑方获胜！"
            else:
                winner_text = "和局"

            print(f"对局结束: {winner_text}")

            # 显示结果
            self.draw_game_state(env, winner_text)
            pygame.time.wait(3000)  # 停留3秒


def main():
    """主函数 - 观看AI对局"""
    print("加载神经网络...")

    network = ChessNet().to(DEVICE)

    # 尝试加载训练好的模型
    if os.path.exists(LATEST_MODEL):
        checkpoint = torch.load(LATEST_MODEL, map_location=DEVICE)
        network.load_state_dict(checkpoint['model_state_dict'])
        print(f"已加载模型: {LATEST_MODEL}")
        print(f"已训练 {checkpoint.get('total_games', 0)} 局")
    else:
        print("未找到训练好的模型，使用随机初始化网络")

    network.eval()

    # 创建可视化器
    visualizer = GameVisualizer()

    # 观看对局
    visualizer.watch_game(network, num_games=5)

    pygame.quit()


if __name__ == "__main__":
    main()
