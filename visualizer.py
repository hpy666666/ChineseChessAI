"""
可视化界面 - 使用Pygame显示棋盘和对局过程
让你能看到AI是怎么下棋的
"""
import pygame
import sys
import os
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

        # 速度控制
        self.speed_delays = {
            '极慢': 3000,   # 3秒/步
            '慢速': 1500,   # 1.5秒/步
            '正常': 500,    # 0.5秒/步
            '快速': 200,    # 0.2秒/步
            '极快': 50      # 0.05秒/步
        }
        self.current_speed = '正常'
        self.paused = False
        self.step_mode = False  # 单步模式

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
        self.screen.blit(move_text, (10, 40))

        # 显示速度和暂停状态
        speed_text = FONT_SMALL.render(f"速度: {self.current_speed} [1-5键调节]", True, COLOR_LINE)
        self.screen.blit(speed_text, (10, 70))

        if self.paused:
            pause_text = FONT_MEDIUM.render("【暂停】空格继续 / Enter单步", True, (255, 0, 0))
            self.screen.blit(pause_text, (WINDOW_WIDTH // 2 - 150, 10))

        # 显示控制提示
        help_text = FONT_SMALL.render("空格:暂停/继续 | Enter:单步 | Q:退出", True, COLOR_LINE)
        self.screen.blit(help_text, (10, WINDOW_HEIGHT - 60))

        if info_text:
            info = FONT_SMALL.render(info_text, True, COLOR_LINE)
            self.screen.blit(info, (10, WINDOW_HEIGHT - 30))

        pygame.display.flip()

    def handle_controls(self):
        """
        处理键盘控制
        返回: True(继续) / False(退出)
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            if event.type == pygame.KEYDOWN:
                # 退出
                if event.key == pygame.K_q:
                    return False

                # 暂停/继续
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                    self.step_mode = False

                # 单步执行
                elif event.key == pygame.K_RETURN:
                    self.step_mode = True
                    self.paused = False

                # 速度调节 1-5
                elif event.key == pygame.K_1:
                    self.current_speed = '极慢'
                elif event.key == pygame.K_2:
                    self.current_speed = '慢速'
                elif event.key == pygame.K_3:
                    self.current_speed = '正常'
                elif event.key == pygame.K_4:
                    self.current_speed = '快速'
                elif event.key == pygame.K_5:
                    self.current_speed = '极快'

        return True

    def get_board_position(self, mouse_x, mouse_y):
        """
        将鼠标坐标转换为棋盘位置
        返回: (row, col) 或 None
        """
        # 检查是否在棋盘范围内
        rel_x = mouse_x - self.offset_x
        rel_y = mouse_y - self.offset_y

        if rel_x < -CELL_SIZE//2 or rel_x > (BOARD_WIDTH - 1) * CELL_SIZE + CELL_SIZE//2:
            return None
        if rel_y < -CELL_SIZE//2 or rel_y > (BOARD_SIZE - 1) * CELL_SIZE + CELL_SIZE//2:
            return None

        # 转换为棋盘坐标(四舍五入到最近的交叉点)
        col = round(rel_x / CELL_SIZE)
        row = round(rel_y / CELL_SIZE)

        # 边界检查
        if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_WIDTH:
            return (row, col)
        return None

    def human_vs_ai(self, network, human_color='red'):
        """
        人机对战模式
        network: AI网络
        human_color: 'red' 或 'black', 人类执哪方
        """
        env = ChineseChess()
        mcts = MCTS(network)

        human_player = 1 if human_color == 'red' else -1
        ai_player = -human_player

        selected_pos = None  # 当前选中的棋子位置
        legal_moves_for_selected = []  # 选中棋子的合法走法

        print(f"\n=== 人机对战 ===")
        print(f"人类执: {'红方' if human_color == 'red' else '黑方'}")
        print(f"控制说明:")
        print(f"  - 鼠标点击选择棋子")
        print(f"  - 再次点击目标位置移动")
        print(f"  - Q键: 退出")

        running = True
        while running and not env.winner:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        return

                # 只在人类回合处理鼠标点击
                if event.type == pygame.MOUSEBUTTONDOWN and env.current_player == human_player:
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    pos = self.get_board_position(mouse_x, mouse_y)

                    if pos:
                        row, col = pos

                        # 如果已经选中了一个棋子
                        if selected_pos:
                            from_r, from_c = selected_pos

                            # 检查是否是合法走法
                            move = (from_r, from_c, row, col)
                            if move in legal_moves_for_selected:
                                # 执行走法
                                env.make_move(move)
                                selected_pos = None
                                legal_moves_for_selected = []
                                print(f"人类走棋: {move}")
                            else:
                                # 如果点击的是自己的其他棋子,切换选择
                                piece = env.board[row, col]
                                if (piece > 0 and human_player == 1) or (piece < 0 and human_player == -1):
                                    selected_pos = (row, col)
                                    # 计算这个棋子的合法走法
                                    all_moves = env.get_legal_moves()
                                    legal_moves_for_selected = [m for m in all_moves if m[0] == row and m[1] == col]
                                else:
                                    # 点击了空位或对方棋子，取消选择
                                    selected_pos = None
                                    legal_moves_for_selected = []
                        else:
                            # 选择一个棋子
                            piece = env.board[row, col]
                            if (piece > 0 and human_player == 1) or (piece < 0 and human_player == -1):
                                selected_pos = (row, col)
                                # 计算这个棋子的合法走法
                                all_moves = env.get_legal_moves()
                                legal_moves_for_selected = [m for m in all_moves if m[0] == row and m[1] == col]

            # 绘制棋盘
            self.draw_board()

            # 绘制所有棋子(高亮选中的)
            for r in range(BOARD_SIZE):
                for c in range(BOARD_WIDTH):
                    highlight = (selected_pos == (r, c))
                    self.draw_piece(r, c, env.board[r, c], highlight)

            # 绘制选中棋子的可移动位置提示
            if legal_moves_for_selected:
                for move in legal_moves_for_selected:
                    _, _, to_r, to_c = move
                    x = self.offset_x + to_c * CELL_SIZE
                    y = self.offset_y + to_r * CELL_SIZE
                    # 绘制半透明圆圈提示
                    pygame.draw.circle(self.screen, (100, 255, 100), (x, y), 10)

            # 显示信息
            current_player_text = "红方" if env.current_player == 1 else "黑方"
            your_turn = " (您的回合)" if env.current_player == human_player else " (AI思考中)"
            text = FONT_MEDIUM.render(current_player_text + your_turn, True, COLOR_LINE)
            self.screen.blit(text, (10, 10))

            move_text = FONT_SMALL.render(f"步数: {env.move_count}", True, COLOR_LINE)
            self.screen.blit(move_text, (10, 40))

            help_text = FONT_SMALL.render("点击棋子选择, 再点目标位置移动 | Q:退出", True, COLOR_LINE)
            self.screen.blit(help_text, (10, WINDOW_HEIGHT - 30))

            pygame.display.flip()
            self.clock.tick(FPS)

            # AI回合
            if env.current_player == ai_player and not env.winner:
                legal_moves = env.get_legal_moves()
                if len(legal_moves) == 0:
                    break

                # 显示AI正在思考
                self.draw_board()
                for r in range(BOARD_SIZE):
                    for c in range(BOARD_WIDTH):
                        self.draw_piece(r, c, env.board[r, c])
                ai_text = FONT_MEDIUM.render("AI思考中...", True, (255, 100, 0))
                self.screen.blit(ai_text, (WINDOW_WIDTH // 2 - 80, WINDOW_HEIGHT // 2))
                pygame.display.flip()

                # MCTS搜索
                visit_counts = mcts.search(env)

                if len(visit_counts) > 0:
                    # AI使用较低温度(更确定)
                    import numpy as np
                    moves = list(visit_counts.keys())
                    counts = np.array(list(visit_counts.values()))
                    # 温度0.3让AI更确定，但仍有一定随机性避免太机械
                    temperature = 0.3
                    counts = counts ** (1.0 / temperature)
                    move_probs = counts / counts.sum()
                    move_idx = np.random.choice(len(moves), p=move_probs)
                    best_move = moves[move_idx]

                    env.make_move(best_move)
                    print(f"AI走棋: {best_move}")

            # 检查游戏是否结束
            # 注意：不在这里检查env.winner，因为while条件已经检查了
            # 只检查最大步数限制（这会在下一次循环判断时触发和局）
            if env.move_count >= MAX_MOVES:
                env.winner = 0  # 100步限制，判和
                break

        # 游戏结束
        winner_text = ""
        if env.winner == 1:
            winner_text = "红方获胜！" + (" 恭喜您赢了！" if human_player == 1 else " AI获胜！")
        elif env.winner == -1:
            winner_text = "黑方获胜！" + (" 恭喜您赢了！" if human_player == -1 else " AI获胜！")
        else:
            winner_text = "和局！"

        # 添加结束原因
        end_reason = env.end_reason if env.end_reason else "未知原因"
        print(f"\n对局结束: {winner_text}")
        print(f"结束原因: {end_reason}")

        # 显示结果并等待
        self.draw_board()
        for r in range(BOARD_SIZE):
            for c in range(BOARD_WIDTH):
                self.draw_piece(r, c, env.board[r, c])

        result_text = FONT_LARGE.render(winner_text, True, (255, 0, 0))
        self.screen.blit(result_text, (WINDOW_WIDTH // 2 - 150, WINDOW_HEIGHT // 2))

        close_text = FONT_SMALL.render("按任意键或关闭窗口退出", True, COLOR_LINE)
        self.screen.blit(close_text, (WINDOW_WIDTH // 2 - 120, WINDOW_HEIGHT // 2 + 50))

        pygame.display.flip()

        # 等待用户关闭
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or event.type == pygame.KEYDOWN:
                    waiting = False

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
            print("控制说明:")
            print("  空格键: 暂停/继续")
            print("  Enter键: 单步执行")
            print("  1-5键: 调整速度(1=极慢, 5=极快)")
            print("  Q键: 退出")

            for move_count in range(MAX_MOVES):
                # 处理控制
                if not self.handle_controls():
                    pygame.quit()
                    sys.exit()

                # 如果暂停,持续更新画面直到解除暂停
                while self.paused:
                    if not self.handle_controls():
                        pygame.quit()
                        sys.exit()
                    self.draw_game_state(env, f"【暂停中】按空格继续,Enter单步")
                    self.clock.tick(30)

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

                # 使用温度采样选择走法(和训练时一样,增加随机性)
                import numpy as np
                moves = list(visit_counts.keys())
                counts = np.array(list(visit_counts.values()))

                # 使用较低温度(0.5)让棋局更合理但仍有变化
                temperature = 0.5
                counts = counts ** (1.0 / temperature)
                move_probs = counts / counts.sum()

                # 随机采样
                move_idx = np.random.choice(len(moves), p=move_probs)
                best_move = moves[move_idx]
                print(f"第{move_count+1}步: {best_move}, 访问次数: {visit_counts[best_move]}")

                # 执行走法
                state, reward, done = env.make_move(best_move)

                # 显示走法后的局面
                self.draw_game_state(env, f"上一步: {best_move}")

                # 根据速度暂停
                delay = self.speed_delays[self.current_speed]
                pygame.time.wait(delay)

                # 如果是单步模式,自动暂停
                if self.step_mode:
                    self.paused = True
                    self.step_mode = False

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

            # 添加结束原因
            end_reason = env.end_reason if env.end_reason else "未知原因"
            print(f"对局结束: {winner_text} - {end_reason}")

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
