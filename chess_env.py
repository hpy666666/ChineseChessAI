"""
中国象棋环境 - 实现所有规则
包括：棋盘、走法生成、合法性检查、胜负判断
"""
import numpy as np
from config import PIECES, BOARD_SIZE, BOARD_WIDTH

class ChineseChess:
    def __init__(self):
        """初始化棋盘"""
        self.reset()

    def reset(self):
        """重置到初始局面"""
        # 创建10x9的棋盘，0表示空位
        self.board = np.zeros((BOARD_SIZE, BOARD_WIDTH), dtype=np.int8)

        # 初始化红方（下方，正数）
        # 车
        self.board[9, 0] = self.board[9, 8] = PIECES['R_ROOK']
        # 马
        self.board[9, 1] = self.board[9, 7] = PIECES['R_KNIGHT']
        # 相
        self.board[9, 2] = self.board[9, 6] = PIECES['R_BISHOP']
        # 士
        self.board[9, 3] = self.board[9, 5] = PIECES['R_ADVISOR']
        # 帅
        self.board[9, 4] = PIECES['R_KING']
        # 炮
        self.board[7, 1] = self.board[7, 7] = PIECES['R_CANNON']
        # 兵
        for i in [0, 2, 4, 6, 8]:
            self.board[6, i] = PIECES['R_PAWN']

        # 初始化黑方（上方，负数）
        self.board[0, 0] = self.board[0, 8] = PIECES['B_ROOK']
        self.board[0, 1] = self.board[0, 7] = PIECES['B_KNIGHT']
        self.board[0, 2] = self.board[0, 6] = PIECES['B_BISHOP']
        self.board[0, 3] = self.board[0, 5] = PIECES['B_ADVISOR']
        self.board[0, 4] = PIECES['B_KING']
        self.board[2, 1] = self.board[2, 7] = PIECES['B_CANNON']
        for i in [0, 2, 4, 6, 8]:
            self.board[3, i] = PIECES['B_PAWN']

        self.current_player = 1  # 1=红方, -1=黑方
        self.move_count = 0
        self.winner = None

        return self.get_state()

    def get_state(self):
        """
        获取当前状态（用于神经网络输入）
        返回: 10x9的棋盘 + 当前玩家
        """
        return self.board.copy(), self.current_player

    def get_legal_moves(self):
        """
        获取当前玩家所有合法走法
        返回: [(from_row, from_col, to_row, to_col), ...]
        """
        moves = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_WIDTH):
                piece = self.board[r, c]
                # 只看当前玩家的棋子
                if piece * self.current_player > 0:
                    moves.extend(self._get_piece_moves(r, c, piece))
        return moves

    def _get_piece_moves(self, r, c, piece):
        """获取指定棋子的所有可能走法"""
        piece_type = abs(piece)
        moves = []

        if piece_type == abs(PIECES['R_KING']):  # 帅/将
            moves = self._king_moves(r, c)
        elif piece_type == abs(PIECES['R_ADVISOR']):  # 士
            moves = self._advisor_moves(r, c)
        elif piece_type == abs(PIECES['R_BISHOP']):  # 相/象
            moves = self._bishop_moves(r, c)
        elif piece_type == abs(PIECES['R_KNIGHT']):  # 马
            moves = self._knight_moves(r, c)
        elif piece_type == abs(PIECES['R_ROOK']):  # 车
            moves = self._rook_moves(r, c)
        elif piece_type == abs(PIECES['R_CANNON']):  # 炮
            moves = self._cannon_moves(r, c)
        elif piece_type == abs(PIECES['R_PAWN']):  # 兵/卒
            moves = self._pawn_moves(r, c)

        # 过滤掉非法走法（吃自己的子、出界等）
        valid_moves = []
        for to_r, to_c in moves:
            if 0 <= to_r < BOARD_SIZE and 0 <= to_c < BOARD_WIDTH:
                target = self.board[to_r, to_c]
                # 不能吃自己的子
                if target * self.current_player <= 0:
                    valid_moves.append((r, c, to_r, to_c))

        return valid_moves

    def _king_moves(self, r, c):
        """帅/将：九宫格内一步"""
        moves = []
        # 九宫格范围
        if self.current_player == 1:  # 红方
            row_range = (7, 10)
        else:  # 黑方
            row_range = (0, 3)
        col_range = (3, 6)

        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if row_range[0] <= nr < row_range[1] and col_range[0] <= nc < col_range[1]:
                moves.append((nr, nc))

        return moves

    def _advisor_moves(self, r, c):
        """士：九宫格内斜走"""
        moves = []
        if self.current_player == 1:
            row_range = (7, 10)
        else:
            row_range = (0, 3)
        col_range = (3, 6)

        for dr, dc in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
            nr, nc = r + dr, c + dc
            if row_range[0] <= nr < row_range[1] and col_range[0] <= nc < col_range[1]:
                moves.append((nr, nc))

        return moves

    def _bishop_moves(self, r, c):
        """相/象：田字格，不过河"""
        moves = []
        river = 5 if self.current_player == 1 else 4

        for dr, dc in [(2, 2), (2, -2), (-2, 2), (-2, -2)]:
            nr, nc = r + dr, c + dc
            # 检查目标位置是否在棋盘内
            if not (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_WIDTH):
                continue
            # 不过河
            if self.current_player == 1 and nr < river:
                continue
            if self.current_player == -1 and nr >= river:
                continue
            # 象眼不能被堵
            block_r, block_c = r + dr // 2, c + dc // 2
            if self.board[block_r, block_c] == 0:
                moves.append((nr, nc))

        return moves

    def _knight_moves(self, r, c):
        """马：日字格，蹩马腿"""
        moves = []
        # 8个方向
        offsets = [
            (2, 1, 1, 0), (2, -1, 1, 0),   # 向下
            (-2, 1, -1, 0), (-2, -1, -1, 0),  # 向上
            (1, 2, 0, 1), (-1, 2, 0, 1),   # 向右
            (1, -2, 0, -1), (-1, -2, 0, -1)   # 向左
        ]

        for dr, dc, block_dr, block_dc in offsets:
            # 先检查马腿位置是否在棋盘内
            block_r, block_c = r + block_dr, c + block_dc
            if 0 <= block_r < BOARD_SIZE and 0 <= block_c < BOARD_WIDTH:
                # 检查马腿是否被堵
                if self.board[block_r, block_c] == 0:
                    moves.append((r + dr, c + dc))

        return moves

    def _rook_moves(self, r, c):
        """车：直线走，不跳子"""
        moves = []
        # 四个方向
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            for step in range(1, max(BOARD_SIZE, BOARD_WIDTH)):
                nr, nc = r + dr * step, c + dc * step
                if not (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_WIDTH):
                    break
                if self.board[nr, nc] != 0:
                    moves.append((nr, nc))  # 可以吃子
                    break
                moves.append((nr, nc))

        return moves

    def _cannon_moves(self, r, c):
        """炮：直线走，隔子吃"""
        moves = []
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            jumped = False
            for step in range(1, max(BOARD_SIZE, BOARD_WIDTH)):
                nr, nc = r + dr * step, c + dc * step
                if not (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_WIDTH):
                    break

                if self.board[nr, nc] == 0:
                    if not jumped:
                        moves.append((nr, nc))  # 不吃子时正常走
                else:
                    if not jumped:
                        jumped = True  # 遇到第一个子，作为炮架
                    else:
                        moves.append((nr, nc))  # 隔子吃
                        break

        return moves

    def _pawn_moves(self, r, c):
        """兵/卒：过河前只能前进，过河后可左右"""
        moves = []
        if self.current_player == 1:  # 红兵
            moves.append((r - 1, c))  # 向上
            if r < 5:  # 过河了
                moves.append((r, c - 1))
                moves.append((r, c + 1))
        else:  # 黑卒
            moves.append((r + 1, c))  # 向下
            if r >= 5:  # 过河了
                moves.append((r, c - 1))
                moves.append((r, c + 1))

        return moves

    def make_move(self, move):
        """
        执行走法
        move: (from_row, from_col, to_row, to_col)
        返回: (新状态, 奖励, 是否结束)
        """
        from_r, from_c, to_r, to_c = move

        # 移动棋子
        captured = self.board[to_r, to_c]
        self.board[to_r, to_c] = self.board[from_r, from_c]
        self.board[from_r, from_c] = 0

        # 检查是否吃掉将/帅
        reward = 0
        done = False
        if abs(captured) == abs(PIECES['R_KING']):
            self.winner = self.current_player
            reward = 1  # 赢了
            done = True

        # 切换玩家
        self.current_player *= -1
        self.move_count += 1

        # 超过最大步数判和
        if self.move_count >= 200:
            done = True
            reward = 0

        return self.get_state(), reward, done

    def render(self):
        """
        文本显示棋盘（调试用）
        """
        piece_names = {
            0: '·',
            1: '帅', 2: '士', 3: '相', 4: '马', 5: '车', 6: '炮', 7: '兵',
            -1: '将', -2: '士', -3: '象', -4: '马', -5: '车', -6: '炮', -7: '卒'
        }

        print("\n  ", end="")
        for i in range(BOARD_WIDTH):
            print(f"{i} ", end="")
        print()

        for r in range(BOARD_SIZE):
            print(f"{r} ", end="")
            for c in range(BOARD_WIDTH):
                print(piece_names[self.board[r, c]], end=" ")
            print()
        print(f"\n当前: {'红方' if self.current_player == 1 else '黑方'}")
        print(f"步数: {self.move_count}")
