"""
中国象棋环境 - 实现所有规则
包括：棋盘、走法生成、合法性检查、胜负判断
"""
import numpy as np
from typing import Tuple, List, Optional, Set
from config import PIECES, BOARD_SIZE, BOARD_WIDTH

class ChineseChess:
    def __init__(self) -> None:
        """初始化棋盘"""
        self.reset()

    def reset(self) -> Tuple[np.ndarray, int]:
        """重置到初始局面"""
        # 创建10x9的棋盘，0表示空位
        self.board = np.zeros((BOARD_SIZE, BOARD_WIDTH), dtype=np.int8)

        # 棋例细则相关
        self.position_history = []  # 局面历史(用于检测重复)
        self.no_capture_count = 0   # 无吃子计数(用于50回合规则)
        self.check_history = []     # 将军历史(用于检测长将)
        self.chase_history = []     # 追捉历史(用于检测长捉)
        self.consecutive_checks = 0  # 连续将军次数(用于防止刷分)

        # 性能优化: 缓存将帅位置，避免每次遍历棋盘查找
        self.red_king_pos = None
        self.black_king_pos = None

        # 对局结束原因
        self.end_reason = None

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
        self.red_king_pos = (9, 4)  # 初始化红帅位置
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
        self.black_king_pos = (0, 4)  # 初始化黑将位置
        self.board[2, 1] = self.board[2, 7] = PIECES['B_CANNON']
        for i in [0, 2, 4, 6, 8]:
            self.board[3, i] = PIECES['B_PAWN']

        self.current_player = 1  # 1=红方, -1=黑方
        self.move_count = 0
        self.winner = None
        self.end_reason = None

        return self.get_state()

    def get_state(self) -> Tuple[np.ndarray, int]:
        """
        获取当前状态（用于神经网络输入）
        返回: 10x9的棋盘 + 当前玩家
        """
        return self.board.copy(), self.current_player

    def get_legal_moves(self) -> List[Tuple[int, int, int, int]]:
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

    def _get_piece_moves(self, r: int, c: int, piece: int) -> List[Tuple[int, int, int, int]]:
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
                    # 检查是否送将(走完后自己被将军)
                    if not self._is_move_suicide(r, c, to_r, to_c):
                        valid_moves.append((r, c, to_r, to_c))

        return valid_moves

    def _king_moves(self, r: int, c: int) -> List[Tuple[int, int]]:
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

    def _advisor_moves(self, r: int, c: int) -> List[Tuple[int, int]]:
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

    def _bishop_moves(self, r: int, c: int) -> List[Tuple[int, int]]:
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

    def _knight_moves(self, r: int, c: int) -> List[Tuple[int, int]]:
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

    def _rook_moves(self, r: int, c: int) -> List[Tuple[int, int]]:
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

    def _cannon_moves(self, r: int, c: int) -> List[Tuple[int, int]]:
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

    def _pawn_moves(self, r: int, c: int) -> List[Tuple[int, int]]:
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

    def make_move(self, move: Tuple[int, int, int, int]) -> Tuple[Tuple[np.ndarray, int], float, bool]:
        """
        执行走法
        move: (from_row, from_col, to_row, to_col)
        返回: (新状态, 奖励, 是否结束)
        """
        from_r, from_c, to_r, to_c = move

        # 记录走法前的状态(用于检测捉)
        threats_before = self._get_threatened_pieces(self.current_player)

        # 移动棋子
        captured = self.board[to_r, to_c]
        moving_piece = self.board[from_r, from_c]
        self.board[to_r, to_c] = moving_piece
        self.board[from_r, from_c] = 0

        # 性能优化: 更新将帅位置缓存
        if moving_piece == PIECES['R_KING']:
            self.red_king_pos = (to_r, to_c)
        elif moving_piece == PIECES['B_KING']:
            self.black_king_pos = (to_r, to_c)
        # 如果吃掉了对方的将/帅，清除其位置缓存
        if captured == PIECES['R_KING']:
            self.red_king_pos = None
        elif captured == PIECES['B_KING']:
            self.black_king_pos = None

        # 更新无吃子计数
        if captured != 0:
            self.no_capture_count = 0  # 重置
        else:
            self.no_capture_count += 1

        # 计算奖励(改进版v2:添加更精细的中间奖励)
        reward = 0
        done = False

        # 1. 吃掉将/帅 -> 立即获胜
        if abs(captured) == abs(PIECES['R_KING']):
            self.winner = self.current_player
            reward = 100  # 最高奖励!
            done = True
            player_name = "红方" if self.current_player == 1 else "黑方"
            self.end_reason = f"{player_name}吃掉对方将帅"

        # 2. 吃子奖励(按价值分配) - 增大奖励引导AI进攻
        elif captured != 0:
            piece_values = {
                abs(PIECES['R_ROOK']): 9,    # 车
                abs(PIECES['R_CANNON']): 4.5, # 炮
                abs(PIECES['R_KNIGHT']): 4,   # 马
                abs(PIECES['R_BISHOP']): 2,   # 象/相
                abs(PIECES['R_ADVISOR']): 2,  # 士
                abs(PIECES['R_PAWN']): 1      # 兵/卒
            }
            base_value = piece_values.get(abs(captured), 0)
            reward = base_value * 2.0  # 吃车=18分,吃兵=2分

            # 额外奖励：吃掉重要防御棋子（士、象）
            if abs(captured) in [abs(PIECES['R_ADVISOR']), abs(PIECES['R_BISHOP'])]:
                reward += 3.0  # 削弱对方防御

        # 3. 将军奖励 - 大幅增加引导AI攻击对方将帅，但防止刷分
        is_checking = self._is_in_check(-self.current_player)
        if not done and is_checking:
            # 实施递减奖励机制，防止重复将军刷分
            if self.consecutive_checks == 0:
                reward += 15.0  # 第一次将军，高额奖励
            elif self.consecutive_checks == 1:
                reward += 10.0  # 第二次将军，减少奖励
            elif self.consecutive_checks == 2:
                reward += 5.0   # 第三次将军，进一步减少
            # 连续将军超过3次，不再给奖励（防止刷分）
            self.consecutive_checks += 1
        else:
            # 如果这步没有将军，重置连续将军计数
            self.consecutive_checks = 0

            # 4. 位置价值奖励（仅在非将军、非吃子时给予小额奖励）
            if captured == 0 and not done:
                position_reward = self._evaluate_position_change(move)
                reward += position_reward * 0.01  # 位置奖励很小，避免干扰主要目标

        # 记录局面历史
        self.position_history.append(self._get_position_hash())

        # 记录是否将军 (已在上面计算过)
        self.check_history.append(is_checking)

        # 记录是否捉子
        threats_after = self._get_threatened_pieces(self.current_player)
        self.chase_history.append(threats_after)

        # 切换玩家（先切换，再检查对方是否被将死）
        self.current_player *= -1
        self.move_count += 1

        # 检查各种和棋/判负条件
        if not done:
            # 1. 将死判负(当前玩家被将死,对方获胜)
            if self._check_checkmate():
                done = True
                reward = 200  # 将死是最高成就，给予最高奖励
                self.winner = -self.current_player  # 对方获胜（因为已切换玩家）
                loser_name = "红方" if self.current_player == 1 else "黑方"
                self.end_reason = f"将死{loser_name}"

            # 2. 三次重复局面判和
            elif self._check_draw_by_repetition():
                done = True
                reward = 0
                self.winner = 0
                self.end_reason = "三次重复局面判和"

            # 3. 50回合无吃子判和
            elif self._check_draw_by_fifty_moves():
                done = True
                reward = 0
                self.winner = 0
                self.end_reason = "50回合无吃子判和"

            # 4. 困毙判负(无合法走法且未被将军,当前玩家输棋,对方获胜)
            elif self._check_stalemate():
                done = True
                reward = 100  # 对方获胜奖励
                self.winner = -self.current_player  # 对方获胜
                loser_name = "红方" if self.current_player == 1 else "黑方"
                self.end_reason = f"困毙{loser_name}"

            # 5. 长将判负(当前走棋方判负)
            elif self._check_perpetual_check():
                done = True
                reward = -10  # 违规重罚
                self.winner = -self.current_player
                loser_name = "红方" if self.current_player == 1 else "黑方"
                self.end_reason = f"长将判负({loser_name})"

            # 6. 长捉判负(当前走棋方判负)
            elif self._check_perpetual_chase():
                done = True
                reward = -10
                self.winner = -self.current_player
                loser_name = "红方" if self.current_player == 1 else "黑方"
                self.end_reason = f"长捉判负({loser_name})"

        # 超过最大步数判和（只在未分出胜负时）- 添加轻微惩罚引导尽快结束
        if not done and self.move_count >= 70:  # 降低步数限制，引导AI尽快结束对局
            done = True
            reward = -2  # 和局惩罚，鼓励进攻
            self.winner = 0
            self.end_reason = f"超过{self.move_count}步判和"

        return self.get_state(), reward, done

    def render(self) -> None:
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

    def _is_move_suicide(self, from_r: int, from_c: int, to_r: int, to_c: int) -> bool:
        """
        检查走法是否会导致自己被将军(送将)
        返回: True(送将,非法) / False(安全,合法)
        """
        # 备份棋盘和将帅位置缓存
        backup_board = self.board.copy()
        backup_red_king = self.red_king_pos
        backup_black_king = self.black_king_pos

        # 模拟走法
        moving_piece = self.board[from_r, from_c]
        self.board[to_r, to_c] = moving_piece
        self.board[from_r, from_c] = 0

        # 更新将帅位置缓存（如果移动的是将帅）
        from config import PIECES
        if moving_piece == PIECES['R_KING']:
            self.red_king_pos = (to_r, to_c)
        elif moving_piece == PIECES['B_KING']:
            self.black_king_pos = (to_r, to_c)

        # 检查是否被将军
        in_check = self._is_in_check(self.current_player)

        # 检查是否导致将帅对脸
        kings_face_to_face = self._are_kings_facing()

        # 恢复棋盘和缓存
        self.board = backup_board
        self.red_king_pos = backup_red_king
        self.black_king_pos = backup_black_king

        return in_check or kings_face_to_face

    def _are_kings_facing(self) -> bool:
        """
        检查将帅是否对脸(同一竖线且中间无子)
        返回: True(对脸,非法) / False(不对脸或有遮挡,合法)

        性能优化: 使用缓存的将帅位置，避免遍历整个棋盘
        """
        # 使用缓存的位置
        if not self.red_king_pos or not self.black_king_pos:
            return False

        red_r, red_c = self.red_king_pos
        black_r, black_c = self.black_king_pos

        # 不在同一竖线,不对脸
        if red_c != black_c:
            return False

        # 在同一竖线,检查中间是否有棋子
        min_r = min(red_r, black_r)
        max_r = max(red_r, black_r)

        # 遍历中间所有位置
        for r in range(min_r + 1, max_r):
            if self.board[r, red_c] != 0:
                # 有棋子遮挡,允许(这是"将军箭"/"对面笑"的战术基础)
                return False

        # 同一竖线且中间无子 -> 对脸!
        return True

    def _get_position_hash(self):
        """
        获取当前局面的哈希值(用于检测重复局面)
        返回: 棋盘状态的唯一标识
        """
        # 将current_player转为0或1(因为-1不能直接用bytes)
        player_byte = 0 if self.current_player == 1 else 1
        return hash(self.board.tobytes() + bytes([player_byte]))

    def _is_in_check(self, player):
        """
        判断某方是否被将军
        player: 1(红方) 或 -1(黑方)
        返回: True/False

        性能优化: 使用缓存的将帅位置，避免遍历棋盘
        """
        # 使用缓存的将帅位置（性能优化）
        king_pos = self.red_king_pos if player == 1 else self.black_king_pos

        if not king_pos:
            return False  # 没有将/帅(已被吃)

        # 检查对方所有棋子是否能攻击到将/帅
        for r in range(BOARD_SIZE):
            for c in range(BOARD_WIDTH):
                piece = self.board[r, c]
                if piece * player < 0:  # 对方的子
                    # 直接获取基础走法(不检查送将,避免递归)
                    piece_type = abs(piece)
                    if piece_type == abs(PIECES['R_KING']):
                        moves = self._king_moves(r, c)
                    elif piece_type == abs(PIECES['R_ADVISOR']):
                        moves = self._advisor_moves(r, c)
                    elif piece_type == abs(PIECES['R_BISHOP']):
                        moves = self._bishop_moves(r, c)
                    elif piece_type == abs(PIECES['R_KNIGHT']):
                        moves = self._knight_moves(r, c)
                    elif piece_type == abs(PIECES['R_ROOK']):
                        moves = self._rook_moves(r, c)
                    elif piece_type == abs(PIECES['R_CANNON']):
                        moves = self._cannon_moves(r, c)
                    elif piece_type == abs(PIECES['R_PAWN']):
                        moves = self._pawn_moves(r, c)
                    else:
                        continue

                    # 检查是否能吃到将/帅
                    for to_r, to_c in moves:
                        if (to_r, to_c) == king_pos:
                            return True
        return False

    def _get_threatened_pieces(self, player):
        """
        获取某方威胁吃掉的对方棋子(用于检测捉)
        player: 威胁方(1或-1)
        返回: [(威胁子位置, 被威胁子位置), ...]
        """
        threats = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_WIDTH):
                piece = self.board[r, c]
                if piece * player > 0:  # 己方的子
                    original_player = self.current_player
                    self.current_player = player
                    moves = self._get_piece_moves(r, c, piece)
                    self.current_player = original_player

                    for move in moves:
                        to_r, to_c = move[2:4]
                        target = self.board[to_r, to_c]
                        # 如果能吃对方的子(排除将/帅,将军另外判断)
                        if target * player < 0 and abs(target) != abs(PIECES['R_KING']):
                            # 检查目标是否有根(被保护)
                            if not self._is_protected(to_r, to_c, -player):
                                threats.append(((r, c), (to_r, to_c)))
        return threats

    def _is_protected(self, r, c, player):
        """
        检查某个位置的棋子是否被保护
        r, c: 棋子位置
        player: 棋子所属方
        返回: True(有根)/False(无根)
        """
        # 检查己方是否有子能保护这个位置
        for pr in range(BOARD_SIZE):
            for pc in range(BOARD_WIDTH):
                piece = self.board[pr, pc]
                if piece * player > 0:  # 己方的子
                    original_player = self.current_player
                    self.current_player = player
                    moves = self._get_piece_moves(pr, pc, piece)
                    self.current_player = original_player

                    for move in moves:
                        if move[2:4] == (r, c):
                            return True
        return False

    def _check_draw_by_repetition(self):
        """
        检查是否因三次重复局面而和棋
        返回: True(判和)/False(继续)
        """
        current_hash = self._get_position_hash()
        count = self.position_history.count(current_hash)
        return count >= 3

    def _check_draw_by_fifty_moves(self):
        """
        检查是否因50回合无吃子而和棋
        返回: True(判和)/False(继续)
        """
        return self.no_capture_count >= 100  # 双方各50回合=100步

    def _check_checkmate(self):
        """
        检查是否将死(无合法走法且正被将军)
        返回: True(将死)/False(不是)
        """
        # 检查是否有合法走法
        legal_moves = self.get_legal_moves()
        if len(legal_moves) > 0:
            return False

        # 无合法走法 + 被将军 = 将死
        if self._is_in_check(self.current_player):
            return True

        return False

    def _check_stalemate(self) -> bool:
        """
        检查是否困毙(无合法走法且未被将军)
        返回: True(困毙)/False(不是)
        """
        # 检查是否有合法走法
        legal_moves = self.get_legal_moves()
        if len(legal_moves) > 0:
            return False

        # 无合法走法 + 未被将军 = 困毙
        if not self._is_in_check(self.current_player):
            return True

        return False

    def _check_perpetual_check(self) -> bool:
        """
        检查是否长将(宽松版本:连续将军过多次)
        返回: True(判负)/False(不是)

        简化规则:只要最近12步中有10步以上是将军,就判定为长将
        (足够宽松,只禁止明显的滥用)
        """
        if len(self.check_history) < 12:
            return False

        # 统计最近12步中将军的次数
        recent_checks = self.check_history[-12:]
        check_count = sum(1 for check in recent_checks if check)

        # 如果12步中有10步以上在将军,判定为长将
        return check_count >= 10

    def _check_perpetual_chase(self) -> bool:
        """
        检查是否长捉(训练初期暂时禁用)
        返回: True(判负)/False(不是)

        训练初期AI随机走棋,几乎每步都可能"捉"无根子,
        导致误判。等AI训练到一定水平后再启用此规则。

        TODO: 训练500局后可以考虑启用
        """
        return False  # 暂时禁用

        # 原检测代码(已禁用):
        # if len(self.chase_history) < 16:
        #     return False
        # recent_chases = self.chase_history[-16:]
        # chase_count = sum(1 for chase in recent_chases if len(chase) > 0)
        # return chase_count >= 15

    def _evaluate_position_change(self, move: Tuple[int, int, int, int]) -> float:
        """
        评估走法带来的位置价值变化
        返回: 位置评分变化 (-10 到 10)

        考虑因素:
        1. 棋子向前推进（进攻性）
        2. 控制中心区域
        3. 棋子活动性（控制更多格子）
        """
        from_r, from_c, to_r, to_c = move
        piece = self.board[to_r, to_c]  # 走法后的位置
        piece_type = abs(piece)

        score = 0

        # 1. 推进奖励：棋子向对方阵营移动
        if self.current_player == 1:  # 红方向上推进
            advance = from_r - to_r
        else:  # 黑方向下推进
            advance = to_r - from_r

        if advance > 0:
            # 不同棋子推进价值不同
            if piece_type == abs(PIECES['R_PAWN']):
                score += advance * 2.0  # 兵卒推进最重要
            elif piece_type in [abs(PIECES['R_ROOK']), abs(PIECES['R_CANNON'])]:
                score += advance * 1.5  # 车炮推进有价值
            elif piece_type == abs(PIECES['R_KNIGHT']):
                score += advance * 1.0  # 马推进有一定价值

        # 2. 中心控制：占据中心位置更有价值
        center_cols = [3, 4, 5]  # 中间三路
        if to_c in center_cols:
            score += 1.5
            if 3 <= to_r <= 6:  # 中心区域
                score += 1.0

        # 3. 兵过河奖励
        if piece_type == abs(PIECES['R_PAWN']):
            if self.current_player == 1 and to_r < 5:  # 红兵过河
                score += 3.0
            elif self.current_player == -1 and to_r >= 5:  # 黑卒过河
                score += 3.0

        # 4. 靠近对方将帅（进攻性）
        opponent_king_pos = self.black_king_pos if self.current_player == 1 else self.red_king_pos
        if opponent_king_pos:
            # 计算曼哈顿距离
            old_dist = abs(from_r - opponent_king_pos[0]) + abs(from_c - opponent_king_pos[1])
            new_dist = abs(to_r - opponent_king_pos[0]) + abs(to_c - opponent_king_pos[1])
            if new_dist < old_dist:
                score += (old_dist - new_dist) * 0.5  # 靠近对方将帅有奖励

        return score

    def _get_material_advantage(self) -> float:
        """
        计算材料优势（双方棋子价值差）
        返回: 当前玩家的材料优势 (-50 到 50)
        """
        piece_values = {
            abs(PIECES['R_ROOK']): 9,
            abs(PIECES['R_CANNON']): 4.5,
            abs(PIECES['R_KNIGHT']): 4,
            abs(PIECES['R_BISHOP']): 2,
            abs(PIECES['R_ADVISOR']): 2,
            abs(PIECES['R_PAWN']): 1
        }

        red_value = 0
        black_value = 0

        for r in range(BOARD_SIZE):
            for c in range(BOARD_WIDTH):
                piece = self.board[r, c]
                if piece > 0:  # 红方
                    red_value += piece_values.get(abs(piece), 0)
                elif piece < 0:  # 黑方
                    black_value += piece_values.get(abs(piece), 0)

        # 返回当前玩家的优势
        if self.current_player == 1:
            return red_value - black_value
        else:
            return black_value - red_value
