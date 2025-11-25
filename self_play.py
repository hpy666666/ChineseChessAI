"""
自我对弈系统 - 使用MCTS（蒙特卡洛树搜索）+ 神经网络
这是AI学习的核心：通过自己和自己下棋来积累经验
"""
import numpy as np
import math
from chess_env import ChineseChess
from config import MCTS_SIMULATIONS, MAX_MOVES

class MCTSNode:
    """蒙特卡洛树的节点"""
    def __init__(self, parent=None, move=None, prior_prob=0):
        self.parent = parent
        self.move = move  # 到达此节点的走法
        self.children = {}  # 子节点 {move: MCTSNode}

        self.visit_count = 0
        self.value_sum = 0  # 累计价值
        self.prior_prob = prior_prob  # 先验概率（来自神经网络）

    def value(self):
        """节点平均价值"""
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def is_leaf(self):
        """是否叶子节点"""
        return len(self.children) == 0

    def select_child(self, c_puct=1.5):
        """
        选择最优子节点（UCB公式）
        平衡 探索(exploration) vs 利用(exploitation)
        """
        best_score = -float('inf')
        best_move = None
        best_child = None

        for move, child in self.children.items():
            # UCB分数 = 平均价值 + 探索奖励
            score = child.value() + c_puct * child.prior_prob * math.sqrt(
                self.visit_count) / (1 + child.visit_count)

            if score > best_score:
                best_score = score
                best_move = move
                best_child = child

        return best_move, best_child

    def expand(self, move_probs):
        """
        扩展节点：为所有合法走法创建子节点
        move_probs: {move: probability}
        """
        for move, prob in move_probs.items():
            if move not in self.children:
                self.children[move] = MCTSNode(parent=self, move=move, prior_prob=prob)

    def update(self, value):
        """
        反向传播：更新此节点及所有祖先节点
        value: 从叶子节点得到的价值评估
        """
        self.visit_count += 1
        self.value_sum += value

        if self.parent:
            # 对手视角的价值是相反的
            self.parent.update(-value)


class MCTS:
    """蒙特卡洛树搜索"""
    def __init__(self, network):
        self.network = network

    def search(self, env, num_simulations=MCTS_SIMULATIONS):
        """
        执行MCTS搜索
        env: ChineseChess环境
        返回: {move: visit_count} 访问次数分布（用于训练）
        """
        root = MCTSNode()

        # 执行多次模拟
        for _ in range(num_simulations):
            # 1. 选择：从根节点向下选择到叶子节点
            node = root
            search_env = self._copy_env(env)

            while not node.is_leaf():
                move, node = node.select_child()
                search_env.make_move(move)

            # 2. 评估：使用神经网络评估叶子节点
            board, current_player = search_env.get_state()
            legal_moves = search_env.get_legal_moves()

            # 检查游戏是否结束
            if len(legal_moves) == 0 or search_env.winner is not None:
                # 游戏结束
                if search_env.winner == current_player:
                    value = 1
                elif search_env.winner == -current_player:
                    value = -1
                else:
                    value = 0
            else:
                # 使用神经网络评估
                move_probs, value = self.network.predict(board, current_player, legal_moves)

                # 3. 扩展：为叶子节点添加子节点
                node.expand(move_probs)

            # 4. 回溯：更新路径上所有节点
            node.update(value)

        # 返回根节点的访问次数分布
        visit_counts = {move: child.visit_count
                       for move, child in root.children.items()}

        return visit_counts

    def _copy_env(self, env):
        """复制环境（用于模拟）"""
        new_env = ChineseChess()
        new_env.board = env.board.copy()
        new_env.current_player = env.current_player
        new_env.move_count = env.move_count
        new_env.winner = env.winner
        return new_env


def self_play_game(network, temperature=1.0, render=False):
    """
    进行一局自我对弈
    network: 神经网络
    temperature: 温度参数（控制随机性，越高越随机）
    返回: [(state, move_probs, current_player), ...], winner
    """
    env = ChineseChess()
    mcts = MCTS(network)

    game_data = []  # 存储 (棋盘状态, MCTS走法分布, 当前玩家)

    for move_num in range(MAX_MOVES):
        board, current_player = env.get_state()
        legal_moves = env.get_legal_moves()

        if len(legal_moves) == 0:
            break

        # MCTS搜索
        visit_counts = mcts.search(env)

        if len(visit_counts) == 0:
            break

        # 转换为概率分布
        moves = list(visit_counts.keys())
        counts = np.array(list(visit_counts.values()))

        # 温度采样
        if temperature < 0.01:
            # 选择访问次数最多的
            move_probs = np.zeros(len(counts))
            move_probs[np.argmax(counts)] = 1
        else:
            # 按访问次数的概率分布采样
            counts = counts ** (1.0 / temperature)
            move_probs = counts / counts.sum()

        # 保存训练数据
        game_data.append((
            board.copy(),
            {move: prob for move, prob in zip(moves, move_probs)},
            current_player
        ))

        # 选择走法
        move_idx = np.random.choice(len(moves), p=move_probs)
        move = moves[move_idx]

        # 执行走法
        state, reward, done = env.make_move(move)

        if render:
            env.render()
            print(f"走法: {move}, 访问次数: {visit_counts[move]}")

        if done:
            break

    # 确定胜者
    winner = env.winner if env.winner else 0

    # 为每个状态分配最终奖励
    game_data_with_reward = []
    for board, move_probs, player in game_data:
        if winner == 0:
            reward = 0  # 和局
        elif winner == player:
            reward = 1  # 赢
        else:
            reward = -1  # 输

        game_data_with_reward.append((board, move_probs, reward))

    return game_data_with_reward, winner


def test_self_play():
    """测试自我对弈"""
    from neural_network import ChessNet
    import torch

    print("测试自我对弈系统...")
    network = ChessNet().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    network.eval()

    print("开始一局自我对弈（只用10次模拟，快速测试）...")
    import config
    config.MCTS_SIMULATIONS = 10

    game_data, winner = self_play_game(network, render=True)

    print(f"\n对局结束！")
    print(f"总步数: {len(game_data)}")
    print(f"胜者: {['和局', '红方', '黑方'][winner]}")
    print(f"收集到 {len(game_data)} 个训练样本")


if __name__ == "__main__":
    test_self_play()
