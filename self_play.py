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
        执行MCTS搜索 (性能优化版: 使用批量神经网络推理)
        env: ChineseChess环境
        返回: {move: visit_count} 访问次数分布（用于训练）
        """
        root = MCTSNode()

        # 批量大小: 每次收集多个叶子节点一起推理
        batch_size = 8  # 可以根据GPU内存调整

        for batch_start in range(0, num_simulations, batch_size):
            batch_end = min(batch_start + batch_size, num_simulations)
            batch_count = batch_end - batch_start

            # 收集批量的叶子节点
            leaf_nodes = []
            leaf_envs = []
            leaf_data = []

            for _ in range(batch_count):
                # 1. 选择：从根节点向下选择到叶子节点
                node = root
                search_env = self._copy_env(env)

                while not node.is_leaf():
                    move, node = node.select_child()
                    search_env.make_move(move)

                # 收集叶子节点
                board, current_player = search_env.get_state()
                legal_moves = search_env.get_legal_moves()

                # 检查游戏是否结束
                if len(legal_moves) == 0 or search_env.winner is not None:
                    # 游戏结束，直接计算价值
                    if search_env.winner == current_player:
                        value = 1
                    elif search_env.winner == -current_player:
                        value = -1
                    else:
                        value = 0
                    # 直接更新
                    node.update(value)
                else:
                    # 需要神经网络评估的节点
                    leaf_nodes.append(node)
                    leaf_envs.append((board, current_player, legal_moves))

            # 2. 批量评估所有叶子节点
            if len(leaf_nodes) > 0:
                results = self.network.predict_batch(leaf_envs)

                # 3. 扩展和回溯
                for node, (move_probs, value) in zip(leaf_nodes, results):
                    node.expand(move_probs)
                    node.update(value)

        # 返回根节点的访问次数分布
        visit_counts = {move: child.visit_count
                       for move, child in root.children.items()}

        return visit_counts

    def _copy_env(self, env):
        """
        复制环境（用于模拟）
        性能优化: 只复制MCTS模拟必需的最小状态
        """
        new_env = ChineseChess()
        # 只复制棋盘（最耗时的操作）
        new_env.board = env.board.copy()
        # 复制基本状态
        new_env.current_player = env.current_player
        new_env.move_count = env.move_count
        new_env.winner = env.winner
        # 复制将帅位置缓存（避免重新查找）
        new_env.red_king_pos = env.red_king_pos
        new_env.black_king_pos = env.black_king_pos
        # 复制无吃子计数（用于50回合判和）
        new_env.no_capture_count = env.no_capture_count
        # 不复制历史记录（MCTS模拟中的历史检测已经足够宽松，不需要）
        # position_history, check_history, chase_history 保持空
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

    # 为每个状态分配奖励(改进版:结合即时奖励和最终结果)
    game_data_with_reward = []
    for i, (board, move_probs, player) in enumerate(game_data):
        # 1. 最终奖励(输赢)
        if winner == 0:
            # 和局: 红方先手有优势,和局视为轻微失败
            # 这样可以鼓励红方进攻,避免AI学会"无脑和棋"
            if player == 1:  # 红方
                final_reward = -0.1  # 轻微惩罚
            else:  # 黑方
                final_reward = 0.1   # 轻微奖励(守和成功)
        elif winner == player:
            final_reward = 1  # 赢
        else:
            final_reward = -1  # 输

        # 2. 即时奖励已经在make_move中计算过了
        # 这里只使用最终奖励,因为即时奖励在MCTS中已经使用

        game_data_with_reward.append((board, move_probs, final_reward))

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
