"""
自我对弈系统 - 使用MCTS（蒙特卡洛树搜索）+ 神经网络
这是AI学习的核心：通过自己和自己下棋来积累经验
"""
import numpy as np
import math
from typing import List, Tuple, Dict, Optional
from chess_env import ChineseChess
from config import MCTS_SIMULATIONS, MAX_MOVES


class InterruptedWithResults(Exception):
    """自定义异常：中断时携带已完成的结果"""
    def __init__(self, results: List[Tuple]) -> None:
        self.results = results
        super().__init__("Training interrupted by user")


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
    def __init__(self, network, num_simulations=None):
        self.network = network
        self.default_simulations = num_simulations if num_simulations else MCTS_SIMULATIONS

    def search(self, env, num_simulations=None):
        """
        执行MCTS搜索 (性能优化版: 使用批量神经网络推理)
        env: ChineseChess环境
        返回: {move: visit_count} 访问次数分布（用于训练）
        """
        if num_simulations is None:
            num_simulations = self.default_simulations

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


def self_play_game(network, temperature=1.0, render=False, num_simulations=None, opponent_network=None):
    """
    进行一局自我对弈或对抗训练
    network: 神经网络（红方）
    temperature: 温度参数（控制随机性，越高越随机）
    num_simulations: MCTS模拟次数（如果为None则使用默认值）
    opponent_network: 对手网络（黑方），如果为None则自我对弈
    返回: [(state, move_probs, reward), ...], winner, end_reason
    """
    env = ChineseChess()

    # 【优化】自我对弈时只创建一个MCTS，对抗时才创建两个
    if opponent_network is None:
        # 自我对弈模式：只用一个MCTS对象
        mcts = MCTS(network, num_simulations=num_simulations)
        mcts_red = mcts
        mcts_black = mcts
    else:
        # 对抗模式：创建两个独立的MCTS
        mcts_red = MCTS(network, num_simulations=num_simulations)
        mcts_black = MCTS(opponent_network, num_simulations=num_simulations)

    game_data = []  # 存储 (棋盘状态, MCTS走法分布, 当前玩家)
    step_rewards = []  # 【新增】存储每一步的即时奖励

    for move_num in range(MAX_MOVES):
        board, current_player = env.get_state()
        legal_moves = env.get_legal_moves()

        if len(legal_moves) == 0:
            break

        # 【修改】根据当前玩家选择MCTS
        mcts = mcts_red if current_player == 1 else mcts_black

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

        # 保存训练数据（只保存红方的数据用于训练新模型）
        if current_player == 1 or opponent_network is None:  # 自我对弈时保存双方数据
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

        # 【新增】保存即时奖励
        step_rewards.append(reward)

        if render:
            env.render()
            print(f"走法: {move}, 即时奖励: {reward:.2f}, 访问次数: {visit_counts[move]}")

        if done:
            break

    # 确定胜者和结束原因
    winner = env.winner if env.winner else 0
    end_reason = env.end_reason if env.end_reason else "未知原因"

    # 为每个状态分配奖励(改进版v3:最终奖励 + 即时奖励累积)
    game_data_with_reward = []
    game_length = len(game_data)

    for i, (board, move_probs, player) in enumerate(game_data):
        # 1. 最终奖励(输赢)
        if winner == 0:
            # 和局: 根据步数和先后手给予不同惩罚
            if game_length >= 60:  # 长时间和棋，惩罚更重
                if player == 1:  # 红方先手，和棋视为失败
                    final_reward = -0.15
                else:  # 黑方守和成功
                    final_reward = 0.05
            else:  # 短时间和棋，惩罚较轻
                if player == 1:
                    final_reward = -0.1
                else:
                    final_reward = 0.1
        elif winner == player:
            # 获胜：步数越少奖励越高
            base_reward = 1.0
            # 快速获胜额外奖励 (30步内获胜给予最高额外奖励)
            if game_length <= 30:
                quick_win_bonus = 0.5
            elif game_length <= 50:
                quick_win_bonus = 0.3
            elif game_length <= 70:
                quick_win_bonus = 0.1
            else:
                quick_win_bonus = 0.0
            final_reward = base_reward + quick_win_bonus
        else:
            # 失败：步数越多损失越大（说明被慢慢压制）
            if game_length >= 60:
                final_reward = -1.2  # 被长期压制
            else:
                final_reward = -1.0

        # 2. 【核心修复】加入即时奖励！
        # 找到对应的即时奖励（考虑到可能只保存红方数据）
        immediate_reward = 0.0
        if i < len(step_rewards):
            immediate_reward = step_rewards[i]

        # 总奖励 = 最终奖励 + 即时奖励的缩放版本
        # 缩放因子0.01：避免即时奖励(18分)压倒最终奖励(1.0分)
        total_reward = final_reward + immediate_reward * 0.01

        game_data_with_reward.append((board, move_probs, total_reward))

    return game_data_with_reward, winner, end_reason


def _play_game_worker(args):
    """
    多进程工作函数（需要在顶层定义以便pickle）
    args: (network_state_dict, opponent_state_dict, temperature, num_simulations, game_id)
    """
    import signal
    import os

    # 禁用Intel MKL的Ctrl+C处理（解决Windows多进程forrtl错误）
    os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"

    # 子进程忽略SIGINT，让主进程处理
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    from neural_network import ChessNet
    from config import DEVICE
    import torch

    network_state_dict, opponent_state_dict, temperature, num_simulations, game_id = args

    try:
        # 在子进程中重新创建网络（红方）
        network = ChessNet().to(DEVICE)
        network.load_state_dict(network_state_dict)
        network.eval()

        # 【新增】如果有对手网络，重建对手网络（黑方）
        opponent_network = None
        if opponent_state_dict is not None:
            opponent_network = ChessNet().to(DEVICE)
            opponent_network.load_state_dict(opponent_state_dict)
            opponent_network.eval()

        # 执行对弈
        game_data, winner, end_reason = self_play_game(
            network,
            temperature,
            render=False,
            num_simulations=num_simulations,
            opponent_network=opponent_network
        )

        return game_data, winner, end_reason

    except KeyboardInterrupt:
        # 子进程收到中断，直接退出
        return None, None, None
    except Exception as e:
        # 发生错误，返回None
        print(f"警告: 对局{game_id}失败: {e}")
        return None, None, None


def parallel_self_play(network, num_games, temperature=1.0, num_simulations=None, num_workers=4, opponent_network=None):
    """
    并行自我对弈或对抗训练
    network: 神经网络（红方）
    num_games: 对弈局数
    temperature: 温度参数
    num_simulations: MCTS模拟次数
    num_workers: 并行进程数
    opponent_network: 对手网络（黑方），如果为None则自我对弈
    返回: [(game_data, winner), ...]
    """
    import multiprocessing as mp
    import torch
    import signal
    import sys

    # 获取网络权重（用于在子进程中重建）
    # 重要：将权重移到CPU，避免CUDA多进程共享问题
    network_state_dict = {k: v.cpu() for k, v in network.state_dict().items()}

    # 【新增】如果有对手网络，也获取其权重
    opponent_state_dict = None
    if opponent_network is not None:
        opponent_state_dict = {k: v.cpu() for k, v in opponent_network.state_dict().items()}

    # 准备参数
    args_list = [(network_state_dict, opponent_state_dict, temperature, num_simulations, i)
                 for i in range(num_games)]

    # 使用进程池并行执行
    results = []
    pool = None

    try:
        # 创建进程池时忽略SIGINT（让主进程处理）
        original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        pool = mp.Pool(processes=num_workers)
        signal.signal(signal.SIGINT, original_sigint_handler)

        # 使用imap_unordered实时获取结果
        async_result = pool.imap_unordered(_play_game_worker, args_list)

        # 实时进度显示
        print(f"   进度: [{'':50}] 0/{num_games} (0.0%)", end='', flush=True)

        for i, result in enumerate(async_result, 1):
            # 检查结果是否有效（可能因中断返回None）
            if result is not None and result[0] is not None:
                results.append(result)

            # 实时更新进度条
            progress = i / num_games
            bar_length = 50
            filled = int(bar_length * progress)
            bar = '=' * filled + ' ' * (bar_length - filled)
            percent = progress * 100

            # 计算有效率
            valid_rate = len(results) / i * 100 if i > 0 else 0

            print(f"\r   进度: [{bar}] {i}/{num_games} ({percent:.1f}%) | 有效:{len(results)} ({valid_rate:.0f}%)",
                  end='', flush=True)

        print()  # 换行

        pool.close()
        pool.join()

    except KeyboardInterrupt:
        print("\n\n⚠️  检测到Ctrl+C，正在终止所有子进程...", flush=True)
        print("请稍候（最多10秒）...", flush=True)

        if pool:
            # 强制终止所有子进程
            try:
                pool.terminate()
                pool.join(timeout=10)
            except Exception:
                pass  # 忽略终止过程中的错误

        print("✓ 已停止对弈", flush=True)
        print(f"提示: 已完成 {len(results)} 局对弈，数据将被保存", flush=True)

        # 抛出自定义异常，携带已完成的结果
        raise InterruptedWithResults(results)

    except Exception as e:
        print(f"\n发生错误: {e}")
        if pool:
            pool.terminate()
            pool.join(timeout=5)
        raise

    finally:
        # 确保进程池被关闭
        if pool:
            try:
                pool.close()
            except:
                pass

    return results


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

    game_data, winner, end_reason = self_play_game(network, render=True)

    print(f"\n对局结束！")
    print(f"总步数: {len(game_data)}")
    print(f"胜者: {['和局', '红方', '黑方'][winner]}")
    print(f"结束原因: {end_reason}")
    print(f"收集到 {len(game_data)} 个训练样本")


if __name__ == "__main__":
    test_self_play()
