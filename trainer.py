"""
训练器 - 管理整个训练流程
1. 收集自我对弈数据
2. 训练神经网络
3. 保存模型
4. 跟踪训练进度
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import os
from collections import deque
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from neural_network import ChessNet
from self_play import self_play_game, parallel_self_play, InterruptedWithResults
from config import *

class ReplayBuffer:
    """经验回放缓冲区 - 存储历史对局数据"""
    def __init__(self, max_size=BUFFER_SIZE):
        self.buffer = deque(maxlen=max_size)

    def push(self, game_data):
        """
        添加一局游戏数据
        game_data: [(board, move_probs, reward), ...]
        """
        for data in game_data:
            self.buffer.append(data)

    def sample(self, batch_size):
        """随机采样一批数据"""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]

        boards, move_probs_list, rewards = zip(*batch)
        return boards, move_probs_list, rewards

    def __len__(self):
        return len(self.buffer)


class Trainer:
    """训练管理器"""
    def __init__(self):
        self.network = ChessNet().to(DEVICE)
        self.optimizer = optim.Adam(self.network.parameters(), lr=LEARNING_RATE)
        self.replay_buffer = ReplayBuffer()

        self.total_games = 0
        self.training_steps = 0

        # 【新增】加载对手网络（旧模型）
        self.opponent_network = None
        opponent_model_path = f"{MODEL_DIR}/old_opponent.pt"
        if os.path.exists(opponent_model_path):
            self.opponent_network = ChessNet().to(DEVICE)
            checkpoint = torch.load(opponent_model_path, map_location=DEVICE)
            self.opponent_network.load_state_dict(checkpoint['model_state_dict'])
            self.opponent_network.eval()  # 对手网络只用于推理，不训练
            print(f"[对抗训练] 已加载对手模型: {opponent_model_path}")
            print(f"[对抗训练] 训练模式: 50%自我对弈 + 50%对抗旧模型")
        else:
            print(f"[自我对弈] 未找到对手模型，使用纯自我对弈模式")

        # 初始化 TensorBoard
        tensorboard_dir = f"{LOG_DIR}/tensorboard"
        self.writer = SummaryWriter(tensorboard_dir)
        print(f"TensorBoard 日志目录: {tensorboard_dir}")
        print(f"启动 TensorBoard: tensorboard --logdir={tensorboard_dir}")

        # 加载已有模型（如果存在）
        if os.path.exists(LATEST_MODEL):
            self.load_model()
            print(f"加载模型: {LATEST_MODEL}")

    def train_loop(self, num_iterations=1000):
        """
        主训练循环
        num_iterations: 训练多少轮
        """
        print("=" * 60)
        print("开始训练中国象棋AI")
        print("=" * 60)
        print(f"设备: {DEVICE}")
        print(f"每轮自我对弈: {SELF_PLAY_GAMES}局")
        print(f"MCTS模拟次数: {MCTS_SIMULATIONS}")
        print(f"缓冲区大小: {BUFFER_SIZE}")
        print("=" * 60)

        for iteration in range(1, num_iterations + 1):
            print(f"\n【第 {iteration}/{num_iterations} 轮训练】")
            print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            try:
                # 1. 自我对弈收集数据
                print(f"\n>> 阶段1: 自我对弈 ({SELF_PLAY_GAMES}局)...")
                stats = self.collect_self_play_data(SELF_PLAY_GAMES)

                # 2. 训练神经网络
                if len(self.replay_buffer) >= BATCH_SIZE:
                    print(f"\n>> 阶段2: 训练神经网络...")
                    avg_loss = self.train_network()
                    current_lr = self.optimizer.param_groups[0]['lr']
                    print(f"   平均损失: {avg_loss:.4f} | 学习率: {current_lr:.6f}")

                # 3. 保存模型
                if iteration % SAVE_INTERVAL == 0:
                    print(f"\n>> 阶段3: 保存模型...")
                    self.save_model()
                    print(f"   模型已保存到: {LATEST_MODEL}")

                # 4. 评估棋力
                if iteration % EVALUATE_INTERVAL == 0:
                    print(f"\n>> 阶段4: 评估棋力...")
                    self.evaluate()

                print(f"\n进度: 总对局数={self.total_games}, 缓冲区={len(self.replay_buffer)}")
                print("-" * 60)

                # 记录训练进度到日志(每轮都记录,包含统计数据)
                self._log_progress(iteration, stats)

            except KeyboardInterrupt:
                # 用户中断：先用已有数据训练并保存，再退出
                print("\n>> 训练被中断，处理已收集的数据...")

                # 如果有足够数据，进行训练
                if len(self.replay_buffer) >= BATCH_SIZE:
                    print(f"\n>> 阶段2: 训练神经网络...")
                    avg_loss = self.train_network()
                    print(f"   平均损失: {avg_loss:.4f}")

                # 保存模型
                print(f"\n>> 保存模型...")
                self.save_model()
                print(f"   模型已保存到: {LATEST_MODEL}")
                print(f"   总对局数: {self.total_games}")

                # 重新抛出异常，让外层知道训练被中断
                raise

    def collect_self_play_data(self, num_games):
        """收集自我对弈数据（多进程优化版，支持对抗训练）"""
        self.network.eval()

        # 动态调整MCTS模拟次数
        num_simulations = get_dynamic_mcts_simulations(self.total_games)
        print(f"   当前训练局数: {self.total_games}, MCTS模拟次数: {num_simulations}")

        # 【新增】如果有对手网络，混合训练：50%自我对弈 + 50%对抗
        if self.opponent_network:
            num_self_play = num_games // 2
            num_vs_opponent = num_games - num_self_play
            print(f"   训练模式: {num_self_play}局自我对弈 + {num_vs_opponent}局对抗旧模型")
        else:
            num_self_play = num_games
            num_vs_opponent = 0
            print(f"   训练模式: {num_self_play}局自我对弈")

        # 前期用较高温度增加探索
        temperature = 1.0 if self.total_games < 500 else 0.5

        # 统计结果
        red_wins = 0
        black_wins = 0
        draws = 0
        total_moves = 0

        # 精彩对局记录（用于保存）
        best_games = []

        # 根据配置选择单进程或多进程
        if USE_MULTIPROCESSING:
            print(f"   使用{NUM_WORKERS}进程并行对弈...")
            print(f"   提示: 按Ctrl+C可以中断训练")

            results = []
            interrupted = False
            try:
                # 【修改】先进行自我对弈
                if num_self_play > 0:
                    print(f"\n   >> 自我对弈阶段 ({num_self_play}局)...")
                    self_play_results = parallel_self_play(
                        self.network,
                        num_games=num_self_play,
                        temperature=temperature,
                        num_simulations=num_simulations,
                        num_workers=NUM_WORKERS,
                        opponent_network=None  # 自我对弈
                    )
                    results.extend(self_play_results)

                # 【新增】对抗旧模型训练
                if num_vs_opponent > 0:
                    print(f"\n   >> 对抗训练阶段 ({num_vs_opponent}局，红方新模型 vs 黑方旧模型)...")
                    vs_results = parallel_self_play(
                        self.network,
                        num_games=num_vs_opponent,
                        temperature=temperature,
                        num_simulations=num_simulations,
                        num_workers=NUM_WORKERS,
                        opponent_network=self.opponent_network  # 对抗旧模型
                    )
                    results.extend(vs_results)

            except InterruptedWithResults as e:
                # 捕获自定义异常，获取已完成的结果
                print("\n对弈被中断")
                results = e.results
                interrupted = True
            except KeyboardInterrupt:
                # 兼容直接的 KeyboardInterrupt（不应该发生，但以防万一）
                print("\n对弈被中断")
                interrupted = True

            # 处理已完成的数据（无论正常完成还是中断）
            print(f"\n   统计结果...")
            for game_data, winner, end_reason in results:
                self.replay_buffer.push(game_data)
                self.total_games += 1
                total_moves += len(game_data)
                moves_count = len(game_data)

                if winner == 1:
                    red_wins += 1
                elif winner == -1:
                    black_wins += 1
                else:
                    draws += 1

                # 收集精彩对局（有胜负的、或步数少的）
                if winner != 0:  # 有胜负
                    best_games.append((game_data, winner, moves_count, end_reason))
                elif moves_count < 50:  # 步数少的和局也可能精彩
                    best_games.append((game_data, winner, moves_count, end_reason))

            # 如果被中断，处理完数据后抛出异常让训练停止
            if interrupted:
                print(f"   [中断] 完成了 {len(results)}/{num_games} 局对弈")
                print(f"   统计: 红方{red_wins}胜 黑方{black_wins}胜 {draws}和")
                raise KeyboardInterrupt
        else:
            # 单进程模式（显示详细进度）
            print(f"   使用单进程对弈（可在config.py启用多进程加速）...")

            # 自我对弈
            for i in range(num_self_play):
                print(f"   [自我对弈] {i+1}/{num_self_play}...", end='', flush=True)
                game_data, winner, end_reason = self_play_game(
                    self.network,
                    temperature=temperature,
                    num_simulations=num_simulations,
                    opponent_network=None
                )
                self.replay_buffer.push(game_data)
                self.total_games += 1
                total_moves += len(game_data)

                if winner == 1:
                    red_wins += 1
                elif winner == -1:
                    black_wins += 1
                else:
                    draws += 1

                # 显示结果
                result = "红胜" if winner == 1 else "黑胜" if winner == -1 else "和局"
                print(f" {result} ({len(game_data)}步)")

                # 每10局显示统计
                if (i + 1) % 10 == 0:
                    avg_moves = total_moves / (i + 1)
                    print(f"   进度: {i+1}/{num_games} | "
                          f"红胜:{red_wins} 黑胜:{black_wins} 和:{draws} | "
                          f"平均步数: {avg_moves:.1f}")

        print(f"   [完成] {num_games} 局对弈")
        print(f"   统计: 红方{red_wins}胜 黑方{black_wins}胜 {draws}和")

        # 保存精彩对局
        if best_games:
            self._save_best_games(best_games)

        # 返回统计数据
        avg_moves = total_moves / num_games if num_games > 0 else 0
        return {
            'red_wins': red_wins,
            'black_wins': black_wins,
            'draws': draws,
            'avg_moves': avg_moves
        }

    def train_network(self):
        """训练神经网络"""
        self.network.train()

        # 动态调整学习率
        new_lr = get_dynamic_learning_rate(self.total_games)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        total_loss = 0
        total_value_loss = 0
        num_batches = min(50, len(self.replay_buffer) // BATCH_SIZE)

        for _ in range(num_batches):
            # 采样数据
            boards, move_probs_list, rewards = self.replay_buffer.sample(BATCH_SIZE)

            # 转换为tensor
            states = torch.stack([
                torch.FloatTensor(self.network.encode_board(board, 1))
                for board in boards
            ]).to(DEVICE)

            target_values = torch.FloatTensor(rewards).unsqueeze(1).to(DEVICE)

            # 前向传播
            policy_logits, pred_values = self.network(states)

            # 计算损失
            value_loss = nn.MSELoss()(pred_values, target_values)

            # 策略损失（简化版：只用价值损失）
            # 完整版需要把move_probs转换为target policy
            loss = value_loss

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            total_value_loss += value_loss.item()
            self.training_steps += 1

        avg_loss = total_loss / num_batches
        avg_value_loss = total_value_loss / num_batches

        # TensorBoard 记录
        self.writer.add_scalar('Loss/total', avg_loss, self.total_games)
        self.writer.add_scalar('Loss/value', avg_value_loss, self.total_games)
        self.writer.add_scalar('Training/learning_rate', new_lr, self.total_games)
        self.writer.add_scalar('Training/buffer_size', len(self.replay_buffer), self.total_games)

        # 记录网络权重分布（每100步）
        if self.training_steps % 100 == 0:
            for name, param in self.network.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self.writer.add_histogram(f'Weights/{name}', param.data, self.training_steps)
                    self.writer.add_histogram(f'Gradients/{name}', param.grad, self.training_steps)

        return avg_loss

    def evaluate(self):
        """评估当前模型棋力"""
        print("   正在评估...")

        # 简单评估：让AI下10局，看看表现
        self.network.eval()
        test_games = 10

        red_wins = 0
        avg_moves = 0

        for _ in range(test_games):
            game_data, winner, end_reason = self_play_game(self.network, temperature=0.1)
            if winner == 1:
                red_wins += 1
            avg_moves += len(game_data)

        avg_moves /= test_games

        print(f"   测试结果 ({test_games}局):")
        print(f"   - 红方胜率: {red_wins/test_games*100:.1f}%")
        print(f"   - 平均步数: {avg_moves:.1f}")

        # 保存评估结果到日志
        os.makedirs(LOG_DIR, exist_ok=True)
        log_file = f"{LOG_DIR}/training.log"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now()} | 总局数:{self.total_games} | "
                   f"红方胜率:{red_wins/test_games*100:.1f}% | "
                   f"平均步数:{avg_moves:.1f} | 类型:评估\n")

    def _log_progress(self, iteration, stats=None):
        """记录训练进度(每轮都记录)"""
        os.makedirs(LOG_DIR, exist_ok=True)
        log_file = f"{LOG_DIR}/training.log"
        with open(log_file, "a", encoding="utf-8") as f:
            if stats:
                # 记录详细统计数据
                f.write(f"{datetime.now()} | 轮次:{iteration} | "
                       f"总局数:{self.total_games} | "
                       f"红胜:{stats['red_wins']} 黑胜:{stats['black_wins']} 和:{stats['draws']} | "
                       f"平均步数:{stats['avg_moves']:.1f} | "
                       f"缓冲区:{len(self.replay_buffer)} | 类型:训练\n")

                # TensorBoard 记录对局统计
                total_games_in_iteration = stats['red_wins'] + stats['black_wins'] + stats['draws']
                if total_games_in_iteration > 0:
                    self.writer.add_scalar('Games/red_win_rate',
                                          stats['red_wins'] / total_games_in_iteration,
                                          self.total_games)
                    self.writer.add_scalar('Games/black_win_rate',
                                          stats['black_wins'] / total_games_in_iteration,
                                          self.total_games)
                    self.writer.add_scalar('Games/draw_rate',
                                          stats['draws'] / total_games_in_iteration,
                                          self.total_games)
                    self.writer.add_scalar('Games/avg_moves', stats['avg_moves'], self.total_games)

                    # 添加胜负比饼图（使用文本替代，因为TensorBoard不直接支持饼图）
                    win_loss_text = f"Red: {stats['red_wins']}, Black: {stats['black_wins']}, Draw: {stats['draws']}"
                    self.writer.add_text('Games/win_loss_distribution', win_loss_text, self.total_games)
            else:
                # 兼容旧格式
                f.write(f"{datetime.now()} | 轮次:{iteration} | "
                       f"总局数:{self.total_games} | "
                       f"缓冲区:{len(self.replay_buffer)} | 类型:训练\n")

        # 刷新 TensorBoard writer
        self.writer.flush()

    def save_model(self):
        """保存模型"""
        os.makedirs(MODEL_DIR, exist_ok=True)

        checkpoint = {
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_games': self.total_games,
            'training_steps': self.training_steps,
        }

        torch.save(checkpoint, LATEST_MODEL)

        # 每100轮保存一个备份
        if self.total_games % 1000 == 0:
            backup_path = f"{MODEL_DIR}/model_{self.total_games}.pt"
            torch.save(checkpoint, backup_path)

    def load_model(self):
        """加载模型"""
        checkpoint = torch.load(LATEST_MODEL, map_location=DEVICE)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.total_games = checkpoint.get('total_games', 0)
        self.training_steps = checkpoint.get('training_steps', 0)

        print(f"已加载模型，已训练 {self.total_games} 局")

    def close(self):
        """关闭训练器并清理资源"""
        if hasattr(self, 'writer'):
            self.writer.close()
            print("TensorBoard writer 已关闭")

    def _save_best_games(self, best_games):
        """保存精彩对局到文件"""
        if not best_games:
            return

        os.makedirs(DATA_DIR, exist_ok=True)
        games_file = f"{DATA_DIR}/best_games.pkl"

        # 加载已有的精彩对局
        existing_games = []
        if os.path.exists(games_file):
            try:
                with open(games_file, 'rb') as f:
                    existing_games = pickle.load(f)
            except:
                existing_games = []

        # 添加新对局（带时间戳）
        from datetime import datetime
        for game_data, winner, moves, game_type in best_games:
            existing_games.append({
                'timestamp': datetime.now(),
                'total_games': self.total_games,
                'game_data': game_data,
                'winner': winner,
                'moves': moves,
                'type': game_type
            })

        # 只保留最近500局精彩对局
        existing_games = existing_games[-500:]

        # 保存
        with open(games_file, 'wb') as f:
            pickle.dump(existing_games, f)

        # 统计信息
        wins = sum(1 for g in best_games if g[1] != 0)
        print(f"   [保存] {len(best_games)} 局精彩对局 (有胜负:{wins}, 总计:{len(existing_games)})")


def main():
    """主函数"""
    trainer = Trainer()

    # 开始训练
    trainer.train_loop(num_iterations=100)


if __name__ == "__main__":
    main()
