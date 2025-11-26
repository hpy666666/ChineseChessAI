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

from neural_network import ChessNet
from self_play import self_play_game
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

            # 1. 自我对弈收集数据
            print(f"\n>> 阶段1: 自我对弈 ({SELF_PLAY_GAMES}局)...")
            self.collect_self_play_data(SELF_PLAY_GAMES)

            # 2. 训练神经网络
            if len(self.replay_buffer) >= BATCH_SIZE:
                print(f"\n>> 阶段2: 训练神经网络...")
                avg_loss = self.train_network()
                print(f"   平均损失: {avg_loss:.4f}")

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

            # 记录训练进度到日志(每轮都记录)
            self._log_progress(iteration)

    def collect_self_play_data(self, num_games):
        """收集自我对弈数据"""
        self.network.eval()

        red_wins = 0
        black_wins = 0
        draws = 0
        total_moves = 0

        for i in range(num_games):
            # 前期用较高温度增加探索
            temperature = 1.0 if self.total_games < 500 else 0.5

            print(f"   对弈 {i+1}/{num_games}...", end='', flush=True)

            game_data, winner = self_play_game(self.network, temperature=temperature)

            self.replay_buffer.push(game_data)
            self.total_games += 1
            total_moves += len(game_data)

            if winner == 1:
                red_wins += 1
            elif winner == -1:
                black_wins += 1
            else:
                draws += 1

            # 每局都显示结果
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

    def train_network(self):
        """训练神经网络"""
        self.network.train()

        total_loss = 0
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
            self.optimizer.step()

            total_loss += loss.item()
            self.training_steps += 1

        return total_loss / num_batches

    def evaluate(self):
        """评估当前模型棋力"""
        print("   正在评估...")

        # 简单评估：让AI下10局，看看表现
        self.network.eval()
        test_games = 10

        red_wins = 0
        avg_moves = 0

        for _ in range(test_games):
            game_data, winner = self_play_game(self.network, temperature=0.1)
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

    def _log_progress(self, iteration):
        """记录训练进度(每轮都记录)"""
        os.makedirs(LOG_DIR, exist_ok=True)
        log_file = f"{LOG_DIR}/training.log"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now()} | 轮次:{iteration} | "
                   f"总局数:{self.total_games} | "
                   f"缓冲区:{len(self.replay_buffer)} | 类型:训练\n")

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


def main():
    """主函数"""
    trainer = Trainer()

    # 开始训练
    trainer.train_loop(num_iterations=100)


if __name__ == "__main__":
    main()
