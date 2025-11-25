"""
神经网络模型 - AI的"大脑"
输入：棋盘状态
输出：1. 每个走法的概率（策略） 2. 当前局面的胜率（价值）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import BOARD_SIZE, BOARD_WIDTH, DEVICE

class ChessNet(nn.Module):
    """
    简化版AlphaZero网络
    使用卷积神经网络提取棋盘特征
    """
    def __init__(self, num_channels=128):
        super(ChessNet, self).__init__()

        # 输入层：棋盘编码
        # 输入维度: [batch, 15, 10, 9]
        # 15个通道：红方7种棋子 + 黑方7种棋子 + 当前玩家标记

        # 特征提取层（多层卷积）
        self.conv1 = nn.Conv2d(15, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)

        # 残差块（提取更深层特征）
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(4)
        ])

        # 策略头（输出走法概率）
        self.policy_conv = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        # 输出所有可能走法的概率 (10*9 * 10*9 = 8100个可能)
        # 简化：只输出 to_position (10*9=90)
        self.policy_fc = nn.Linear(32 * BOARD_SIZE * BOARD_WIDTH,
                                   BOARD_SIZE * BOARD_WIDTH * 90)

        # 价值头（输出局面评分 -1到1）
        self.value_conv = nn.Conv2d(num_channels, 8, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(8)
        self.value_fc1 = nn.Linear(8 * BOARD_SIZE * BOARD_WIDTH, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x):
        """
        前向传播
        x: [batch, 15, 10, 9] 棋盘状态
        返回: (策略logits, 价值)
        """
        # 特征提取
        x = F.relu(self.bn1(self.conv1(x)))

        # 残差块
        for res_block in self.res_blocks:
            x = res_block(x)

        # 策略头
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)

        # 价值头
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))  # 输出 -1 到 1

        return policy, value

    def predict(self, board, current_player, legal_moves):
        """
        预测单个棋盘状态
        board: numpy array [10, 9]
        current_player: 1 or -1
        legal_moves: [(from_r, from_c, to_r, to_c), ...]
        返回: (move_probs, value)
        """
        # 编码棋盘
        state = self.encode_board(board, current_player)
        state = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            policy_logits, value = self.forward(state)

        # 转换为走法概率
        move_probs = self._logits_to_move_probs(
            policy_logits[0].cpu().numpy(),
            legal_moves
        )

        return move_probs, value.item()

    def encode_board(self, board, current_player):
        """
        将棋盘编码为神经网络输入
        返回: [15, 10, 9] numpy array
        """
        encoded = np.zeros((15, BOARD_SIZE, BOARD_WIDTH), dtype=np.float32)

        # 前7个通道：红方棋子
        for i in range(1, 8):
            encoded[i-1] = (board == i).astype(np.float32)

        # 中间7个通道：黑方棋子
        for i in range(1, 8):
            encoded[i+6] = (board == -i).astype(np.float32)

        # 最后1个通道：当前玩家（全1或全0）
        encoded[14] = np.ones((BOARD_SIZE, BOARD_WIDTH)) * (current_player == 1)

        return encoded

    def _logits_to_move_probs(self, logits, legal_moves):
        """
        将网络输出转换为合法走法的概率分布
        """
        if len(legal_moves) == 0:
            return {}

        # 简化版：只看目标位置
        move_probs = {}
        for move in legal_moves:
            from_r, from_c, to_r, to_c = move
            # 索引：from_position * 90 + to_position
            idx = (from_r * BOARD_WIDTH + from_c) * 90 + (to_r * BOARD_WIDTH + to_c)
            if idx < len(logits):
                move_probs[move] = logits[idx]

        # Softmax归一化
        probs = np.array(list(move_probs.values()))
        probs = np.exp(probs - np.max(probs))
        probs = probs / np.sum(probs)

        return {move: prob for move, prob in zip(move_probs.keys(), probs)}


class ResidualBlock(nn.Module):
    """残差块 - 帮助网络学习更深层特征"""
    def __init__(self, num_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # 残差连接
        out = F.relu(out)
        return out


def test_network():
    """测试网络是否能正常运行"""
    print("测试神经网络...")
    net = ChessNet().to(DEVICE)

    # 创建随机输入
    batch_size = 4
    x = torch.randn(batch_size, 15, BOARD_SIZE, BOARD_WIDTH).to(DEVICE)

    policy, value = net(x)
    print(f"输入形状: {x.shape}")
    print(f"策略输出形状: {policy.shape}")
    print(f"价值输出形状: {value.shape}")
    print(f"价值范围: {value.min().item():.2f} ~ {value.max().item():.2f}")

    # 计算参数量
    total_params = sum(p.numel() for p in net.parameters())
    print(f"总参数量: {total_params:,}")
    print("网络测试通过！")


if __name__ == "__main__":
    test_network()
