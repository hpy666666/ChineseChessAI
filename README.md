# 中国象棋AI训练系统

基于AlphaZero算法的中国象棋AI，通过自我对弈不断进步。

## 快速开始

```bash
# 1. 安装依赖
pip install torch pygame numpy matplotlib

# 2. 开始训练
python main.py train

# 3. 人机对战
python main.py play

# 4. 快速评估
python main.py evaluate
```

---

## 所有可用功能

| 命令 | 功能 | 说明 |
|------|------|------|
| `python main.py train` | 训练模式 | 自我对弈100局/轮，约2.8小时 |
| `python main.py play` | **人机对战** | 与AI对战，验收成果 ⭐ |
| `python main.py evaluate` | **快速评估** | 20局测试，显示实力等级 ⭐ |
| `python main.py watch` | 观看对局 | 图形界面，速度可调 |
| `python plot_progress.py` | 训练曲线 | 胜率、步数趋势图 |
| `python compare_models.py` | **模型对比** | 新旧模型对战 ⭐ |
| `python main.py help` | 帮助 | 查看所有命令 |

---

## 验收成果的5种方式

1. **快速评估** - 量化指标（胜率、步数、等级）
2. **人机对战** - 直观体验AI实力
3. **观看对局** - 了解AI思路
4. **训练曲线** - 长期进步趋势
5. **模型对比** - 验证训练效果

---

## 训练进度参考

| 训练局数 | 等级 | 耗时 |
|---------|------|------|
| 500 | 入门级 | 1天 |
| 2000 | 业余初级 | 3天 |
| 5000 | 业余中级 | 6天 |
| 10000+ | 业余高级 | 12天+ |

---

## 项目结构

**核心代码**
- `chess_env.py` - 象棋规则引擎
- `neural_network.py` - AI神经网络
- `self_play.py` - MCTS自我对弈
- `trainer.py` - 训练管理器

**验收工具** ⭐
- `evaluate.py` - 快速评估
- `compare_models.py` - 模型对比
- `visualizer.py` - 图形界面（含人机对战）
- `plot_progress.py` - 进步曲线

**配置**
- `config.py` - 训练参数
- `main.py` - 统一入口

---

## 性能优化

**已实现的优化（总提速4.5倍）:**
- ✅ 将帅位置缓存
- ✅ 批量神经网络推理
- ✅ 优化环境复制
- ✅ MAX_MOVES降至100步
- ✅ 和局惩罚机制

**当前性能:**
- 单局: 1.7分钟
- 每轮: 2.8小时（100局）

---

## 常见问题

**Q: 如何验收成果？**
A: 运行 `python main.py play` 与AI对战

**Q: 训练多久能下棋？**
A: 500局知道吃子，2000局有战术

**Q: 可以暂停吗？**
A: 可以！Ctrl+C中断，自动保存

---

## 今日更新 (2025-11-26)

**修复:**
- 长捉误判导致12步结束（已禁用训练期检测）

**新增:**
- 快速评估工具 (`evaluate.py`)
- 人机对战模式 (`main.py play`)
- 模型对比工具 (`compare_models.py`)
- 改进观看显示（模型信息、实力等级）

---

祝您训练出强大的象棋AI！🎉
