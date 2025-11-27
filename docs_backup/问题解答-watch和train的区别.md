# 中国象棋AI - 您发现的问题详细解答

## 日期: 2025-11-26

---

## 问题1: watch中每局棋的走法完全相同

### 现象
运行 `python main.py watch` 时,您发现:
- 第1局: 固定的走法序列
- 第2局: 完全相同的走法序列
- 每一步都一样,没有任何变化

### 根本原因

这是因为 **train模式** 和 **watch模式** 使用了不同的走法选择策略:

#### Train模式 (self_play.py:166-184)
```python
# 温度采样 - 增加随机性
if temperature < 0.01:
    move_probs = np.zeros(len(counts))
    move_probs[np.argmax(counts)] = 1
else:
    counts = counts ** (1.0 / temperature)
    move_probs = counts / counts.sum()

# 随机选择
move_idx = np.random.choice(len(moves), p=move_probs)
```

#### Watch模式 (原来的visualizer.py:241)
```python
# 贪婪选择 - 总是选最佳
best_move = max(visit_counts, key=visit_counts.get)
```

### 为什么会一模一样?

因为您的模型刚训练5局,还完全是随机状态,MCTS搜索的访问次数分布总是:
```
move1: 2次
move2: 2次
move3: 2次
...
```

用 `max()` 总是选择第一个2次的走法 → 每次都一样!

### 解决方案 ✅

**已修复!** 现在watch模式也使用温度采样:

```python
# 使用温度采样选择走法(和训练时一样,增加随机性)
temperature = 0.5
counts = counts ** (1.0 / temperature)
move_probs = counts / counts.sum()
move_idx = np.random.choice(len(moves), p=move_probs)
```

现在每局都会有不同的走法了!

---

## 问题2: train和watch的"局"不是一个概念吗?

### 您的发现 ✅ 完全正确!

这两个"局"指的是完全不同的东西:

### Train模式 - 训练对局

```
>> 阶段1: 自我对弈 (100局)...
   对弈 1/100... 和局 (200步)
   对弈 2/100... 红胜 (87步)
```

**这些对局的目的:**
1. 收集训练数据 → 存入经验回放缓冲区
2. 每局结束后,数据用于训练神经网络
3. `self.total_games` 计数器会增加
4. 这些数据会真正影响AI的学习

**代码位置:** `trainer.py:103-144` (collect_self_play_data)

### Watch模式 - 演示对局

```
=== 第 1/5 局 ===
=== 第 2/5 局 ===
```

**这些对局的目的:**
1. 仅用于可视化展示
2. 不存储数据
3. 不参与训练
4. `self.total_games` 不会增加
5. 只是让您看看AI当前水平

**代码位置:** `visualizer.py:193-275` (watch_game)

### 对比表格

| 特性 | Train的"局" | Watch的"局" |
|------|------------|-------------|
| 目的 | 收集训练数据 | 可视化展示 |
| 数据存储 | 存入replay_buffer | 不存储 |
| 影响训练 | 是 | 否 |
| 计数增加 | total_games +1 | 不增加 |
| 走法选择 | 温度采样(随机) | 原来贪婪,现已改为温度采样 |
| 使用MCTS | 是(50次模拟) | 是(50次模拟) |

### 为什么watch结束后train不增加?

因为watch只是"观看",就像看录像一样,不会影响训练进度!

---

## 问题3: plot_progress.py没有图像

### 错误信息
```
日志文件不存在: logs\training.log
无法读取训练数据。请先运行训练程序生成日志。
```

### 根本原因

**原来的设计:** `training.log` 只在"评估"时创建

查看 trainer.py:207-213:
```python
def evaluate(self):
    """评估当前模型棋力"""
    # ... 评估代码 ...

    # 保存评估结果到日志
    log_file = f"{LOG_DIR}/training.log"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(...)  # 只有这里写日志!
```

查看 trainer.py:96-98:
```python
# 4. 评估棋力
if iteration % EVALUATE_INTERVAL == 0:  # EVALUATE_INTERVAL = 5
    self.evaluate()
```

**问题:**
- 默认配置: 每5轮才评估一次
- 您才训练第1-2轮,还没到第5轮
- 所以 `training.log` 还没创建!

### 解决方案 ✅

**已修复!** 现在每轮都记录进度:

1. 添加了 `_log_progress()` 方法 (trainer.py:218-225):
```python
def _log_progress(self, iteration):
    """记录训练进度(每轮都记录)"""
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = f"{LOG_DIR}/training.log"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now()} | 轮次:{iteration} | "
               f"总局数:{self.total_games} | "
               f"缓冲区:{len(self.replay_buffer)} | 类型:训练\n")
```

2. 每轮结束都调用 (trainer.py:104):
```python
# 记录训练进度到日志(每轮都记录)
self._log_progress(iteration)
```

现在从第1轮开始就有日志了!

### 日志格式

现在 `logs/training.log` 包含两种记录:

```
2025-11-26 14:05:23 | 轮次:1 | 总局数:100 | 缓冲区:5000 | 类型:训练
2025-11-26 14:10:45 | 轮次:2 | 总局数:200 | 缓冲区:8000 | 类型:训练
...
2025-11-26 15:00:12 | 总局数:500 | 红方胜率:48.0% | 平均步数:132.5 | 类型:评估
```

---

## 总结: 已完成的修复

### ✅ 修复1: Watch模式添加随机性
- **文件:** visualizer.py
- **修改:** 从贪婪选择改为温度采样
- **效果:** 每局对局都会有不同的走法

### ✅ 修复2: 从第1轮就创建训练日志
- **文件:** trainer.py
- **添加:** `_log_progress()` 方法
- **效果:** `plot_progress.py` 从第1轮就能工作

### 📚 澄清: Train vs Watch的"局"
- Train的局 = 训练数据收集
- Watch的局 = 可视化展示
- 它们是完全独立的概念

---

## 下一步使用建议

### 1. 测试修复效果

```bash
# 运行训练(会在第1轮就创建logs/training.log)
python main.py train

# 等待第1轮完成后,查看进度图
python plot_progress.py

# 观看对局(现在每局都不一样了)
python main.py watch
```

### 2. 观察训练进度

现在您可以:
- 从第1轮开始看到进度曲线
- Watch模式看到不同的对局
- 理解train和watch的区别

### 3. 预期效果

- **前100局:** 完全随机,没有章法
- **100-500局:** 开始学会基本走法
- **500-2000局:** 学会吃子
- **2000+局:** 出现简单战术

---

## 技术细节解释

### 温度参数 (Temperature)

```python
# 高温度 (1.0) - 更随机
temperature = 1.0
counts = counts ** (1.0 / 1.0)  # 不变
# 结果: 访问次数[10, 8, 2] → 概率[0.50, 0.40, 0.10]

# 低温度 (0.5) - 更倾向最佳
temperature = 0.5
counts = counts ** (1.0 / 0.5)  # 平方
# 结果: 访问次数[10, 8, 2] → [100, 64, 4] → 概率[0.60, 0.38, 0.02]

# 极低温度 (0.1) - 几乎贪婪
temperature = 0.1
counts = counts ** (1.0 / 0.1)  # 10次方
# 结果: 几乎总是选访问次数最多的
```

### 为什么训练用高温度?

- **前期(500局内):** temperature=1.0 → 充分探索
- **后期(500局后):** temperature=0.5 → 更多利用已学知识
- **观看模式:** temperature=0.5 → 平衡展示效果和变化性

---

希望这份文档完全解答了您的疑问! 🎉

如有其他问题,欢迎继续提问。
