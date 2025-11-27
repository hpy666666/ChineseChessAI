# 🐛 关键Bug修复：胜利判断被100步限制覆盖

## 问题描述

用户报告：人机对战中，明确获胜（例如用炮吃掉对方将），但系统显示"和局"。

## 根本原因

在 `chess_env.py` 的 `make_move()` 函数中，存在**逻辑顺序错误**：

```python
# 1. 吃掉将/帅 -> 立即获胜
if abs(captured) == abs(PIECES['R_KING']):
    self.winner = self.current_player  # 设置winner=1（红方）
    reward = 100
    done = True

# ... 其他判断 ...

# 切换玩家
self.current_player *= -1
self.move_count += 1

# ❌ BUG: 超过最大步数判和（没有检查done标志！）
if self.move_count >= 100:
    done = True
    reward = 0
    self.winner = 0  # 覆盖了之前的winner=1！
```

**问题流程：**
1. 第99步，红方用炮吃黑将
2. 第286-289行：正确设置 `winner=1`, `done=True`
3. 第357-358行：切换玩家，`move_count` 增加到 100
4. **第361-364行：检查到 `move_count >= 100`，直接覆盖 `winner=0`**
5. 结果：获胜变成了和局！

## 修复方案

在第361行的条件判断中，增加 `not done` 检查：

```python
# ✅ 修复后：只在未分出胜负时，才判和
if not done and self.move_count >= 100:
    done = True
    reward = 0
    self.winner = 0
```

**修复逻辑：**
- 如果已经设置了 `done=True`（吃将、将死等），就不再覆盖 `winner`
- 只有在 `done=False` 时，才因为100步限制判和

## 测试验证

### 测试1：第99步吃将
```python
env.move_count = 99
env.make_move((1, 4, 0, 4))  # 红车吃黑将

# 结果：
# move_count: 100
# done: True
# winner: 1  ✅ 正确：红方获胜
```

### 测试2：第99步普通走棋
```python
env.move_count = 99
env.make_move((5, 4, 5, 5))  # 红车移动（不吃子）

# 结果：
# move_count: 100
# done: True
# winner: 0  ✅ 正确：100步判和
```

## 影响范围

### 修复前
- ❌ 人机对战：第100步附近获胜会变成和局
- ❌ 训练：第100步的胜局会被错误记录为和局
- ❌ 模型学习：无法正确学习"在接近100步时也要争取获胜"

### 修复后
- ✅ 人机对战：任何时候吃将都会正确判定获胜
- ✅ 训练：胜负判断完全正确
- ✅ 模型学习：能正确区分"获胜"和"和局"

## 修改文件

1. **chess_env.py** (第361行)
   - 添加 `not done` 条件检查

2. **visualizer.py** (第358行) - 附带优化
   - 移除冗余的 `len(env.get_legal_moves()) == 0` 检查
   - 只保留100步限制检查

## 其他相关修复

在同一次排查中，还修复了：
1. ✅ 缺少"将死"判断（已在之前修复）
2. ✅ 训练日志格式问题（已在之前修复）
3. ✅ 模型对比命令说明（已在之前修复）

---

## 使用建议

### 立即验证
```bash
# 重新训练，胜负判断现在完全正确
python main.py train

# 人机对战验收（尝试吃将获胜）
python main.py play
```

### 预期效果
- 人机对战中吃掉对方将，会显示"红方获胜！恭喜您赢了！"
- 训练中的胜负比例会更合理（随着AI水平提升）
- 不会再出现"明明赢了却显示和局"的情况

---

**修复时间：** 2025-11-26
**影响级别：** ⚠️ 严重（影响所有胜负判断）
**修复状态：** ✅ 已完成并测试通过
