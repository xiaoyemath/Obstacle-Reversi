# Obstacle Reversi

Obstacle Reversi 是一个带随机障碍物的黑白棋变体。项目包含完整的游戏环境、桌面端、Web 端，以及一套自对弈训练流程。

棋盘规则基于 Reversi / Othello：玩家落子后沿 8 个方向翻转被夹住的对方棋子。不同之处是棋盘上会生成障碍物，射线遇到障碍物会被阻断。

## 功能概览

- 障碍物黑白棋环境，兼容 Gymnasium API
- Pygame 桌面人机对弈
- FastAPI 后端 + HTML Web 客户端
- AlphaZero 风格神经网络：策略头 + 价值头
- MCTS 搜索，支持时间预算和持久化搜索树
- 自对弈数据生成、D4 对称增强、经验回放和检查点保存
- 支持通过 `MODEL_PATH` 加载训练好的 `.pth` 模型

没有训练模型时也可以运行，AI 会使用随机初始化的网络，因此棋力会比较随缘。

## 快速开始

### 1. 安装依赖

```bash
pip install gymnasium numpy torch fastapi uvicorn pygame
```

### 2. 启动 Web 版

```bash
cd /Users/yeliqin/Thegame
python server.py
```

然后打开：

```text
http://127.0.0.1:8000/
```

也可以直接打开 `index.html`，但后端服务仍然必须在 `127.0.0.1:8000` 运行，否则页面会请求不到 API。

### 3. 启动桌面版

```bash
python main.py
```

桌面版快捷键：

- `R`：重新开始
- `Esc`：退出

### 4. 加载训练模型

```bash
MODEL_PATH=checkpoints/iter_099.pth python server.py
MODEL_PATH=checkpoints/iter_099.pth python main.py
```

检查点里会保存棋盘大小、通道数和残差块数量。加载时程序会自动按检查点架构创建网络。

## 游戏规则

- 默认棋盘为 `8 x 8`
- `-1` 表示黑子，`1` 表示白子，`0` 表示空格，`2` 表示障碍物
- 初始中心 4 子按经典黑白棋摆放
- 障碍物随机生成，默认占剩余空格约 10%
- 落子必须至少翻转一枚对方棋子
- 射线遇到边界、空格或障碍物会停止
- 当前玩家没有合法走法时必须 Pass
- 双方都没有合法走法时游戏结束
- 终局按黑白棋子数量判胜负，数量相同则平局

Web 版在终局时会显示类似 `Game Over - Black wins by 2` 的提示；如果只是当前玩家无棋可下，会显示 `Black has no legal moves - Pass`。

## 项目结构

```text
.
├── obstacle_reversi_env.py   # 游戏环境、规则、合法步判断、Pygame 渲染
├── alpha_reversi_net.py      # AlphaZero 风格双头 CNN
├── mcts.py                   # MCTS / PUCT 搜索
├── self_play.py              # 自对弈与经验回放
├── trainer.py                # 训练循环、loss、优化器、检查点保存
├── train.py                  # 命令行训练入口
├── server.py                 # FastAPI 后端与会话管理
├── main.py                   # Pygame 桌面客户端
├── index.html                # Web 客户端
├── checkpoints/              # 训练检查点目录
└── checkpoints_test/         # 测试检查点目录
```

## 核心模块说明

### `obstacle_reversi_env.py`

实现 `ObstacleReversiEnv`，继承 `gymnasium.Env`。

主要接口：

- `reset(seed=None, options=None)`：重置棋盘，返回 `(observation, info)`
- `step(action)`：执行落子或 pass，返回 Gymnasium 标准五元组
- `get_legal_actions(player)`：返回合法落子布尔矩阵
- `is_terminal_position()`：判断双方是否都没有合法步
- `render(cell_size=80)`：用 Pygame 绘制棋盘

动作编码：

- `0 .. N*N-1`：棋盘位置，使用 `divmod(action, N)` 转为 `(row, col)`
- `N*N` 或 `-1`：Pass

### `alpha_reversi_net.py`

神经网络输入为 `(B, 4, N, N)`：

1. 当前玩家棋子
2. 对手棋子
3. 障碍物
4. 当前回合标记

输出包括：

- `p`：策略分布，形状 `(B, N*N)`
- `v`：局面价值，形状 `(B, 1)`，范围 `[-1, 1]`

网络由初始卷积、残差塔、策略头和价值头组成。策略头支持 `legal_mask`，非法动作会在 softmax 前被屏蔽。

### `mcts.py`

实现 AlphaZero 常用的 PUCT 搜索流程：

```text
select -> expand -> evaluate -> backpropagate
```

公开接口：

- `get_action_with_time_budget(board, player, time_limit_seconds, consecutive_passes=0)`
- `get_action_probs(root=None, temperature=1.0)`
- `advance_root(action)`

MCTS 会复用上一步保留下来的搜索树；真实游戏执行一步后调用 `advance_root(action)`，下一次搜索就可以从对应子树继续。

### `server.py`

提供 Web 版后端服务。每个游戏会话拥有独立的：

- `ObstacleReversiEnv`
- `MCTS`
- `threading.Lock`

主要接口：

| 方法 | 路径 | 说明 |
|---|---|---|
| `GET` | `/` | 返回 Web 客户端页面 |
| `GET` | `/api/new_game` | 创建新游戏 |
| `POST` | `/api/human_move` | 人类落子 |
| `POST` | `/api/human_pass` | 人类 pass |
| `POST` | `/api/ai_move` | AI 搜索并落子 |
| `GET` | `/api/game/{session_id}` | 获取当前游戏状态 |
| `DELETE` | `/api/game/{session_id}` | 删除游戏会话 |
| `GET` | `/api/health` | 健康检查 |

## 训练模型

使用命令行启动训练：

```bash
python train.py --N 8 --channels 128 --res-blocks 4 --iter 100
```

常用参数：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--N` | `8` | 棋盘大小 |
| `--obstacle-ratio` | `0.1` | 障碍物比例 |
| `--channels` | `128` | CNN 通道数 |
| `--res-blocks` | `4` | 残差块数量 |
| `--c-puct` | `1.4` | MCTS 探索系数 |
| `--time-limit` | `2.0` | 每步 MCTS 搜索秒数 |
| `--temp-threshold` | `15` | 前多少步使用温度探索 |
| `--iter` | `100` | 训练迭代轮数 |
| `--games-per-iter` | `50` | 每轮自对弈局数 |
| `--workers` | `4` | 并行自对弈线程数 |
| `--batch-size` | `64` | batch 大小 |
| `--batches-per-iter` | `20` | 每轮梯度更新次数 |
| `--save-dir` | `checkpoints` | 检查点保存目录 |
| `--resume` | 空 | 从检查点恢复 |
| `--gpu` | 关闭 | 如果 CUDA 可用则使用 GPU |

小规模测试训练：

```bash
python train.py --N 6 --channels 64 --res-blocks 2 --iter 3 \
  --games-per-iter 2 --workers 1 --time-limit 0.3
```

从检查点恢复：

```bash
python train.py --resume checkpoints/iter_050.pth --iter 50
```

## 数据流

训练阶段：

```text
train.py
  -> AlphaZeroTrainer.train_loop()
    -> generate_data()
      -> SelfPlayWorker.play_one_game()
        -> MCTS.get_action_with_time_budget()
        -> ObstacleReversiEnv.step()
        -> ExperienceReplayBuffer.push()
    -> train_step()
      -> ExperienceReplayBuffer.sample()
      -> AlphaReversiNet.forward()
      -> loss.backward()
      -> torch.save(checkpoint)
```

推理阶段：

```text
Web / Pygame client
  -> ObstacleReversiEnv
  -> MCTS
  -> AlphaReversiNet
  -> chosen action
```

## 常见问题

### 页面显示 `Failed to fetch`

后端服务没有运行，或者端口不是 `8000`。先启动：

```bash
python server.py
```

然后访问：

```text
http://127.0.0.1:8000/
```

### 页面能打开，但 AI 很弱

这是正常的。没有 `MODEL_PATH` 时，AI 使用随机权重。需要先训练，或者加载已有检查点。

### 加载检查点时报网络尺寸不匹配

优先使用本项目保存的检查点格式。检查点应包含：

- `model_state_dict`
- `N`
- `num_channels`
- `num_res_blocks`

`train.py`、`server.py` 和 `main.py` 会自动读取这些元数据。

### 游戏好像卡在某个玩家回合

如果当前玩家没有合法步，Web 版会显示 Pass 提示。双方都没有合法步时会直接结束并显示胜负。

## 开发检查

快速编译检查：

```bash
python -m py_compile obstacle_reversi_env.py alpha_reversi_net.py mcts.py \
  self_play.py trainer.py train.py server.py main.py
```

健康检查：

```bash
python server.py
curl http://127.0.0.1:8000/api/health
```
