# SmolVLA Cube-to-Tray

在 MuJoCo 中采集 Panda 专家轨迹，微调 SmolVLA，并对比微调前后的闭环表现。

```text
Task: Pick up the red cube and place it in the tray.
Success: 方块完整位于托盘内并稳定保持 2 秒。
```

## 项目结构

```text
.
├── pyproject.toml
├── README.md
├── src/
│   └── smolvla_task/
│       ├── controllers/
│       │   ├── panda_ik_controller.py       # 带偏置力补偿的 Panda IK
│       │   ├── panda_ik_controller_no_gravity.py  # 无补偿对照控制器
│       │   └── scripted_expert.py           # 脚本专家策略
│       ├── envs/
│       │   ├── cube_tray_env.py             # MuJoCo 任务环境
│       │   └── assets/franka_emika_panda/   # Panda、方块和托盘模型
│       └── utils/
│           ├── dataset_validation.py        # 数据集校验与 replay
│           ├── episode_monitor.py           # 掉落与碰撞监测
│           └── mujoco_video_recorder.py     # 双相机视频录制
├── scripts/
│   ├── collect_data.py                      # 采集 LeRobotDataset
│   ├── evaluate.py                          # 单模型闭环评估
│   ├── compare_models.py                    # 50-seed 模型对比
│   ├── compare_gravity_compensation.py      # 重力补偿运动精度对比
│   └── test_scripted_pick.py                # 脚本专家 smoke
│
├── models/ 
│   └── smolvla_cube_tray_finetuned/         # 最终推理模型
├── datasets/ 
│   └── smolvla_cube_tray/                   # 100-episode 数据集
└── outputs/ 
    ├── train/                               # checkpoint 和训练状态
    ├── evaluation/                          # 指标、CSV 和评估视频
    └── video/                               # 专家测试视频
```

`models/`、`datasets/` 和 `outputs/` 均不进入 Git。

## 环境配置

默认已经安装 Conda，并在项目根目录执行以下命令。

```bash
# 1. 创建并激活独立环境
conda create -n lerobot python=3.12 -y
conda activate lerobot

# 2. 安装视频编解码工具
conda install -c conda-forge "ffmpeg=9" -y

# 3. 更新 Python 打包工具
python -m pip install --upgrade \
  pip wheel "setuptools>=77,<82"

# 4. 安装当前机器使用的 CUDA 13.0 版 PyTorch
python -m pip install \
  torch==2.11.0 \
  torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu130

# 5. 安装项目、MuJoCo 和固定版本的 LeRobot
python -m pip install -e .
```

LeRobot 固定在：

```text
2595896f8a5c70f06adc1bcdf446d3aaa4cc3f20
```

## 实验步骤

### 1. 验证脚本专家

```bash
python scripts/test_scripted_pick.py \
  --seed 42 \
  --headless \
  --video false
```

### 2. 采集 100 个 episode

```bash
python scripts/collect_data.py \
  --num-episodes 100 \
  --start-seed 0 \
  --repo-id local/smolvla_cube_tray \
  --root datasets/smolvla_cube_tray
```

已有数据集时添加 `--resume`。

### 3. 微调 SmolVLA

```bash
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.input_features=null \
  --policy.output_features=null \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --policy.train_state_proj=true \
  --policy.optimizer_lr=1e-4 \
  --policy.scheduler_warmup_steps=4000 \
  --policy.scheduler_decay_steps=120000 \
  --policy.scheduler_decay_lr=2.5e-6 \
  --dataset.repo_id=local/smolvla_cube_tray \
  --dataset.root=datasets/smolvla_cube_tray \
  --dataset.eval_split=0.2 \
  --batch_size=1 \
  --accelerator.mixed_precision=bf16 \
  --accelerator.gradient_accumulation.steps=4 \
  --num_workers=2 \
  --steps=80000 \
  --eval_steps=2000 \
  --max_eval_samples=512 \
  --env_eval_freq=0 \
  --log_freq=40 \
  --save_checkpoint=true \
  --save_freq=8000 \
  --output_dir=outputs/train/smolvla_cube_tray \
  --job_name=smolvla_cube_tray \
  --wandb.enable=false
```

训练 checkpoint 位于 `outputs/train/`；最终推理模型位于
`models/smolvla_cube_tray_finetuned/`。

### 4. 单模型评估

```bash
python scripts/evaluate.py \
  --policy-path models/smolvla_cube_tray_finetuned \
  --dataset-repo-id local/smolvla_cube_tray \
  --dataset-root datasets/smolvla_cube_tray \
  --seed 100 \
  --n-action-steps 10 \
  --max-episode-seconds 30 \
  --run-name smolvla_finetuned \
  --offline
```

基础模型使用 `--policy-path lerobot/smolvla_base`。结果写入：

```text
outputs/evaluation/<run_name>_seed_<seed>/
├── metrics.json
├── front.mp4
└── wrist.mp4
```

### 5. 50-seed 对比实验

```bash
python scripts/compare_models.py \
  --base-policy-path lerobot/smolvla_base \
  --finetuned-policy-path models/smolvla_cube_tray_finetuned \
  --dataset-repo-id local/smolvla_cube_tray \
  --dataset-root datasets/smolvla_cube_tray \
  --start-seed 100 \
  --num-seeds 50 \
  --n-action-steps 10 \
  --max-episode-seconds 30 \
  --run-name smolvla_base_vs_finetuned \
  --offline
```

输出：

```text
outputs/evaluation/smolvla_base_vs_finetuned_seeds_100_149/
├── summary.json    # 聚合结果
├── episodes.csv    # 逐 episode 结果
├── base/
└── finetuned/
```

统计指标：成功率、任务耗时、推理延迟、掉落率和碰撞率。对比实验默认不录制视频。

### 6. 重力补偿精度对比

```bash
python scripts/compare_gravity_compensation.py \
  --seed 42 \
  --output-dir outputs/evaluation/gravity_compensation
```

两个控制器从完全相同的状态出发，以 25 Hz 执行 8 段平滑位置轨迹，
保持初始末端姿态。每段移动 2 秒、保持 2 秒，最后 0.4 秒统计稳态误差。
默认收敛阈值为位置 5 mm、姿态 0.03 rad，整个稳态窗口满足阈值才算收敛。

```text
outputs/evaluation/gravity_compensation/
├── summary.json            # 聚合精度与 with - without 差值
├── waypoints.csv           # 每个目标的收敛和稳态误差
├── trajectory.csv          # 每周期目标/实际位姿、关节命令与补偿量
├── with_gravity/metrics.json
└── without_gravity/metrics.json
```

统计位置 RMSE、最大误差、稳态误差、姿态 RMSE、关节跟踪 RMSE 和收敛率。
`qfrc_bias` 包含重力及速度相关偏置；该实验比较现有控制器的补偿开关，
不是纯重力力矩控制。关节跟踪误差以执行器命令为参考，补偿命令本身包含偏置。
时间参数须为 0.04 秒的整数倍；重复运行同一输出目录会覆盖已有结果。
