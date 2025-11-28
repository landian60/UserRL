# IntentionGym 训练快速开始

## 🚀 一键安装

```bash
cd /home/liuguanming/Multimodal-Agent/UserRL
bash setup_userrl.sh
```

## 📝 配置与运行

### 1. 配置 API Key

```bash
# 编辑训练脚本
vim train_intentiongym.sh

# 修改这一行（第 14 行）
# export DASHSCOPE_API_KEY="你的DashScope_API_KEY"
```


### 2. 启动训练

```bash
# 激活环境
conda activate userrl_py310

# 运行训练
bash train_intentiongym.sh
```

### 3. 查看日志

```bash
# 实时查看
tail -f train_intention.log

# 查看最后 100 行
tail -n 100 train_intention.log
```

## 📊 训练配置

| 配置项 | 值 | 说明 |
|--------|-----|------|
| **GPU** | 卡1 & 卡2 | `CUDA_VISIBLE_DEVICES=1,2` |
| **模型** | Qwen3-0.6B | 小模型，训练快速 |
| **数据** | IntentionGym | 380 训练样本，40 验证样本 |
| **Batch Size** | 64 | 双卡适配 |
| **训练轮数** | 10 epochs | 约 50 steps |
| **Backend** | SGLang | 高效推理引擎 |
| **用户模拟** | qwen-plus (DashScope) | 通过 API 调用 |

## 🔧 常见问题

### ❌ 训练没有启动

```bash
# 检查进程
ps aux | grep "verl.trainer.main_ppo"

# 查看错误日志
cat train_intention.log | grep -A 10 "Error"
```

### ❌ CUDA Out of Memory

降低 Batch Size：
```bash
# 编辑 train_intentiongym.sh
TRAIN_BATCH_SIZE=32  # 从 64 改为 32
```

### ❌ API Key 错误

检查环境变量：
```bash
echo $DASHSCOPE_API_KEY
```

### ❌ 端口被占用

清理进程：
```bash
ray stop
pkill -f "sglang"
pkill -f "verl.trainer"
```

## 📁 重要文件

```
UserRL/
├── train_intentiongym.sh          # 训练脚本
├── setup_userrl.sh                 # 环境安装脚本
├── INSTALLATION_GUIDE.md           # 详细安装指南
├── train_intentiongym_README.md   # 完整使用文档
├── QUICK_START_IntentionGym.md    # 本文件
├── train_intention.log             # 训练日志（运行后生成）
└── data/intention_multiturn/       # 训练数据
    ├── train.parquet
    └── test.parquet
```

## 🎯 训练流程

```
1. 数据加载 (380 训练样本)
   ↓
2. 模型初始化 (Qwen3-0.6B)
   ↓
3. SGLang 预热 (Capturing batches)
   ↓
4. 开始训练循环
   ↓
   - Rollout: 模型与环境交互
   - Reward: 计算奖励信号
   - Update: 更新模型参数
   ↓
5. 每 1 epoch 保存 checkpoint
   ↓
6. 每 5 epochs 验证一次
   ↓
7. 训练完成 (10 epochs)
```

## 💾 Checkpoint 位置

```bash
# 模型 checkpoint 保存在
checkpoints/UserRL/IntentionGym_Qwen0.6B_dualGPU/
```

## 📈 监控训练

### WandB (推荐)

训练会自动上传到 WandB：
- 项目名: `UserRL`
- 实验名: `IntentionGym_Qwen0.6B_dualGPU`
- 地址: https://wandb.ai

### 仅控制台日志

如不想使用 WandB，编辑 `train_intentiongym.sh`：
```bash
trainer.logger=['console']  # 移除 wandb
```

## 🛠️ 高级配置

### 修改学习率

```bash
# 在 train_intentiongym.sh 中
LEARNING_RATE=5e-7  # 默认 1e-6
```

### 修改训练轮数

```bash
# 在 train_intentiongym.sh 中
TOTAL_EPOCHS=20  # 默认 10
```

### 使用不同的 GPU

```bash
# 在 train_intentiongym.sh 中
export CUDA_VISIBLE_DEVICES=0,1  # 使用卡0和卡1
```

### 修改用户模拟模型

```bash
# 在 train_intentiongym.sh 中
export MULTITURN_MODEL_NAME="qwen-turbo"  # 更快更便宜
# export MULTITURN_MODEL_NAME="qwen-max"    # 最强性能
```

## 📚 相关文档

- [完整安装指南](./INSTALLATION_GUIDE.md) - 解决所有依赖冲突
- [详细使用说明](./train_intentiongym_README.md) - 完整配置选项
- [UserRL 主文档](./README.md) - 项目总体介绍

## ✅ 检查清单

在开始训练前，确认：

- [ ] Python 3.10 环境已创建并激活
- [ ] 所有依赖已安装（运行 `setup_userrl.sh`）
- [ ] DashScope API Key 已配置
- [ ] 数据文件存在（`data/intention_multiturn/*.parquet`）
- [ ] GPU 可用（`nvidia-smi` 查看）
- [ ] 卡1和卡2空闲（通过 `nvidia-smi` 确认）

## 🎓 训练完成后

```bash
# 1. 查看最佳模型
ls -lh checkpoints/UserRL/IntentionGym_Qwen0.6B_dualGPU/

# 2. 评估模型（可选）
cd eval/
# 参考 eval/README.md

# 3. 清理资源
ray stop
conda deactivate
```

---

**祝训练顺利！** 🚀

如遇到问题，请查看：
- 训练日志: `train_intention.log`
- 安装指南: `INSTALLATION_GUIDE.md`
- 详细文档: `train_intentiongym_README.md`

