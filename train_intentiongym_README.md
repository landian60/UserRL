# IntentionGym 训练脚本使用说明

## 脚本位置
`/home/liuguanming/Multimodal-Agent/UserRL/train_intentiongym.sh`

## ⚠️ 重要：环境安装

**首次使用前，请先阅读完整的安装指南：**
- 📖 [INSTALLATION_GUIDE.md](./INSTALLATION_GUIDE.md) - 详细的依赖安装和问题解决方案

### 快速安装（推荐 Python 3.10）

```bash
# 1. 创建 Python 3.10 环境（推荐）
conda create -n userrl_py310 python=3.10 -y
conda activate userrl_py310

# 2. 运行安装脚本
cd /home/liuguanming/Multimodal-Agent/UserRL
bash setup_userrl.sh  # 参见 INSTALLATION_GUIDE.md 中的完整脚本
```

### 使用现有环境

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate userrl_py310  # 或 userrl（如果使用 Python 3.12）
```

**注意事项：**
- ✅ 使用 Python 3.10（推荐）避免 asyncio 兼容性问题
- ❌ 不要同时安装 vllm 和 sglang（会冲突）
- ✅ 确保安装了 torchao、xgrammar、ninja 等依赖

### 3. 配置 DashScope API Key
编辑 `train_intentiongym.sh`，将以下行替换为你的真实 DashScope API Key：
```bash
export DASHSCOPE_API_KEY="你的DashScope_API_KEY"
```

获取 API Key: https://dashscope.console.aliyun.com/apiKey

**可选模型**（在脚本中修改 `MULTITURN_MODEL_NAME`）：
- `qwen-turbo` - 更快更便宜
- `qwen-plus` - 平衡性能和成本（默认推荐）
- `qwen-max` - 最强性能

如果要使用本地模型作为用户模拟器，修改为：
```bash
export OPENAI_BASE_URL="http://localhost:8000/v1"
export MULTITURN_MODEL_NAME="Qwen/Qwen3-32B"  # 或其他本地模型
```

### 4. 检查数据文件
确认以下数据文件存在：
- `/home/liuguanming/Multimodal-Agent/UserRL/data/intention_multiturn/train.parquet`
- `/home/liuguanming/Multimodal-Agent/UserRL/data/intention_multiturn/test.parquet`

如果不存在，运行数据预处理脚本：
```bash
cd /home/liuguanming/Multimodal-Agent/UserRL
python examples/data_preprocess/intention_multiturn_w_tool.py
```

## 开始训练

**在运行前，请先在脚本中配置 DashScope API Key：**
```bash
vim /home/liuguanming/Multimodal-Agent/UserRL/train_intentiongym.sh
# 找到并修改：export DASHSCOPE_API_KEY="你的DashScope_API_KEY"
```

### 方式一：直接运行脚本
```bash
cd /home/liuguanming/Multimodal-Agent/UserRL
bash train_intentiongym.sh
```

### 方式二：后台运行（推荐）
```bash
cd /home/liuguanming/Multimodal-Agent/UserRL
nohup bash train_intentiongym.sh > train_intention.log 2>&1 &
```

查看日志：
```bash
tail -f train_intention.log
```

## 训练参数说明

### 硬件配置
- **GPU**: 使用卡1和卡2 (`CUDA_VISIBLE_DEVICES=1,2`)
- **显卡数量**: 2 张 (`trainer.n_gpus_per_node=2`)

### 模型配置
- **模型**: Qwen3-0.6B
- **路径**: `/home/liuguanming/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca`

### 训练超参数
- **Batch Size**: 64（可根据显存调整到 128）
- **Learning Rate**: 1e-6
- **总轮数**: 10 epochs（在脚本中 `TOTAL_EPOCHS=10`）
- **最大对话轮数**: 16 turns
- **保存频率**: 每 1 epoch
- **验证频率**: 每 5 epochs

## 自定义参数

### 修改训练轮数
编辑脚本中的 `TOTAL_EPOCHS` 变量：
```bash
TOTAL_EPOCHS=20  # 训练 20 轮
```

### 修改 Batch Size（如果显存充足）
```bash
TRAIN_BATCH_SIZE=128
```

### 修改学习率
```bash
LEARNING_RATE=5e-7
```

### 使用不同的 GPU
```bash
export CUDA_VISIBLE_DEVICES=0,1  # 改用卡0和卡1
```

## 监控训练

### 1. 控制台输出
训练日志会实时打印在终端

### 2. WandB 监控
如果配置了 wandb，可以在 https://wandb.ai 查看：
- 项目名称: `UserRL`
- 实验名称: `IntentionGym_Qwen0.6B_dualGPU`

如果不想使用 wandb，修改脚本中的：
```bash
trainer.logger=['console']  # 只使用控制台日志
```

### 3. Checkpoint 保存
模型 checkpoint 会保存在：
```
/home/liuguanming/Multimodal-Agent/UserRL/output/
```

## 恢复训练

如果训练中断，可以从 checkpoint 恢复：
```bash
bash train_intentiongym.sh trainer.resume_from=/path/to/checkpoint
```

## 常见问题

### 1. CUDA Out of Memory
降低 `TRAIN_BATCH_SIZE` 或 `ppo_micro_batch_size_per_gpu`

### 2. 数据文件不存在
运行数据预处理脚本生成 parquet 文件

### 3. API Key 错误
检查 `DASHSCOPE_API_KEY` 是否正确配置，获取地址：https://dashscope.console.aliyun.com/apiKey

### 4. 端口占用
如果提示端口被占用，可能是之前的进程未关闭：
```bash
pkill -f "sglang"
pkill -f "verl.trainer"
```

## 训练完成后

训练完成后，最佳模型会保存在 output 目录，可以用于：
1. 评估测试集性能
2. 部署推理服务
3. 继续微调训练

查看评估说明：
```bash
cat /home/liuguanming/Multimodal-Agent/UserRL/eval/README.md
```

