# UserRL IntentionGym 环境安装指南

## 问题总结

在安装过程中遇到的主要依赖冲突：

1. **Python 版本问题**: Python 3.12 与 uvloop/asyncio 有兼容性问题
2. **vllm 与 sglang 冲突**: 两者不能共存，需要选择其一
3. **transformers 版本冲突**: 不同版本与 peft、sglang 有兼容性问题
4. **torch 版本冲突**: flashinfer 需要 torch 2.9，但 sglang 0.4.7 需要 torch 2.7

## 解决方案

### 方案一：使用 Python 3.10（推荐）

#### 1. 创建 Python 3.10 环境

```bash
conda create -n userrl_py310 python=3.10 -y
conda activate userrl_py310
cd /home/liuguanming/Multimodal-Agent/UserRL
```

#### 2. 安装核心依赖（按顺序）

```bash
# 第一步：安装 PyTorch 2.7.0
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# 第二步：安装 flash-attn
pip install flash-attn --no-build-isolation

# 第三步：安装 transformers（兼容版本）
pip install 'transformers>=4.46.0,<4.52.0'

# 第四步：安装 sglang（不包含 vllm）
pip install 'sglang[srt]==0.4.7' --no-deps
pip install aiohttp requests tqdm IPython setproctitle tiktoken einops openai
pip install outlines sentencepiece msgspec partial-json-parser compressed-tensors

# 第五步：安装其他必要依赖
pip install torchao xgrammar ninja

# 第六步：安装 UserRL 基础包
pip install -e . --no-deps

# 第七步：安装剩余依赖
pip install accelerate codetiming datasets dill hydra-core numpy pandas peft \
    pyarrow pybind11 pylatexenc 'ray[default]>=2.41.0' tensordict torchdata \
    wandb packaging torch-memory-saver fastapi uvicorn sentencepiece \
    compressed-tensors msgspec partial-json-parser pyzmq uvloop liger-kernel
```

#### 3. 安装 Gym 环境

```bash
bash install_gyms.sh
```

#### 4. 验证安装

```bash
python -c "
import torch
import sglang
import transformers
import peft
import ray
print(f'✅ torch {torch.__version__}')
print(f'✅ sglang {sglang.__version__}')
print(f'✅ transformers {transformers.__version__}')
print(f'✅ peft {peft.__version__}')
print(f'✅ ray {ray.__version__}')
print('所有核心包安装成功！')
"
```

### 方案二：修复现有 Python 3.12 环境（不推荐）

如果必须使用 Python 3.12，需要：

1. **卸载 vllm**（与 sglang 冲突）
```bash
conda activate userrl
pip uninstall vllm -y
```

2. **安装兼容版本的 transformers**
```bash
pip install 'transformers>=4.51.0' --upgrade
```

3. **安装缺失的依赖**
```bash
pip install torchao xgrammar ninja liger-kernel uvicorn
```

4. **注意**: Python 3.12 仍可能有 asyncio 事件循环问题，建议使用 Python 3.10

## 关键依赖版本

| 包名 | 推荐版本 | 说明 |
|-----|---------|------|
| **Python** | 3.10 | 避免 asyncio 兼容性问题 |
| **torch** | 2.7.0 | sglang 0.4.7 要求 |
| **sglang** | 0.4.7 | Rollout backend |
| **transformers** | 4.46.0-4.51.x | 与 peft 兼容 |
| **peft** | 0.18.0 | LoRA 支持 |
| **ray** | >=2.41.0 | 分布式训练 |
| **flash-attn** | 最新 | 加速注意力计算 |

## 重要说明

### ❌ 不要安装 vllm
- vllm 与 sglang 有依赖冲突
- 训练脚本使用 sglang 作为 backend，不需要 vllm
- 如果已安装，使用 `pip uninstall vllm -y` 卸载

### ✅ 必须安装的额外包
```bash
pip install torchao xgrammar ninja  # sglang 运行时依赖
pip install liger-kernel uvicorn    # UserRL 额外依赖
```

### ⚠️ 常见错误

#### 1. `ModuleNotFoundError: No module named 'torchao'`
```bash
pip install torchao
```

#### 2. `ModuleNotFoundError: No module named 'xgrammar'`
```bash
pip install xgrammar
```

#### 3. `RuntimeError: There is no current event loop`
- 原因：Python 3.12 的 asyncio 兼容性问题
- 解决：使用 Python 3.10 环境

#### 4. `ValueError: 'aimv2' is already used by a Transformers config`
- 原因：vllm 与 transformers 版本冲突
- 解决：卸载 vllm，升级 transformers

## 安装后配置

### 1. 配置 DashScope API Key

编辑 `train_intentiongym.sh`：
```bash
export DASHSCOPE_API_KEY="你的DashScope_API_KEY"
```

或在系统环境变量中设置：
```bash
echo 'export DASHSCOPE_API_KEY="sk-xxx"' >> ~/.bashrc
source ~/.bashrc
```

### 2. 检查数据文件

确认以下文件存在：
```bash
ls -lh /home/liuguanming/Multimodal-Agent/UserRL/data/intention_multiturn/
# 应该看到 train.parquet 和 test.parquet
```

如果不存在，运行数据预处理：
```bash
python examples/data_preprocess/intention_multiturn_w_tool.py
```

### 3. 验证 GPU 可用性

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

## 完整安装脚本

将以下内容保存为 `setup_userrl.sh`：

```bash
#!/bin/bash

# UserRL IntentionGym 环境安装脚本
set -e

echo "=== 创建 Python 3.10 环境 ==="
conda create -n userrl_py310 python=3.10 -y
conda activate userrl_py310

cd /home/liuguanming/Multimodal-Agent/UserRL

echo "=== 安装 PyTorch 2.7.0 ==="
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

echo "=== 安装 flash-attn ==="
pip install flash-attn --no-build-isolation

echo "=== 安装 transformers ==="
pip install 'transformers>=4.46.0,<4.52.0'

echo "=== 安装核心依赖 ==="
pip install accelerate codetiming datasets dill hydra-core numpy pandas peft \
    pyarrow pybind11 pylatexenc 'ray[default]>=2.41.0' tensordict torchdata \
    wandb packaging sentencepiece msgspec partial-json-parser compressed-tensors

echo "=== 安装 sglang 依赖 ==="
pip install aiohttp requests tqdm IPython setproctitle tiktoken einops openai outlines

echo "=== 安装额外依赖 ==="
pip install torchao xgrammar ninja liger-kernel uvicorn torch-memory-saver fastapi uvloop pyzmq

echo "=== 安装 sglang ==="
pip install 'sglang[srt]==0.4.7'

echo "=== 安装 UserRL ==="
pip install -e .

echo "=== 安装 Gym 环境 ==="
bash install_gyms.sh

echo "=== 验证安装 ==="
python -c "
import torch
import sglang
import transformers
import peft
import ray
import intentiongym
print('\\n✅ 所有包安装成功！')
print(f'  - torch: {torch.__version__}')
print(f'  - sglang: {sglang.__version__}')
print(f'  - transformers: {transformers.__version__}')
print(f'  - peft: {peft.__version__}')
print(f'  - ray: {ray.__version__}')
print(f'  - CUDA available: {torch.cuda.is_available()}')
"

echo "=== 安装完成 ==="
echo "激活环境: conda activate userrl_py310"
echo "运行训练: bash train_intentiongym.sh"
```

使用方法：
```bash
chmod +x setup_userrl.sh
bash setup_userrl.sh
```

## 故障排查

### 查看详细错误信息

```bash
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
bash train_intentiongym.sh
```

### 清理并重新安装

```bash
# 删除旧环境
conda env remove -n userrl_py310

# 停止 Ray
ray stop

# 清理缓存
pip cache purge

# 重新安装
bash setup_userrl.sh
```

## 参考资源

- [UserRL 官方文档](https://github.com/SalesforceAIResearch/UserRL)
- [SGLang 文档](https://github.com/sgl-project/sglang)
- [DashScope API](https://dashscope.console.aliyun.com/apiKey)

## 更新日志

- **2025-11-26**: 初始版本，记录 Python 3.10 + sglang 0.4.7 安装方案
- **2025-11-26**: 解决 vllm 与 sglang 冲突，移除 vllm
- **2025-11-26**: 添加 torchao、xgrammar 等缺失依赖
- **2025-11-26**: 修复 transformers 与 peft 版本兼容性问题

