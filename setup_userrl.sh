#!/bin/bash

# UserRL IntentionGym 环境安装脚本
# 解决所有依赖冲突，使用 Python 3.10
set -e

echo "======================================"
echo "  UserRL IntentionGym 环境安装脚本"
echo "======================================"
echo ""

# 检查是否在正确的目录
if [ ! -f "setup.py" ]; then
    echo "❌ 错误：请在 UserRL 项目根目录运行此脚本"
    exit 1
fi

# 检查 conda 是否可用
if ! command -v conda &> /dev/null; then
    echo "❌ 错误：未找到 conda，请先安装 Miniconda 或 Anaconda"
    exit 1
fi

echo "=== 步骤 1/9: 创建 Python 3.10 环境 ==="
echo "环境名称: userrl_py310"
read -p "是否继续？(y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
fi

conda create -n userrl_py310 python=3.10 -y
eval "$(conda shell.bash hook)"
conda activate userrl_py310

echo ""
echo "=== 步骤 2/9: 安装 PyTorch 2.7.0 + CUDA 12.6 ==="
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

echo ""
echo "=== 步骤 3/9: 安装 flash-attn ==="
pip install flash-attn --no-build-isolation

echo ""
echo "=== 步骤 4/9: 安装 transformers ==="
pip install 'transformers>=4.46.0,<4.52.0'

echo ""
echo "=== 步骤 5/9: 安装核心依赖 ==="
pip install accelerate codetiming datasets dill hydra-core numpy pandas peft \
    pyarrow pybind11 pylatexenc 'ray[default]>=2.41.0' tensordict torchdata \
    wandb packaging sentencepiece msgspec partial-json-parser compressed-tensors

echo ""
echo "=== 步骤 6/9: 安装 sglang 相关依赖 ==="
pip install aiohttp requests tqdm IPython setproctitle tiktoken einops openai outlines

echo ""
echo "=== 步骤 7/9: 安装额外运行时依赖 ==="
pip install torchao xgrammar ninja liger-kernel uvicorn torch-memory-saver fastapi uvloop pyzmq

echo ""
echo "=== 步骤 8/9: 安装 sglang ==="
pip install 'sglang[srt]==0.4.7'

echo ""
echo "=== 步骤 9/9: 安装 UserRL 和 Gym 环境 ==="
pip install -e .
bash install_gyms.sh

echo ""
echo "=== 验证安装 ==="
python -c "
import sys
import torch
import sglang
import transformers
import peft
import ray
import intentiongym

print('\n' + '='*50)
print('✅ 安装验证成功！')
print('='*50)
print(f'Python 版本: {sys.version.split()[0]}')
print(f'torch: {torch.__version__}')
print(f'sglang: {sglang.__version__}')
print(f'transformers: {transformers.__version__}')
print(f'peft: {peft.__version__}')
print(f'ray: {ray.__version__}')
print(f'CUDA 可用: {torch.cuda.is_available()}')
print(f'GPU 数量: {torch.cuda.device_count()}')
print('='*50)
"

echo ""
echo "======================================"
echo "  🎉 安装完成！"
echo "======================================"
echo ""
echo "下一步操作："
echo "1. 激活环境: conda activate userrl_py310"
echo "2. 配置 API Key: 编辑 train_intentiongym.sh，设置 DASHSCOPE_API_KEY"
echo "3. 运行训练: bash train_intentiongym.sh"
echo ""
echo "详细文档："
echo "- 安装指南: INSTALLATION_GUIDE.md"
echo "- 使用说明: train_intentiongym_README.md"
echo ""

