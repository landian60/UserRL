#!/bin/bash
if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi
# 用户模拟器启动脚本 - 单独使用GPU卡0
# 模型: Qwen3-30B-A3B-Instruct-2507-FP8

set -x

# ==================== 环境配置 ====================
# 指定使用GPU卡0
export CUDA_VISIBLE_DEVICES=3

# 模型路径
# MODEL_PATH="/home/liuguanming/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
# MODEL_PATH="/home/liuguanming/.cache/huggingface/hub/models--Qwen--Qwen3-14B/snapshots/8268fe3026cb304910457689366670e803a6fd56"
MODEL_PATH="/vePFS-Mindverse/share/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507/snapshots/0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe/"

# 服务配置
HOST="127.0.0.1"
PORT=8000

# ==================== 启动服务 ====================
echo "正在启动用户模拟器服务..."
echo "模型路径: $MODEL_PATH"
echo "GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "服务地址: http://${HOST}:${PORT}"

# 使用 vLLM 启动服务（如果您使用的是 vLLM）
# 如果使用其他推理框架（如 SGLang），请相应修改命令
# python3 -m vllm.entrypoints.openai.api_server \
#     --model "$MODEL_PATH" \
#     --host "$HOST" \
#     --port "$PORT" \
#     --served-model-name "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8" \
#     --trust-remote-code \
#     --gpu-memory-utilization 0.90 \
#     --max-model-len 32768 \
#     --dtype auto

# 如果您使用 SGLang，请使用以下命令替代上面的 vLLM 命令：
# 优化显存配置：
# - mem-fraction-static: 0.65 (降低显存占用，释放更多内存用于并发请求)
# - context-length: 24576 (根据实际需求调整)
# - max-running-requests: 64 (增加同时运行的最大请求数，从默认值提升)
# - max-total-tokens: 65536 (增加最大总token数，支持更多并发请求，约支持32-64个并发)
python3 -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --trust-remote-code \
    --served-model-name "Qwen/Qwen3-30B-A3B-Instruct-2507" \
    --mem-fraction-static 0.8 \
    --context-length 24576 \
    --max-running-requests 32 \
    --max-total-tokens 196608

