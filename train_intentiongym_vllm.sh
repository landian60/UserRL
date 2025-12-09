#!/bin/bash
if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi
# IntentionGym 训练脚本 - vLLM Backend 版本（更好的 LoRA 支持）
# 训练模型: Qwen3-8B (运行在卡1和卡2，使用 LoRA)
# 用户模拟器: Qwen3-30B-A3B-Instruct-2507-FP8 (运行在卡0)
# 
# vLLM 相比 SGLang 的优势：
# - 原生 LoRA 支持，无需在推理时合并权重
# - 更稳定的多 LoRA 切换
# - 更成熟的生产环境支持
#
# 注意：使用前请先启动用户模拟器服务：
#   bash start_user_simulator.sh
# 该服务将在卡0上运行，端口 8000

set -x
# ==================== 环境配置 ====================
# 训练只用卡1和卡2，卡0专门负责用户模拟器
export CUDA_VISIBLE_DEVICES=1,2

# 用户模拟器配置
: "${OPENAI_API_KEY:=dummy-key}"
: "${OPENAI_BASE_URL:=http://127.0.0.1:8000/v1}"
: "${MULTITURN_MODEL_NAME:=Qwen/Qwen3-30B-A3B-Instruct-2507}"
export OPENAI_API_KEY OPENAI_BASE_URL MULTITURN_MODEL_NAME

ulimit -n 65535

# 解决 Python 3.12 asyncio 兼容性问题
export PYTHONUNBUFFERED=1
export RAY_DISABLE_IMPORT_WARNING=1

# Hugging Face 模型缓存路径配置
export HUGGINGFACE_HUB_CACHE=/vePFS-Mindverse/share/huggingface/hub
export HF_HOME=/vePFS-Mindverse/share/huggingface

# vLLM 相关环境变量 - 启用运行时 LoRA 更新
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=true
export VLLM_LOGGING_LEVEL=WARN
# 禁用 v1 engine 以避免 FlashInfer 架构检测问题（使用旧的稳定 engine）
export VLLM_USE_V1=0
# 注意：不设置 VLLM_ATTENTION_BACKEND=XFORMERS，因为 xformers 与 flash-attn 2.8.3 不兼容
# 使用默认的 Flash Attention 后端，配合 enforce_eager=True 和 enable_chunked_prefill=False

# ==================== 路径配置 ====================
PROJECT_DIR="/root/UserRL"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"

# 模型路径 - 使用 Qwen3-8B
MODEL_PATH="/vePFS-Mindverse/share/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"

# 数据路径 - IntentionGym 专用
TRAIN_DATA="$PROJECT_DIR/data/intention_multiturn/train.parquet"
VAL_DATA="$PROJECT_DIR/data/intention_multiturn/test.parquet"

# ==================== 训练参数 ====================
TRAIN_BATCH_SIZE=8
MAX_PROMPT_LENGTH=1152
MAX_RESPONSE_LENGTH=4096
LEARNING_RATE=1e-6
TOTAL_EPOCHS=20

# ==================== LoRA 参数 ====================
# vLLM 原生支持 LoRA，训练时自动启用 enable_lora=True
LORA_RANK=32                  # LoRA rank
LORA_ALPHA=64                 # LoRA alpha，通常设置为 rank 的 2 倍
TARGET_MODULES="all-linear"   # 目标模块

# ==================== vLLM 特有参数 ====================
# vLLM LoRA 相关配置
MAX_LORAS=1                   # 最大同时加载的 LoRA 数量
MAX_LORA_RANK=64              # 最大支持的 LoRA rank（需 >= LORA_RANK）

# vLLM 推理优化参数
ENFORCE_EAGER=True            # 禁用 CUDA Graph，LoRA 必须设为 True
FREE_CACHE_ENGINE=True        # 生成后释放 KV Cache
ENABLE_CHUNKED_PREFILL=False  # 禁用分块预填充，避免多轮对话中动态输入长度导致的 Flash Attention 断言错误

# ==================== 执行训练 ====================
cd "$PROJECT_DIR" || exit 1

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='grpo_multiturn' \
    algorithm.adv_estimator=grpo_multiturn \
    algorithm.gamma=0.8 \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.lora_rank=$LORA_RANK \
    actor_rollout_ref.model.lora_alpha=$LORA_ALPHA \
    actor_rollout_ref.model.target_modules=$TARGET_MODULES \
    actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.50 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.enforce_eager=$ENFORCE_EAGER \
    actor_rollout_ref.rollout.free_cache_engine=$FREE_CACHE_ENGINE \
    actor_rollout_ref.rollout.enable_chunked_prefill=$ENABLE_CHUNKED_PREFILL \
    actor_rollout_ref.rollout.load_format=dummy_dtensor \
    actor_rollout_ref.rollout.multi_turn.max_turns=16 \
    actor_rollout_ref.rollout.multi_turn.model_name="$MULTITURN_MODEL_NAME" \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$CONFIG_PATH/tool_config/interact_tool_config.yaml" \
    actor_rollout_ref.rollout.multi_turn.turn_level_method="Equalized" \
    actor_rollout_ref.rollout.multi_turn.trajectory_score_method="Sum" \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='UserRL' \
    trainer.experiment_name='IntentionGym_Qwen8B_vLLM_LoRA' \
    trainer.default_local_dir='/vePFS-Mindverse/user/intern/tmp/UserRL/IntentionGym_Qwen8B_vLLM_LoRA' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.val_before_train=False \
    trainer.total_epochs=$TOTAL_EPOCHS $@ $1


