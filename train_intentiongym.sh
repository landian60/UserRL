#!/bin/bash
if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi
# IntentionGym 训练脚本 - 三卡全量微调配置（卡0、1、2训练）
# 训练模型: Qwen3-8B (运行在卡0、1、2，全量微调，不使用 LoRA)
# 用户模拟器: Qwen3-30B-A3B-Instruct-2507-FP8 (运行在其他卡上)
# 环境: userrl (Python 3.12 - 已修复 asyncio 兼容性)
#
# 注意：使用前请先启动用户模拟器服务：
#   bash start_user_simulator.sh
# 该服务将在其他卡上运行，端口 8000
# 
# 注意：训练进程使用卡0、1、2进行全量微调，FSDP自动将模型参数和优化器状态分片到三张卡

set -x
# ==================== 环境配置 ====================
# 训练使用卡0、1、2进行全量微调
export CUDA_VISIBLE_DEVICES=0,1,2

# 用户模拟器配置
# 模型路径: /home/liuguanming/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507-FP8/snapshots/5a5a776300a41aaa681dd7ff0106608ef2bc90db
# 该模型通过 API 服务运行在卡0上
: "${OPENAI_API_KEY:=dummy-key}"
: "${OPENAI_BASE_URL:=http://127.0.0.1:8000/v1}"
: "${MULTITURN_MODEL_NAME:=Qwen/Qwen3-30B-A3B-Instruct-2507}"
# : "${MULTITURN_MODEL_NAME:=Qwen/Qwen3-14B}"
export OPENAI_API_KEY OPENAI_BASE_URL MULTITURN_MODEL_NAME

# 以下为原 DashScope 配置，保留作参考（默认已禁用）
# : ${DASHSCOPE_API_KEY:="你的DashScope_API_KEY"}
# export OPENAI_API_KEY="$DASHSCOPE_API_KEY"
# export OPENAI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
# export MULTITURN_MODEL_NAME="qwen-plus"
# 其他可选的 DashScope 模型：
# export MULTITURN_MODEL_NAME="qwen-turbo"     # 更快更便宜
# export MULTITURN_MODEL_NAME="qwen-max"       # 最强性能
# export MULTITURN_MODEL_NAME="qwen-plus"      # 平衡性能和成本（推荐）

ulimit -n 65535

# 解决 Python 3.12 asyncio 兼容性问题
export PYTHONUNBUFFERED=1
export RAY_DISABLE_IMPORT_WARNING=1

# Hugging Face 模型缓存路径配置
export HUGGINGFACE_HUB_CACHE=/vePFS-Mindverse/share/huggingface/hub
export HF_HOME=/vePFS-Mindverse/share/huggingface

# ==================== 路径配置 ====================
# PROJECT_DIR="/home/liuguanming/Multimodal-Agent/UserRL"
PROJECT_DIR="/root/UserRL"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"

# 模型路径 - 使用 Qwen3-4B
# MODEL_PATH="/home/liuguanming/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/531c80e289d6cff3a7cd8c0db8110231d23a6f7a"
# MODEL_PATH="/vePFS-Mindverse/share/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
MODEL_PATH="/vePFS-Mindverse/share/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
# 数据路径 - IntentionGym 专用
TRAIN_DATA="$PROJECT_DIR/data/intention_multiturn/train.parquet"
VAL_DATA="$PROJECT_DIR/data/intention_multiturn/test.parquet"

# ==================== 训练参数 ====================
# 全量微调配置：使用三张80GB显存卡，FSDP自动分片模型参数和优化器状态
TRAIN_BATCH_SIZE=6            # 全量微调显存占用更大，适当减小batch size
MAX_PROMPT_LENGTH=1152
MAX_RESPONSE_LENGTH=4096
LEARNING_RATE=1e-6
TOTAL_EPOCHS=10

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
    actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.actor.optim.min_lr_ratio=0.1 \
    actor_rollout_ref.actor.optim.num_cycles=0.5 \
    actor_rollout_ref.actor.optim.warmup_style=cosine \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=6 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.01 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.50 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.multi_turn.max_turns=16 \
    actor_rollout_ref.rollout.multi_turn.model_name="$MULTITURN_MODEL_NAME" \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$CONFIG_PATH/tool_config/interact_tool_config.yaml" \
    actor_rollout_ref.rollout.multi_turn.turn_level_method="Equalized" \
    actor_rollout_ref.rollout.multi_turn.trajectory_score_method="Sum" \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='UserRL' \
    trainer.experiment_name='IntentionGym_Qwen8B_3GPU_Full_Group8_kl' \
    trainer.default_local_dir='/vePFS-Mindverse/user/intern/lgm/checkpoints/IntentionGym_Qwen8B_3GPU_Full_Group8_kl' \
    trainer.n_gpus_per_node=3 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.val_before_train=False \
    trainer.total_epochs=$TOTAL_EPOCHS $@ $1

