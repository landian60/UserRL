#!/bin/bash
if [ -z "${BASH_VERSION:-}" ]; then
    exec bash "$0" "$@"
fi
# IntentionGym 训练脚本 - 双卡配置（卡1和卡2）
# 训练模型: Qwen3-4B (运行在卡1和卡2)
# 用户模拟器: Qwen3-30B-A3B-Instruct-2507-FP8 (运行在卡0)
# 环境: userrl (Python 3.12 - 已修复 asyncio 兼容性)
#
# 注意：使用前请先启动用户模拟器服务：
#   bash start_user_simulator.sh
# 该服务将在卡0上运行，端口 8000

set -x
# ==================== 环境配置 ====================
# 指定使用卡1和卡2进行训练
export CUDA_VISIBLE_DEVICES=1,2

# 用户模拟器配置
# 模型路径: /home/liuguanming/.cache/huggingface/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507-FP8/snapshots/5a5a776300a41aaa681dd7ff0106608ef2bc90db
# 该模型通过 API 服务运行在卡0上
: "${OPENAI_API_KEY:=dummy-key}"
: "${OPENAI_BASE_URL:=http://127.0.0.1:8000/v1}"
# : "${MULTITURN_MODEL_NAME:=Qwen/Qwen3-30B-A3B-Instruct-2507-FP8}"
: "${MULTITURN_MODEL_NAME:=Qwen/Qwen3-14B}"
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

# ==================== 路径配置 ====================
PROJECT_DIR="/home/liuguanming/Multimodal-Agent/UserRL"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"

# 模型路径 - 使用 Qwen3-4B
MODEL_PATH="/home/liuguanming/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/531c80e289d6cff3a7cd8c0db8110231d23a6f7a"

# 数据路径 - IntentionGym 专用
TRAIN_DATA="$PROJECT_DIR/data/intention_multiturn/train.parquet"
VAL_DATA="$PROJECT_DIR/data/intention_multiturn/test.parquet"

# ==================== 训练参数 ====================
# 4B 模型显存优化：减小 batch size 以适应更大的模型
TRAIN_BATCH_SIZE=16           # 4B 模型显存占用更大，从 64 减小到 32
MAX_PROMPT_LENGTH=1152
MAX_RESPONSE_LENGTH=4096
LEARNING_RATE=1e-6
TOTAL_EPOCHS=20               # 增加训练轮数，从 10 增加到 20

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
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.60 \
    actor_rollout_ref.rollout.n=2 \
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
    trainer.experiment_name='IntentionGym_Qwen4B_dualGPU_UserSimulatorQwen14B' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=1 \
    trainer.test_freq=5 \
    trainer.val_before_train=False \
    trainer.total_epochs=$TOTAL_EPOCHS $@

