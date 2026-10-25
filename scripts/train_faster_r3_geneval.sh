#!/bin/bash
# Faster-R3: the measured 8-GPU GenEval recipe (docs/training.md#launch).
# Optimizations are enabled by default. Set R3_SOURCE_TREE to an upstream checkout for the baseline.
# Needs the GenEval reward server at REWARD_URL (scripts/reward/gunicorn_geneval.py).
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_PATH="${MODEL_PATH:?set MODEL_PATH to the BAGEL-7B-MoT checkpoint directory}"
LOG_DIR="${LOG_DIR:-./output}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NUM_SHARD="${NUM_SHARD:-${NPROC_PER_NODE}}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29509}"
REWARD_URL="${REWARD_URL:-http://127.0.0.1:18085}"
EXP_NAME="${EXP_NAME:-faster_r3_geneval}"
TOTAL_STEPS="${TOTAL_STEPS:-1000}"
GLOBAL_SEED="${GLOBAL_SEED:-4396}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-./data/rl_train/train_geneval.jsonl}"
VAL_DATA_PATH="${VAL_DATA_PATH:-}"

# Both measured sides use ZeRO-3 and this identical loading adaptation.
export R3_LOAD_CONCURRENCY="${R3_LOAD_CONCURRENCY:-8}"
if (( NPROC_PER_NODE != 8 || NUM_SHARD != 8 )); then
  echo "This launcher reproduces the measured 8-rank / 8-shard setup." >&2
  exit 2
fi
mkdir -p "${LOG_DIR}"
if [[ -z "${VAL_DATA_PATH}" ]]; then
  # The measured runs evaluated on the first NPROC_PER_NODE prompts of the GenEval test split (step 0 and every 50 steps).
  VAL_DATA_PATH="${LOG_DIR}/geneval_eval_first${NPROC_PER_NODE}.jsonl"
  head -n "${NPROC_PER_NODE}" data/rl_test/test_geneval.jsonl > "${VAL_DATA_PATH}"
fi

exec torchrun --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" \
  --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
  scripts/h100_entry.py \
  --dataset_name geneval \
  --llm_path "${MODEL_PATH}/llm_config.json" \
  --vae_path "${MODEL_PATH}/ae.safetensors" \
  --vit_path "${MODEL_PATH}/vit_config.json" \
  --train_data_path "${TRAIN_DATA_PATH}" \
  --val_data_path "${VAL_DATA_PATH}" \
  --debug False \
  --log_dir "${LOG_DIR}" \
  --wandb_project R3 \
  --exp_name "${EXP_NAME}" \
  --resume_from "${MODEL_PATH}" \
  --resume_model_only True \
  --finetune_from_ema True \
  --auto_resume False \
  --llm_qk_norm True \
  --tie_word_embeddings False \
  --layer_module Qwen2MoTDecoderLayer \
  --max_output_token_n_gen 196 \
  --max_output_token_n_edit 384 \
  --rounds 2 \
  --group_size 16 \
  --policy_group_size 4 \
  --think_mode True \
  --reward_fn geneval \
  --reward_server_urls "${REWARD_URL}" \
  --reward_server_port -1 \
  --max_latent_size 64 \
  --wandb_name "${EXP_NAME}" \
  --wandb_offline True \
  --global_seed "${GLOBAL_SEED}" \
  --data_seed 42 \
  --num_shard "${NUM_SHARD}" \
  --total_steps "${TOTAL_STEPS}" \
  --save_every 50 \
  --ce_weight 1.0 \
  --mse_weight 2.0 \
  --kl_weight_text 0.001 \
  --kl_weight_image 0.001 \
  --freeze_vit True \
  --freeze_und False \
  --tune_text_cot True \
  --tune_image_cot True \
  --text_temperature 0.9 \
  --top_k 50 \
  --num_timesteps_gen 16 \
  --num_timesteps_edit 21 \
  --cfg_text_scale 4.0 \
  --cfg_img_scale 1.5 \
  --cfg_renorm_type_gen global \
  --cfg_renorm_type_edit text_channel \
  --lr 5e-6 \
  --warmup_steps 10 \
  --eps 1e-8 \
  --log_every 1 \
  --beta2 0.95 \
  --format_reward_weight 1.0 \
  --eta_mode monotonic \
  --constant_eta_gen 0.7 \
  --constant_eta_edit 1.0 \
  --pack_start_round_idx 0 \
  --enable_sde_gen True \
  --enable_sde_edit True \
  --eval_freq 50 \
  --client_type openai \
  --reward_api_key EMPTY \
  --reward_model_name Qwen2.5-VL-72B-Instruct-AWQ \
  --lr_scheduler constant \
  --max_grad_norm 1.0 \
  "$@"
