### SFT dataset preparation
python scripts/infer/vllm_rollout.py \
  --input-parquet datasets/OpenThoughts3-1.2M-math.parquet \
  --model-path Kevin16/Qwen3-4B-Non-Thinking-RL-Math \
  --gpu-ids 2,3 \
  --enable-thinking false \
  --enable-rejection-sampling true \
  --max-attempts-per-rollout 3

### SFT
cd LlamaFactory
CUDA_VISIBLE_DEVICES=7 WANDB_PROJECT=Qwen3_1.7B_openthoughts_sft llamafactory-cli train examples/train_full/qwen3_base_full_sft.yaml deepspeed=

### GRPO
bash grpo.sh

### OPD
bash on_policy_distillation.sh


CUDA_VISIBLE_DEVICES=7 bash scripts/opd/math/full.sh > /dev/null 2>&1 &

CUDA_VISIBLE_DEVICES=7 bash scripts/compress_opd/math/btt_v2_combined_opd.sh > /dev/null 2>&1 &