# BlockTT / SVD SFT recipes

These configs drive `src/compress`-backed BlockTT and SVD finetuning from
LlamaFactory. They mirror the `--train-mode blocktt` and `--train-mode svd`
paths in `run_rl.py`, but for SFT instead of RL.

## Recipes

- `qwen3_base_blocktt_sft.yaml` — plain BTT (lossless decomposition + finetune).
- `qwen3_base_blocktt_calibrated_sft.yaml` — calibrated BTT (`calib_mode: v2`).
- `qwen3_base_svd_sft.yaml` — plain SVD.

Run with:

```bash
conda activate sft
llamafactory-cli train LlamaFactory/examples/train_blocktt/qwen3_base_blocktt_sft.yaml
```

## YAML knobs

| Key | Used by | Meaning |
|---|---|---|
| `finetuning_type` | both | `blocktt` or `svd`. |
| `trainable_type` | both | `all` / `mlp` / `attn` — which modules get compressed. |
| `train_position` | both | blocktt: `small` / `large` / `both`. svd: `output` / `input` / `both`. |
| `s_merged_to` | both | `frozen` / `trainable` / `output` / `input` / `split` / `keep_frozen` / `keep_trainable`. |
| `decomp_mode` | blocktt | `input_one_block` / `output_one_block` or dict literal. |
| `blocktt_rank` | blocktt | `"full"` or positive integer string. For calibrated mode use `"full"` or a float in `(0, 1]`. |
| `convert_mode` | blocktt | `svd` (default) or `qr`. `qr` ignores `s_merged_to`. |
| `train_bias` | blocktt | Train BTT biases. |
| `blocktt_normalize_after_update` | blocktt | Normalize trainable cores after each step. |
| `blocktt_factorize_by_head` | blocktt | Align attention BTT blocks with head structure. |
| `calib_mode` | both | `none` / `v2` / `v2_bp` / `v2_combined` / `twosteps` for BTT; `svd_v2` / `svd_v2_combined` for SVD. |
| `calib_source` | both | `c4` / `traces` / `training_data`. |
| `calib_num_seqs`, `calib_max_length`, `calib_seed`, `calib_batch_size` | both | Calibration sampling. |
| `calib_traces_path` | both | Required when `calib_source=traces`. |
| `compression_ratio` | svd calibrated | Fraction of compressible params to retain, `(0, 1]`. |

## DeepSpeed

ZeRO-2 is supported (and used by default in these recipes). **ZeRO-3 is
rejected at config-parse time** — custom BTT/SVD layers don't survive
parameter sharding under ZeRO-3.

## Checkpoint layout

Per save:

```
output_dir/
  checkpoint-200/             # factored state_dict (BTT/SVD modules);
                              # use this for resume_from_checkpoint
  checkpoint-200-merged/      # dense HF weights; drop-in for vLLM / eval
  ...
  final-merged/               # written at end-of-train, regardless of save_steps
```

## Caveats

- `learning_rate: 1.0e-4` is seeded from `run_rl.py`'s `MODE_DEFAULTS`. Tune for SFT.
- BlockTT/SVD cannot be combined with GaLore, APOLLO, or BAdam.
- `enable_liger_kernel: true` is fine — Liger patches HF attention/MLP modules but does not replace the inner `nn.Linear`, so BlockTT/SVD conversion (which runs in `init_adapter` after Liger patching) sees the post-Liger graph.
