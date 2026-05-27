# Integrating BlockTT / SVD / LoRA / QLoRA PEFT into verl

Status: Approved (design only, no code yet)
Author: yequan
Date: 2026-05-26

## Goal

Add four parameter-efficient training modes to verl so that both GRPO (`grpo.sh`)
and on-policy distillation (`on_policy_distillation.sh`) can train under any of
them with a single env-var switch:

1. **LoRA** — already supported by verl; this design re-homes its config into a
   new `peft.*` group while keeping behavior identical.
2. **QLoRA** — bnb 4-bit base + bf16 LoRA adapters. New.
3. **BlockTT** — Block tensor-train factorization from `src/compress`, in three
   variants: plain (SVD/QR init), calibrated (`calib_mode=v2`/`twosteps`), and
   **qfura** (BlockTT with NF4-quantized frozen core via `QBTTLinear`).
4. **SVD** — `SVDCompressedLinear` from `src/compress`, plain or calibrated
   (`calib_mode=svd_v2`).

All four modes share one config dataclass (`PEFTConfig`), one strategy
interface (`PEFTAdapter`), and the same checkpoint/eval/resume contract.

`PEFT_MODE=none` is the default; existing `bash grpo.sh` and
`bash on_policy_distillation.sh` invocations remain byte-identical to today.

## Non-goals

- Megatron backend (only FSDP).
- Ref policy or reward model PEFT (actor only).
- Critic PEFT (no critic in grpo/OPD; verl's existing critic LoRA path is
  untouched).
- AWQ/GPTQ of the teacher (reward_model slot).
- New compress algorithms beyond what `src/compress/integration.py` exposes
  today.

## Architecture

A new package `verl/verl/workers/peft/` introduces a `PEFTAdapter` strategy
interface plus five concrete adapters. The actor worker constructs one adapter
from `peft.*` config and routes three concerns through it:

1. **apply** — wrap or rewrite the HF model before FSDP wrap.
2. **export_for_vllm** — produce dense tensors for rollout sync, or signal
   "use the existing LoRA-aware path".
3. **save_pretrained** — write a checkpoint format that downstream eval can
   load with stock `AutoModelForCausalLM.from_pretrained` (compress/none) or
   `PeftModel.from_pretrained` (LoRA/QLoRA).

### Files touched

- `verl/verl/workers/peft/` (new package):
  - `base.py` — `PEFTAdapter` ABC + `NullAdapter`.
  - `lora.py`, `qlora.py`, `blocktt.py`, `svd.py` — concrete adapters.
  - `__init__.py` — `PEFTAdapter.from_config` factory.
- `verl/verl/workers/config/peft.py` (new) — `PEFTConfig` dataclass.
- `verl/verl/workers/fsdp_workers.py` — replace inline `if self._is_lora:` block
  in `ActorRolloutRefWorker._build_model_optimizer` with adapter dispatch;
  replace the LoRA-specific branch in `save_checkpoint`.
- `verl/verl/workers/sharding_manager/fsdp_vllm.py` — in `__enter__`'s param
  collection and `update_params`, consult `adapter.export_for_vllm`; on `None`,
  fall through to verl's existing path.
- `verl/verl/workers/config/model.py` — keep legacy `lora_rank` / `lora_alpha` /
  `target_modules` / `lora_adapter_path` fields, but treat them as a
  deprecated shim that populates `peft.lora.*` if `peft.mode` is unset.
- `verl/verl/trainer/config/ppo_trainer.yaml` and
  `_generated_ppo_trainer.yaml` — add the `peft.*` group under
  `actor_rollout_ref`.
- `grpo.sh`, `on_policy_distillation.sh` — add `$PEFT_ARGS` block.
- `scripts/val/eval/gen_vllm.py` — detect `adapter_config.json` in the model
  dir and load as PEFT-on-base when present.

### Config tree

```
actor_rollout_ref.peft
├── mode: none | lora | qlora | blocktt | svd
├── target_modules: "all" | "mlp" | "attn" | list[str]
├── lora:
│   ├── rank: int = 0
│   ├── alpha: int = 16
│   ├── dropout: float = 0.0
│   ├── bias: "none" | "all" | "lora_only" = "none"
│   └── adapter_path: Optional[str] = null   # resume from pre-trained adapter
├── qlora:
│   ├── bnb_4bit_quant_type: "nf4" | "fp4" = "nf4"
│   ├── bnb_4bit_compute_dtype: "bfloat16" | "float16" = "bfloat16"
│   └── bnb_4bit_use_double_quant: bool = true
├── blocktt:
│   ├── decomp_mode: str = "input_one_block"
│   │   # scalar input_one_block | output_one_block | input | output, OR
│   │   # dict literal: qkv=...,o=...,mlp_upgate=...,mlp_down=...
│   ├── rank: "full" | int | float = "full"
│   │   # "full" or int for plain mode; float in (0,1] for calibrated mode
│   ├── convert_mode: "svd" | "qr" = "svd"
│   ├── train_position: "small" | "large" | "both" = "small"
│   ├── s_merged_to: str = "frozen"
│   │   # frozen | trainable | output | input | split | keep_frozen | keep_trainable
│   ├── factorize_by_head: bool = true
│   ├── train_bias: bool = true
│   ├── normalize_after_update: bool = false
│   └── qfura:
│       └── enabled: bool = false   # NF4-quantize the frozen core
├── svd:
│   ├── train_position: "output" | "input" | "both" = "output"
│   ├── s_merged_to: str = "frozen"
│   └── compression_ratio: float = 1.0   # only used with calib.mode = svd_v2
└── calib:
    ├── mode: "none" | "v2" | "v2_bp" | "v2_combined" | "twosteps" |
    │           "svd_v2" | "svd_v2_combined" = "none"
    ├── source: "c4" | "traces" | "training_data" = "c4"
    ├── traces_path: Optional[str] = null
    ├── num_seqs: int = 128
    ├── max_length: int = 2048
    ├── batch_size: int = 8
    ├── seed: int = 3
    └── cpu_offload: bool = false   # run calibration on CPU to cut peak GPU mem
```

## PEFTAdapter interface

`verl/workers/peft/base.py`:

```python
class PEFTAdapter:
    mode: str   # class attribute

    @classmethod
    def from_config(cls, peft_cfg, *, model_config) -> "PEFTAdapter": ...

    def needs_calibration(self) -> bool: ...

    def apply(self, model, *, tokenizer, calib_loader_builder) -> nn.Module: ...

    def export_for_vllm(self, fsdp_module) -> dict[str, Tensor] | None:
        # None ⇒ fall through to verl's existing weight-collection path
        # (which handles LoRA via collect_lora_params + TensorLoRARequest).
        ...

    def vllm_engine_kwargs(self) -> dict: ...

    def save_pretrained(self, fsdp_module, out_dir: str) -> None: ...

    def topology_meta(self) -> dict: ...

    @classmethod
    def rebuild_from_meta(cls, model, meta: dict) -> nn.Module: ...

    def peft_config(self):
        # Returned to FSDPVLLMShardingManager.update_params so it can build
        # a TensorLoRARequest. None for non-LoRA modes.
        return None
```

### Per-adapter behavior

| Mode | `apply` | `export_for_vllm` | `save_pretrained` | `vllm_engine_kwargs` |
|---|---|---|---|---|
| **NullAdapter** | identity | `None` | `model.save_pretrained` | `{}` |
| **LoRAAdapter** | `get_peft_model(LoraConfig(...))` | `None` (verl's existing path) | PEFT adapter dir | `{enable_lora=True, max_loras=1, max_lora_rank=…}` |
| **QLoRAAdapter** | reload base with `BitsAndBytesConfig(load_in_4bit=…)`, then `get_peft_model` | `None` | PEFT adapter dir + `base_model_path.txt` | same as LoRA |
| **BlockTTAdapter** | plain: `convert_linear_to_btt_compress` + `configure_compress_btt_trainability`; calib: `apply_calibrated_btt(model, args, calib_loader=calib_loader_builder())`; qfura: `convert_and_quantize_linear_to_qbtt_streaming` | `materialize_calibrated_btt_weights(model)` | `materialize_calibrated_btt_to_linear` → `save_pretrained` | `{}` |
| **SVDAdapter** | plain: `convert_linear_to_svd_compress` + `configure_compress_svd_trainability`; calib: `apply_calibrated_svd(model, args, calib_loader=calib_loader_builder())` | iterate `SVDCompressedLinear`, `materialize_dense_weight()` per layer | `materialize_svd_to_linear` → `save_pretrained` | `{}` |

`NullAdapter` lets `mode="none"` flow through the same code paths as every
other mode, so the worker never special-cases "no PEFT".

## Data flow

### Init flow (per actor worker rank, in `_build_model_optimizer`)

```
1. Load HF model:
   - lora / blocktt / svd / none:
       AutoModelForCausalLM.from_pretrained(path, torch_dtype=…)
   - qlora:
       AutoModelForCausalLM.from_pretrained(path,
         quantization_config=BitsAndBytesConfig(load_in_4bit=True,
           bnb_4bit_quant_type=peft.qlora.bnb_4bit_quant_type,
           bnb_4bit_compute_dtype=peft.qlora.bnb_4bit_compute_dtype,
           bnb_4bit_use_double_quant=peft.qlora.bnb_4bit_use_double_quant))

2. adapter = PEFTAdapter.from_config(peft_cfg, model_config=…)

3. If peft_meta.json exists at trainer.default_local_dir:
       meta = json.load(default_local_dir / "peft_meta.json")
       # Resume-time CLI/meta consistency check (see Risk 6).
       compare_peft_meta_to_cli(meta, peft_cfg)
       model = adapter.rebuild_from_meta(model, meta)
       # FSDP shards loaded by the existing verl resume path.
   Else:
       calib_loader = None
       if adapter.needs_calibration():
           calib_loader = build_calib_loader(peft.calib, tokenizer=…, …)
       model = adapter.apply(model, tokenizer=…,
                             calib_loader_builder=lambda: calib_loader)

4. apply_monkey_patch(…) , gradient_checkpointing_enable, model.to(torch_dtype)

5. FSDP wrap:
       auto_wrap_policy = get_fsdp_wrap_policy(
           module=model,
           config=fsdp_config.get("wrap_policy", None),
           is_lora=(adapter.mode in {"lora", "qlora"}),
       )

6. Optimizer over [p for p in model.parameters() if p.requires_grad].
```

Calibration runs on rank 0 only (compress's `decompose_with_loader` is
single-device). Other ranks wait at the post-apply barrier. For a 7B model on
A800 80 GB this fits; for tighter cards, `peft.calib.cpu_offload=true` opts in
to CPU calibration.

### Training step

Unchanged. `dp_actor.update_policy` reads FSDP-wrapped params exactly as
before. BlockTT/SVD factors are leaf parameters; FSDP shards them. Frozen
cores (set by `configure_compress_btt_trainability` /
`configure_compress_svd_trainability`) have `requires_grad=False` and never
enter the optimizer.

### Rollout sync (each rollout, in `FSDPVLLMShardingManager.__enter__` → `update_params`)

```
with FSDP.summon_full_params(actor_module, writeback=False):
    exported = adapter.export_for_vllm(actor_module)
    peft_config = adapter.peft_config()

if exported is not None:
    # blocktt / svd / null path: full dense base sync, no TensorLoRARequest.
    params_for_vllm = exported
else:
    # lora / qlora path: keep verl's existing logic exactly.
    params_for_vllm = collect_lora_params(…)

self.update_params(params_for_vllm, peft_config=peft_config)
```

For BlockTT/SVD, `export_for_vllm` emits the *base-model* parameter names
(via `module.named_modules()` walks), never the factor names. vLLM's
`model.load_weights` accepts them as a normal full-weight sync. `base_sync_done`
stays meaningful: compress modes are always full-base sync (cores change every
step), LoRA flips to adapter-only after the first sync.

**QLoRA**: the actor holds an `nn.Linear4bit` base + bf16 LoRA adapters. vLLM
runs the base in bf16 (loaded from `model.path` at engine init) and only the
adapter is pushed via TensorLoRARequest. This requires `model.path` to point at
the unquantized HF model; quantization happens in-worker only. Documented as a
config constraint.

### Save flow (every `save_freq` steps, in `save_checkpoint`)

```
# Existing FSDP shard save — unchanged.
…

# New: HF-format eval-loadable artifact.
adapter.save_pretrained(actor_module, out_dir / "merged_hf")

# New: sidecar metadata (rank 0 only, written once on first save):
if rank == 0 and not (out_dir / "peft_meta.json").exists():
    json.dump(adapter.topology_meta(), out_dir / "peft_meta.json")
```

### Resume flow

```
if (default_local_dir / "peft_meta.json").exists():
    1. Load HF model (or 4-bit reload for QLoRA).
    2. compare_peft_meta_to_cli(meta, peft_cfg)   # errors on drift (Risk 6)
    3. adapter.rebuild_from_meta(model, meta):
       - blocktt: rebuild_btt_from_topology(model, btt_topology.json)
       - svd:     replace target nn.Linear with SVDCompressedLinear at stored ranks
       - lora/qlora: get_peft_model(LoraConfig(**meta["lora_config"]))
       - null:    no-op
    4. FSDP wrap + load FSDP shards (existing verl resume path).
else:
    Cold start (run apply() with optional calibration).
```

Resume never re-calibrates.

## On-disk checkpoint layout

```
<default_local_dir>/
  global_step_N/
    actor/                          # existing FSDP shard format, unchanged
      __0_0.distcp, …
      fsdp_config.json
    merged_hf/                      # NEW: produced by adapter.save_pretrained()
      config.json, generation_config.json, tokenizer.*
      # Compress (blocktt, svd) and full (none):
      model.safetensors / model-*.safetensors + model.safetensors.index.json
      # LoRA / QLoRA:
      adapter_config.json
      adapter_model.safetensors
      base_model_path.txt           # records actor_rollout_ref.model.path (QLoRA only;
                                    # LoRA stores base path in adapter_config.json already)
    peft_meta.json                  # NEW (rank 0, written on first save):
                                    #   {mode, target_modules, lora{}, blocktt{},
                                    #    svd{}, calib{}, qfura, compress_topology}
    compress/                       # NEW, blocktt only:
      btt_topology.json
  latest_checkpointed_iteration.txt
```

`merged_hf/` is the eval-facing artifact; `actor/` + `peft_meta.json` +
`compress/` together are the resume-facing artifact. They are independent.

### Loadability contracts

| Mode | `merged_hf/` loaded by | Resume rebuilds via |
|---|---|---|
| `none` | `AutoModelForCausalLM.from_pretrained(merged_hf)` | nothing — FSDP shards load into plain HF model |
| `lora` | `PeftModel.from_pretrained(base, merged_hf)` | `get_peft_model(LoraConfig(**peft_meta.lora))` + FSDP shard load |
| `qlora` | same as lora; vLLM loads `from_pretrained(model.path)` bf16 base + adapter | same as lora; 4-bit reload uses `peft_meta.qlora.*` |
| `blocktt` (plain/calib/qfura) | `AutoModelForCausalLM.from_pretrained(merged_hf)` | `rebuild_btt_from_topology(model, btt_topology.json)` + FSDP shard load |
| `svd` (plain/calib) | `AutoModelForCausalLM.from_pretrained(merged_hf)` | re-insert `SVDCompressedLinear` at stored ranks + FSDP shard load |

Compress and `none` collapse to a single eval path (stock `from_pretrained`).
LoRA/QLoRA share a second path (base + adapter). No mode requires bespoke
eval awareness beyond that two-way split.

### Eval-side changes

Only one downstream tool needs to change:

- **`scripts/val/eval/gen_vllm.py`** — when `MODEL_NAMES` points at a dir
  containing `adapter_config.json`, read it (and `base_model_path.txt` for
  QLoRA, else the base path is in `adapter_config.json`), load the base via
  vLLM with `enable_lora=True`, and send the adapter as a LoRARequest. If no
  `adapter_config.json`, current behavior (load as full model).

No change needed in `scripts/val/eval/grade.py`, `scripts/infer/vllm_rollout.py`,
or verl's in-training validation (it goes through the same rollout engine that
the adapter already configured via `vllm_engine_kwargs`).

## Launch script changes

`grpo.sh` and `on_policy_distillation.sh` each get a `PEFT_MODE`-driven
`$PEFT_ARGS` block; defaults are `PEFT_MODE=none` so existing invocations are
unchanged. `CKPT_PATH` and `EXPERIMENT_NAME` append `_$PEFT_MODE` to avoid
collisions.

```bash
export PEFT_MODE=${PEFT_MODE:-none}
export PEFT_TARGET_MODULES=${PEFT_TARGET_MODULES:-all}

export LORA_RANK=${LORA_RANK:-16}
export LORA_ALPHA=${LORA_ALPHA:-32}
export LORA_DROPOUT=${LORA_DROPOUT:-0.0}

export QLORA_QUANT_TYPE=${QLORA_QUANT_TYPE:-nf4}
export QLORA_DOUBLE_QUANT=${QLORA_DOUBLE_QUANT:-True}
export QLORA_COMPUTE_DTYPE=${QLORA_COMPUTE_DTYPE:-bfloat16}

export BTT_DECOMP_MODE=${BTT_DECOMP_MODE:-input_one_block}
export BTT_RANK=${BTT_RANK:-full}
export BTT_TRAIN_POSITION=${BTT_TRAIN_POSITION:-small}
export BTT_S_MERGED_TO=${BTT_S_MERGED_TO:-frozen}
export BTT_CONVERT_MODE=${BTT_CONVERT_MODE:-svd}
export BTT_FACTORIZE_BY_HEAD=${BTT_FACTORIZE_BY_HEAD:-True}
export BTT_NORMALIZE_AFTER_UPDATE=${BTT_NORMALIZE_AFTER_UPDATE:-False}
export BTT_QFURA=${BTT_QFURA:-False}

export SVD_TRAIN_POSITION=${SVD_TRAIN_POSITION:-output}
export SVD_S_MERGED_TO=${SVD_S_MERGED_TO:-frozen}
export SVD_COMPRESSION_RATIO=${SVD_COMPRESSION_RATIO:-1.0}

export CALIB_MODE=${CALIB_MODE:-none}
export CALIB_SOURCE=${CALIB_SOURCE:-c4}
export CALIB_NUM_SEQS=${CALIB_NUM_SEQS:-128}
export CALIB_MAX_LENGTH=${CALIB_MAX_LENGTH:-2048}
export CALIB_BATCH_SIZE=${CALIB_BATCH_SIZE:-8}
export CALIB_SEED=${CALIB_SEED:-3}
export CALIB_TRACES_PATH=${CALIB_TRACES_PATH:-}

PEFT_ARGS="+actor_rollout_ref.peft.mode=$PEFT_MODE \
+actor_rollout_ref.peft.target_modules=$PEFT_TARGET_MODULES"

case "$PEFT_MODE" in
  none) ;;
  lora)
    PEFT_ARGS="$PEFT_ARGS \
      +actor_rollout_ref.peft.lora.rank=$LORA_RANK \
      +actor_rollout_ref.peft.lora.alpha=$LORA_ALPHA \
      +actor_rollout_ref.peft.lora.dropout=$LORA_DROPOUT" ;;
  qlora)
    PEFT_ARGS="$PEFT_ARGS \
      +actor_rollout_ref.peft.lora.rank=$LORA_RANK \
      +actor_rollout_ref.peft.lora.alpha=$LORA_ALPHA \
      +actor_rollout_ref.peft.qlora.bnb_4bit_quant_type=$QLORA_QUANT_TYPE \
      +actor_rollout_ref.peft.qlora.bnb_4bit_use_double_quant=$QLORA_DOUBLE_QUANT \
      +actor_rollout_ref.peft.qlora.bnb_4bit_compute_dtype=$QLORA_COMPUTE_DTYPE" ;;
  blocktt)
    PEFT_ARGS="$PEFT_ARGS \
      +actor_rollout_ref.peft.blocktt.decomp_mode=$BTT_DECOMP_MODE \
      +actor_rollout_ref.peft.blocktt.rank=$BTT_RANK \
      +actor_rollout_ref.peft.blocktt.train_position=$BTT_TRAIN_POSITION \
      +actor_rollout_ref.peft.blocktt.s_merged_to=$BTT_S_MERGED_TO \
      +actor_rollout_ref.peft.blocktt.convert_mode=$BTT_CONVERT_MODE \
      +actor_rollout_ref.peft.blocktt.factorize_by_head=$BTT_FACTORIZE_BY_HEAD \
      +actor_rollout_ref.peft.blocktt.normalize_after_update=$BTT_NORMALIZE_AFTER_UPDATE \
      +actor_rollout_ref.peft.blocktt.qfura.enabled=$BTT_QFURA" ;;
  svd)
    PEFT_ARGS="$PEFT_ARGS \
      +actor_rollout_ref.peft.svd.train_position=$SVD_TRAIN_POSITION \
      +actor_rollout_ref.peft.svd.s_merged_to=$SVD_S_MERGED_TO \
      +actor_rollout_ref.peft.svd.compression_ratio=$SVD_COMPRESSION_RATIO" ;;
  *) echo "Unknown PEFT_MODE=$PEFT_MODE" >&2; exit 1 ;;
esac

if [ "$CALIB_MODE" != "none" ]; then
  PEFT_ARGS="$PEFT_ARGS \
    +actor_rollout_ref.peft.calib.mode=$CALIB_MODE \
    +actor_rollout_ref.peft.calib.source=$CALIB_SOURCE \
    +actor_rollout_ref.peft.calib.num_seqs=$CALIB_NUM_SEQS \
    +actor_rollout_ref.peft.calib.max_length=$CALIB_MAX_LENGTH \
    +actor_rollout_ref.peft.calib.batch_size=$CALIB_BATCH_SIZE \
    +actor_rollout_ref.peft.calib.seed=$CALIB_SEED"
  if [ -n "$CALIB_TRACES_PATH" ]; then
    PEFT_ARGS="$PEFT_ARGS +actor_rollout_ref.peft.calib.traces_path=$CALIB_TRACES_PATH"
  fi
fi

python3 -m verl.trainer.main_ppo \
    … existing args … \
    $PEFT_ARGS
```

### Example invocations

```bash
# Plain BlockTT OPD, input_one_block, small side trainable (run_rl.py default)
PEFT_MODE=blocktt BTT_DECOMP_MODE=input_one_block BTT_TRAIN_POSITION=small \
  bash on_policy_distillation.sh

# qfura: BlockTT with NF4-quantized frozen core
PEFT_MODE=blocktt BTT_QFURA=True BTT_TRAIN_POSITION=small \
  bash on_policy_distillation.sh

# Calibrated BlockTT (v2) for GRPO with traces source
PEFT_MODE=blocktt CALIB_MODE=v2 CALIB_SOURCE=traces \
  CALIB_TRACES_PATH=datasets/traces/qwen3_4b_dapo.jsonl \
  bash grpo.sh

# Calibrated SVD with 0.5 compression ratio
PEFT_MODE=svd CALIB_MODE=svd_v2 SVD_TRAIN_POSITION=output SVD_COMPRESSION_RATIO=0.5 \
  bash grpo.sh

# QLoRA OPD on 7B
PEFT_MODE=qlora LORA_RANK=32 LORA_ALPHA=64 \
  bash on_policy_distillation.sh

# Plain LoRA — exercises verl's existing PEFT path under the new config
PEFT_MODE=lora LORA_RANK=16 LORA_ALPHA=32 \
  bash on_policy_distillation.sh
```

### Back-compat shim for legacy LoRA users

If a YAML or override sets `actor_rollout_ref.model.lora_rank=N` without
setting `actor_rollout_ref.peft.mode`, `HFModelConfig.__post_init__` copies
the legacy fields into `peft.mode=lora`, `peft.lora.rank=N`, etc., and emits
a one-line deprecation log. Every existing script keeps working.

## Testing

### Unit (`verl/tests/peft/`, fast)

- `test_peft_config.py` — Hydra parse → `PEFTConfig` dataclass; legacy
  `model.lora_*` shim populates `peft.lora.*` correctly.
- `test_adapter_apply.py` — on a 2-layer tiny model, for each adapter:
  module types after apply, `requires_grad` correctness, target-module
  coverage by `peft.target_modules`.
- `test_export_for_vllm.py` — dense materialization matches dense matmul,
  exported key set equals original `nn.Linear` parameter keys.
- `test_checkpoint_roundtrip.py` — save → reload (stock `from_pretrained` for
  compress/none, `PeftModel.from_pretrained` for lora/qlora) → logit parity
  within 1e-4 on a fixed input. Matrix covers
  `{none, lora, qlora, blocktt-plain, blocktt-calib, blocktt-qfura,
    svd-plain, svd-calib}`.
- `test_resume_topology.py` — write `peft_meta.json` + `btt_topology.json`,
  call `rebuild_from_meta` on a fresh model, verify module topology matches.

### Integration (`pytest -m gpu`, one GPU, ~5 min each)

- `test_actor_worker_init.py` — instantiate `ActorRolloutRefWorker` with each
  PEFT mode on a 0.5B model; verify post-FSDP-wrap state + `export_for_vllm`
  shape.
- `test_calibration_smoke.py` — `calib_mode=v2`, `calib_source=c4`,
  `calib_num_seqs=4` on a 0.5B model; assert BTT topology installed and at
  least one core changed from random init.

### End-to-end smoke (`scripts/peft_smoke.sh`, manual)

5-step run per mode on `dapo-math-17k-1percent.parquet`, asserting one full
save + resume cycle completes. Not pytest; documented in the README of the
new package.

## Risks and mitigations

1. **vLLM weight-name drift.** Compress modes emit
   `model.layers.N.<sub>.weight` keys. verl's `model.load_weights` already
   handles fused-QKV and other stacked-param renames for the full-base path
   (see `check_target_modules`, `stacked_params` in
   `sharding_manager/fsdp_vllm.py`). Compress modes go through the same call;
   smoke test on both Qwen3 (fused QKV) and Llama (separate QKV).

2. **FSDP wrap policy with custom modules.** `BTTLinear` and
   `SVDCompressedLinear` ride inside their parent transformer block's FSDP
   unit (transformer-block-level wrap policy unchanged). Smoke test confirms
   no extra wrap config is needed.

3. **Calibration peak memory at 7B.** Un-sharded model on rank 0 ≈ 14 GB bf16
   + activations for one calib batch. Fits A800 80 GB. For tighter cards,
   `peft.calib.cpu_offload=true` runs calibration on CPU (compress supports
   linear-by-linear streaming).

4. **QLoRA + FSDP interaction.** `Linear4bit` quantization state lives in
   buffers, not parameters. Follow huggingface/peft + FSDP recipe:
   `use_orig_params=True`, adapters sharded normally. Dedicated smoke test
   for QLoRA save → reload → forward parity.

5. **`base_sync_done` semantics.** Compress modes are full-base sync every
   step (factors change every update). Adapter signals this by always
   returning a non-`None` `export_for_vllm` value; the existing
   `base_sync_done` flag is harmless because the LoRA fast-path branch is
   never entered.

6. **Resume meta drift.** If the user changes any `peft.*` value between
   launches, `rebuild_from_meta` would silently use the stored value.
   Mitigation: on resume, compare every key in CLI `peft.*` against
   `peft_meta.json` and error on mismatch with a message pointing at
   `peft_meta.json` so the user can either delete the checkpoint or revert
   the override.

## Open questions (out of scope)

- Apply the same compress decomposition to the ref policy so KL is between
  "BTT-compressed init" and "trained BTT" — clean for certain ablations,
  needs ref to share calibration with actor.
- AWQ/GPTQ for the reward_model slot (teacher) — separate optimization, not
  PEFT integration.
- Megatron backend — separate `peft/megatron_*.py` package, future work.

## Summary of shipped surface

- New `verl/workers/peft/` package: `PEFTAdapter` ABC + 5 adapters (None,
  LoRA, QLoRA, BlockTT, SVD). BlockTT covers plain + calibrated + qfura.
- New `verl/workers/config/peft.py` dataclass and `peft.*` Hydra group.
  Legacy `model.lora_*` still works via shim.
- Edits to `fsdp_workers.py` (actor only), `sharding_manager/fsdp_vllm.py`,
  and `save_checkpoint` to delegate through the adapter.
- One eval-side change: `scripts/val/eval/gen_vllm.py` detects LoRA dirs.
- `grpo.sh` / `on_policy_distillation.sh` gain a `PEFT_MODE`-driven
  `$PEFT_ARGS` block; default `PEFT_MODE=none` is byte-identical to today.
- Tests covering checkpoint round-trip, calibration smoke, resume topology
  rebuild, and per-adapter `apply`.
