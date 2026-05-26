# Integrating BlockTT & SVD finetuning into LlamaFactory

**Status:** Design approved (awaiting user spec review)
**Date:** 2026-05-26
**Author:** yequan (with Claude)

## 1. Problem & scope

`src/compress` is the canonical home for BlockTT (`BTTLinear`) and SVD (`SVDCompressedLinear`) module conversion + finetuning. It is currently exercised only through `run_rl.py` (RL training). We want the same compression-aware finetuning available to **SFT runs driven by LlamaFactory**, via YAML configs analogous to `LlamaFactory/examples/train_full/qwen3_base_full_sft.yaml`.

**In scope**
- Two new `finetuning_type` literals: `blocktt`, `svd`.
- Plain (calibration-free) init AND calibrated init (c4 / traces / training_data sources).
- YAML examples under `LlamaFactory/examples/train_blocktt/`.
- Checkpoint export as plain HF dense `nn.Linear` weights at every save and at end-of-training.

**Out of scope**
- Modifying `src/compress` itself. We only call its public API.
- WebUI surface (`finetuning_type` will appear in the underlying parser; WebUI components are not extended).
- New trainer class (we use `CustomSeq2SeqTrainer` unchanged; behavior is layered via `TrainerCallback`s).
- ZeRO-3 support. Hard-error if user pairs `blocktt|svd` with a z3 config.

## 2. File-level changes

```
LlamaFactory/
  examples/train_blocktt/                                  (NEW)
    qwen3_base_blocktt_sft.yaml                            plain BTT
    qwen3_base_blocktt_calibrated_sft.yaml                 calibrated BTT
    qwen3_base_svd_sft.yaml                                plain SVD
    README.md                                              knob reference
  src/llamafactory/
    hparams/finetuning_args.py                             +CompressArguments mixin
                                                            extend finetuning_type Literal
                                                            __post_init__ validators
    model/adapter.py                                       dispatch to init_compress_model
    model/compress_setup.py                                (NEW) lazy compress import,
                                                            plain + calibrated conversion,
                                                            trainability config
    train/callbacks.py                                     +CompressNormalizeCallback
                                                            +CompressSaveCallback
    train/sft/workflow.py                                  register the two callbacks
```

No changes to: trainer classes, dataset/template code, eval pipeline, DeepSpeed configs, `src/compress`.

## 3. Hparams (CompressArguments mixin)

New `@dataclass CompressArguments` mixed into `FinetuningArguments`, same pattern as `GaloreArguments` / `ApolloArguments` / `BadamArgument`. Fields map 1:1 onto `run_rl.py` flags so the YAML reads like the CLI (underscore-style names).

```python
@dataclass
class CompressArguments:
    # Shared (blocktt + svd)
    trainable_type: Literal["all", "mlp", "attn"] = "all"
    train_position: Optional[Literal["output","input","small","large","both"]] = None
    s_merged_to: Optional[Literal["frozen","trainable","output","input",
                                   "split","keep_frozen","keep_trainable"]] = None

    # BlockTT-only
    decomp_mode: str = "input_one_block"            # scalar or dict literal
    blocktt_rank: str = "full"                      # "full" or positive int (string)
    convert_mode: Literal["svd", "qr"] = "svd"
    train_bias: bool = True                         # inverse of --no-train-bias
    blocktt_normalize_after_update: bool = False
    blocktt_factorize_by_head: bool = True

    # Calibrated init (mirrors add_calibrated_btt_args, hyphen_style=False)
    calib_mode: Literal["none","v2","v2_bp","v2_combined",
                        "twosteps","svd_v2","svd_v2_combined"] = "none"
    calib_source: Literal["c4","traces","training_data"] = "c4"
    calib_samples: int = 256
    calib_seq_len: int = 2048
    calib_batch_size: int = 1
    calib_traces_path: Optional[str] = None
    # Remaining --calib-* flags map 1:1 onto fields named calib_<flag>.
    # See compress.integration.add_calibrated_btt_args for the authoritative list.
```

`FinetuningArguments.finetuning_type` literal is extended from
`Literal["lora","oft","freeze","full"]` to
`Literal["lora","oft","freeze","full","blocktt","svd"]`. The assert on `__post_init__` is updated accordingly.

### 3.1 Validation (hard errors in `__post_init__`)

| Condition | Behavior |
|---|---|
| `finetuning_type in {blocktt,svd}` AND `"z3" in deepspeed` path | Raise `ValueError` |
| `finetuning_type == "blocktt"` AND `train_position not in {small,large,both,None}` | Raise |
| `finetuning_type == "svd"` AND `train_position not in {output,input,both,None}` | Raise |
| `finetuning_type == "blocktt"` AND `train_position == "both"` AND `s_merged_to in {frozen,trainable}` | Raise (mirrors `run_rl.py`) |
| `finetuning_type == "blocktt"` AND `convert_mode == "qr"` AND `s_merged_to is not None` | Warn-and-ignore (mirrors `run_rl.py`) |
| `calib_mode != "none"` AND `finetuning_type not in {blocktt,svd}` | Raise |
| `calib_mode in {svd_v2, svd_v2_combined}` AND `finetuning_type != "svd"` | Raise |
| `calib_mode in {v2, v2_bp, v2_combined, twosteps}` AND `finetuning_type != "blocktt"` | Raise |
| `blocktt_rank` not parseable as `"full"` or positive int | Raise |
| `finetuning_type in {blocktt,svd}` AND any of `use_galore` / `use_apollo` / `use_badam` | Raise |

### 3.2 Soft defaults (applied only when user left field None)

- `train_position` → `"small"` for blocktt, `"output"` for svd
- `s_merged_to`    → `"frozen"` for blocktt & svd

### 3.3 Cross-validation of calibrated subset

After the dataclass settles, `compress_setup.init_compress_model()` calls `compress.integration.validate_calibrated_btt_args(ns, argv=None, hyphen_style=False)` with an `argparse.Namespace` view of `FinetuningArguments`. This reuses the same validator `run_rl.py` uses, kept in sync with `src/compress` without duplication.

## 4. Adapter dispatch & compress_setup module

### 4.1 `model/adapter.py`

Add a new branch to `init_adapter()`, placed after the `freeze` / `lora|oft` / `full` branches:

```python
elif finetuning_args.finetuning_type in {"blocktt", "svd"}:
    model = init_compress_model(config, model, model_args, finetuning_args, is_trainable)
```

Trainable-param logging remains unchanged (it iterates over `model.named_parameters()` and `param.requires_grad`, which is exactly what `configure_compress_*_trainability` sets).

### 4.2 `model/compress_setup.py` (NEW)

Sole owner of the `from compress.integration import ...` line — keeps the rest of LlamaFactory unaware of `src/compress`. Sketch:

```python
def _ensure_compress_on_path() -> None:
    """Lazily prepend <repo_root>/src to sys.path. Idempotent."""
    repo_root = pathlib.Path(__file__).resolve().parents[4]   # …/OPD/
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

def init_compress_model(config, model, model_args, finetuning_args, is_trainable):
    if not is_trainable:
        return model     # inference loads dense HF; nothing to do
    _ensure_compress_on_path()
    from compress.integration import (
        apply_calibrated_btt, apply_calibrated_svd, build_calib_loader,
        convert_linear_to_btt_compress, convert_linear_to_svd_compress,
        configure_compress_btt_trainability, configure_compress_svd_trainability,
        get_blocktt_target_module_names, get_svd_target_module_names,
        resolve_blocktt_decomp_modes, validate_calibrated_btt_args,
    )
    fa = finetuning_args
    method = fa.finetuning_type

    if fa.calib_mode != "none":
        ns = _to_namespace(fa)
        validate_calibrated_btt_args(ns, argv=None, hyphen_style=False)
        loader = build_calib_loader(
            source=fa.calib_source, tokenizer=_load_tokenizer(model_args),
            samples=fa.calib_samples, seq_len=fa.calib_seq_len,
            batch_size=fa.calib_batch_size, traces_path=fa.calib_traces_path,
        )
        # apply_calibrated_btt handles BTT-family calib_modes (v2/v2_bp/
        # v2_combined/twosteps); apply_calibrated_svd handles SVD-family
        # calib_modes (svd_v2/svd_v2_combined). The mode <-> method match
        # was already enforced by the validators in Section 3.1.
        if method == "blocktt":
            apply_calibrated_btt(model, args_namespace=ns, calib_loader=loader)
        else:
            apply_calibrated_svd(model, args_namespace=ns, calib_loader=loader)
    else:
        if method == "blocktt":
            targets = get_blocktt_target_module_names(fa.trainable_type)
            decomp_mode, module_decomp_modes = resolve_blocktt_decomp_modes(
                fa.decomp_mode, include_names=targets,
            )
            convert_linear_to_btt_compress(
                model, target_module_names=targets,
                module_decomp_modes=module_decomp_modes,
                rank=_resolve_rank(fa.blocktt_rank),
                convert_mode=fa.convert_mode,
                s_merged_to=fa.s_merged_to,
                factorize_by_head=fa.blocktt_factorize_by_head,
            )
            configure_compress_btt_trainability(
                model, train_position=fa.train_position, train_bias=fa.train_bias,
            )
        else:                                                 # "svd"
            targets = get_svd_target_module_names(fa.trainable_type)
            convert_linear_to_svd_compress(
                model, target_module_names=targets, s_merged_to=fa.s_merged_to,
            )
            configure_compress_svd_trainability(
                model, train_position=fa.train_position,
            )
    return model
```

`_to_namespace(fa)` builds an `argparse.Namespace` view so `validate_calibrated_btt_args` / `apply_calibrated_btt` (designed against `run_rl.py`'s argparse) keep working unchanged.

`_resolve_rank` parses the `blocktt_rank` string and returns whatever `convert_linear_to_btt_compress`'s `rank` argument accepts: the literal string `"full"` for the lossless case, or `int(s)` for a positive integer string. Invalid input raises `ValueError` (also enforced by the validator in Section 3.1). Mirrors `run_rl.py::resolve_blocktt_rank`.

`_load_tokenizer(model_args)` loads the tokenizer lazily for calib only — avoids importing `transformers.AutoTokenizer` on every adapter init.

## 5. Callbacks

Registered only when `finetuning_args.finetuning_type in {blocktt, svd}`. Both live in `LlamaFactory/src/llamafactory/train/callbacks.py`.

### 5.1 CompressNormalizeCallback

Mirrors `run_rl.py`'s post-optimizer-step normalization of trainable BTT cores.

```python
class CompressNormalizeCallback(TrainerCallback):
    def __init__(self, finetuning_args):
        self.enabled = (
            finetuning_args.finetuning_type == "blocktt"
            and finetuning_args.blocktt_normalize_after_update
        )
    def on_step_end(self, args, state, control, model=None, **kwargs):
        if not self.enabled or model is None:
            return
        from compress.integration import normalize_trainable_blocktt_cores_
        normalize_trainable_blocktt_cores_(model)
```

`on_step_end` fires after the optimizer step + scheduler step in HF Trainer — the correct hook for post-update normalization.

### 5.2 CompressSaveCallback

Writes a sibling **merged** checkpoint dir on every Trainer save AND at end-of-train. The merged dir is drop-in for vLLM / eval; the regular factored `checkpoint-<step>/` dir is what `resume_from_checkpoint` consumes.

```python
class CompressSaveCallback(TrainerCallback):
    def __init__(self, finetuning_args):
        self.method = finetuning_args.finetuning_type
        self.calibrated = finetuning_args.calib_mode != "none"

    def on_save(self, args, state, control, model=None, **kwargs):
        if not state.is_world_process_zero:
            return
        out = pathlib.Path(args.output_dir) / f"checkpoint-{state.global_step}-merged"
        out.mkdir(parents=True, exist_ok=True)
        self._dump(model, str(out))

    def on_train_end(self, args, state, control, model=None, **kwargs):
        if not state.is_world_process_zero:
            return
        out = pathlib.Path(args.output_dir) / "final-merged"
        out.mkdir(parents=True, exist_ok=True)
        self._dump(model, str(out))

    def _dump(self, model, ckpt_dir):
        if self.calibrated:
            from compress.integration import save_calibrated_btt_hf_pretrained
            save_calibrated_btt_hf_pretrained(model, ckpt_dir)
        else:
            _materialize_and_save(model, ckpt_dir)   # local helper
```

`_materialize_and_save` is a private helper in `callbacks.py`. It must **not** mutate the training model. Concretely:

1. Build a fresh `state_dict` by walking `model.named_modules()` and, for each `BTTLinear` / `SVDCompressedLinear`, materializing a dense `weight` tensor (detached clone) under that module's full parameter name (e.g. `model.layers.0.mlp.down_proj.weight`). Non-compressed parameters pass through with `.detach().clone()`.
2. Construct a peer dense model: `peer = AutoModelForCausalLM.from_config(model.config)` (no checkpoint download — config-only init on `meta` device, then `to_empty` on CPU), then call `peer.load_state_dict(merged_state_dict, strict=True)`.
3. Call `peer.save_pretrained(ckpt_dir)` and discard `peer`.

The live training model and its `BTTLinear`/`SVDCompressedLinear` modules are never touched. This is identical in spirit to `run_rl.py::export_weights_for_vllm` but writes to disk instead of vLLM.

**Under DeepSpeed ZeRO-2**, optimizer state is sharded but parameter tensors are still fully replicated per rank, so step 1 reads complete `nn.Parameter` data without any gather. (ZeRO-3, which shards parameters, is disallowed by the validator in Section 3.1, so we don't need a gather path here.) Merged saves run only on `state.is_world_process_zero` to avoid N-way duplicate writes.

(Materialization helpers are mirrored from `run_rl.py::materialize_btt_weight` and `materialize_svd_weight`; we copy the small wrapper functions into `callbacks.py` rather than reaching back into `run_rl.py`.)

### 5.3 Registration in workflow.py

In `LlamaFactory/src/llamafactory/train/sft/workflow.py`:

```python
if finetuning_args.finetuning_type in {"blocktt", "svd"}:
    callbacks = list(callbacks) + [
        CompressNormalizeCallback(finetuning_args),
        CompressSaveCallback(finetuning_args),
    ]
```

## 6. Example YAML configs

Under `LlamaFactory/examples/train_blocktt/`. All three inherit defaults from `examples/train_full/qwen3_base_full_sft.yaml`; only the `### method` block, `output_dir`, and run-name fields differ.

### 6.1 `qwen3_base_blocktt_sft.yaml` (plain BTT)

```yaml
### model
model_name_or_path: Qwen/Qwen3-1.7B-Base
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: blocktt
deepspeed: examples/deepspeed/ds_z2_config.json
flash_attn: fa2
enable_liger_kernel: true

# Compress / BlockTT knobs
decomp_mode: input_one_block
blocktt_rank: full
convert_mode: svd
trainable_type: all
train_position: small           # blocktt: small | large | both
s_merged_to: frozen
train_bias: true
blocktt_normalize_after_update: false
blocktt_factorize_by_head: true
calib_mode: none

### dataset
dataset: openthought3_qwen3_4b
template: qwen3
enable_thinking: false
cutoff_len: 20480
preprocessing_num_workers: 64
dataloader_num_workers: 64

### output
output_dir: ../model/Qwen3-1.7B-Base-BlockTT-OpenThought3-4B
logging_steps: 5
save_steps: 200
plot_loss: true
overwrite_output_dir: true
save_only_model: true
report_to: wandb

### train
per_device_train_batch_size: 8
gradient_accumulation_steps: 1
gradient_checkpointing: true
learning_rate: 1.0e-4
num_train_epochs: 2.0
lr_scheduler_type: cosine
warmup_ratio: 0.05
bf16: true
ddp_timeout: 180000000
resume_from_checkpoint: null

### eval
val_size: 0.05
per_device_eval_batch_size: 4
eval_strategy: steps
eval_steps: 100

### swanlab / wandb
use_swanlab: false
swanlab_project: llamafactory
swanlab_run_name: Qwen3-1.7B-Base-BlockTT-OpenThought3-4B
run_name: Qwen3-1.7B-Base-BlockTT-OpenThought3-4B
```

### 6.2 `qwen3_base_blocktt_calibrated_sft.yaml` (calibrated BTT)

Same as 6.1, except the compress block becomes:

```yaml
calib_mode: v2                  # v2 | v2_bp | v2_combined | twosteps
calib_source: c4                # c4 | traces | training_data
calib_samples: 256
calib_seq_len: 2048
calib_batch_size: 1
calib_traces_path: null
```

### 6.3 `qwen3_base_svd_sft.yaml` (plain SVD)

Same as 6.1, except the compress block becomes:

```yaml
finetuning_type: svd
trainable_type: all
train_position: output          # svd: output | input | both
s_merged_to: frozen
calib_mode: none                # set to svd_v2 / svd_v2_combined for calibrated SVD
```

(No `decomp_mode`, `blocktt_rank`, `convert_mode`, `train_bias`, `blocktt_normalize_after_update`, `blocktt_factorize_by_head` — they have no meaning for SVD.)

### 6.4 `README.md`

Short knob reference table cross-linked to the root `README.md` and to `compress.integration`. The YAML files remain the source of truth for examples; the README explains the meaning of each flag.

## 7. Edge cases

- **`is_trainable=False` (eval/inference loading):** `init_compress_model` short-circuits and returns the model unchanged. Inference points `model_name_or_path` at a merged checkpoint and uses the normal `full` path — `finetuning_type: blocktt|svd` is **only** for training.
- **Resume from checkpoint:** `resume_from_checkpoint` must reference `checkpoint-<step>/` (factored state_dict), NOT `checkpoint-<step>-merged/`. Documented in `examples/train_blocktt/README.md`.
- **`gradient_checkpointing: true`:** `BTTLinear` and `SVDCompressedLinear` implement standard `nn.Module.forward`, so gradient checkpointing is expected to work. Smoke-test in implementation.
- **Liger kernel:** patches HF attention/MLP at the module level and does not replace the inner `nn.Linear`. Since `init_compress_model` runs in `init_adapter` (after `patcher.patch_model` / Liger), the conversion sees the post-Liger graph. Smoke-test in implementation.
- **`save_only_model: true`** (SFT default): no optimizer state on save — fully compatible. `save_only_model: false` saves optimizer state for the (factored) BTT/SVD parameters; also fine.
- **Final-step save:** `save_steps: 200` may not land on the final step. `CompressSaveCallback.on_train_end` writes `final-merged/` unconditionally.

## 8. Testing plan (implementation phase)

1. Smoke: `Qwen3-0.6B-Base` + tiny dataset slice + plain BTT (`calib_mode: none`) + 5 steps, single GPU, no DeepSpeed. Confirm forward, backward, save, reload-from-merged.
2. Same with plain SVD.
3. Same with calibrated BTT (`calib_mode: v2`, `calib_source: c4`, `calib_samples: 16`).
4. Multi-GPU ZeRO-2, 1 epoch on the OpenThought3 slice — confirm grads sync, end-of-train `final-merged/` is reloadable in vLLM.
5. Negative test: `finetuning_type: blocktt` + `deepspeed: ds_z3_config.json` errors out at YAML-parse time.

## 9. Open questions / future work

- **Galore / Apollo / BAdam interplay:** disallowed in v1. If we later want Muon-style routing (which `run_rl.py` does for BTT/SVD params), we'd extend LF's optimizer factory similarly. Not in scope here.
- **WebUI surface:** new `finetuning_type` values won't appear in the LF WebUI dropdown without component edits. CLI/YAML only for v1.
- **`v1` API subdirectory** (`LlamaFactory/src/llamafactory/v1/`): the v1 surface has its own arg model; this design targets the legacy hparams system used by `examples/train_full/qwen3_base_full_sft.yaml`. v1 integration is a separate task.
