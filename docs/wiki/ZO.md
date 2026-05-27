# Zeroth-order trainers in OPD: Weight Perturbation (ES) and Node Perturbation (NP)

This page documents the two zeroth-order trainers vendored into `verl/` as sibling
modules to `verl.trainer.main_ppo`. Both estimate `∇_W L` without a backward pass,
using only forward rollouts through vLLM engines. They differ in **what** gets
perturbed and therefore in **variance scaling**, **memory footprint**, and the
**update shape**.

| Aspect | ES (weight perturbation) | NP (node perturbation) |
|---|---|---|
| Perturbed object | The full parameter tensor `W` | The output `y = Wx` of a chosen linear layer |
| Variance per query | `O(#params)` | `O(d_out)` per layer (much smaller) |
| Forwards per "gradient" | `2·F` antithetic (or `1·F + baseline`) | `2·F` antithetic |
| Update shape | Scalar-weighted sum of full-tensor Gaussian noise | Sum of rank-1 outer products `δy_t ⊗ x_t` |
| Population dimension | Yes (`population_size` independent seeds) | No (variance comes from tokens) |
| Multi-engine usage | Population sharding (one seed per engine) | Data-parallel prompt sharding |
| vLLM `enforce_eager` | Optional (off by default) | **Mandatory** (forward hooks die under `torch.compile`) |
| Loss signals | Rule reward only (v1) | Rule reward (v1); teacher reverse-KL skeleton |

Both share the same "never store the noise tensor" invariant: every `ε` is
regenerated on demand from a deterministic seed (`torch.Generator.manual_seed`),
so the only state crossing the Ray boundary is one or more **integers**.

---

## 1. Where the code lives

```
verl/verl/trainer/
  main_es.py                                   Hydra entry for ES
  main_np.py                                   Hydra entry for NP
  config/es_trainer.yaml                       ES defaults
  config/np_trainer.yaml                       NP defaults
  es/{__init__.py, ray_trainer.py, task_utils.py}
  np/{__init__.py, ray_trainer.py, loss_fns.py}

verl/verl/workers/rollout/vllm_rollout/
  es_worker_extension.py                       ES forward-hook-free worker methods
  np_worker_extension.py                       NP forward-hook installer + apply_node_update

opd_es.sh                                      Top-level launcher for ES
opd_np.sh                                      Top-level launcher for NP
```

Both trainers reuse:

- `verl.trainer.es.task_utils.get_task_components` for built-in tasks
  (`countdown`, `gsm8k`, `math`, `math500`, `olympiadbench`, `uspto50k`,
  `common_gen`, `mbpp`, `rocstories`, `opd_math`). NP imports it directly from
  the ES module — there is no duplicate copy.
- `verl.utils.reward_score.ttrl_math.reward_func` for OPD-math rewards.
- vLLM NCCL primitives (`PyNcclCommunicator`, `StatelessProcessGroup`) — same
  pattern in both worker extensions.

No edits to `main_ppo.py`, `ppo/core_algos.py`, `dp_actor.py`, `fsdp_workers.py`,
or `vllm_rollout_spmd.py`.

---

## 2. ES (weight perturbation) — design

ES implements the OpenAI-ES update:

```
θ_{t+1} = θ_t + (α / (n·σ)) · Σ_{i=1..n} F_i · ε_i,    ε_i ~ N(0, I)
```

where `F_i` is the per-seed normalized reward and `ε_i` is a full-tensor
Gaussian noise generated from a single 64-bit seed.

### 2.1 Per-step loop

`verl/verl/trainer/es/ray_trainer.py` runs:

1. **Draw `population_size` seeds** for the iteration (deterministic from
   `global_seed + iteration`).
2. **For each batch of `num_engines` seeds in parallel:**
   1. `perturb_self_weights(seed, sigma)` — each engine adds `σ · ε_i` to every
      parameter (`ε_i` regenerated from the seed inside the worker).
   2. `engine.generate(prompts, sampling_params)` — collect rollouts.
   3. `restore_self_weights(seed, sigma)` — subtract the noise back out.
   4. Compute per-rollout rewards via `reward_fn`.
3. **Normalize rewards** across the population: `F_i = (r_i − μ) / σ_r`.
4. **`update_weights_from_seeds(seeds, coeffs, alpha, population_size)`** on
   engine 0 — accumulates `Σ F_i · ε_i` *one parameter tensor at a time*,
   regenerating each `ε_i` per parameter to keep memory bounded. Applies the
   final `(α / n) · Σ F_i · ε_i` to `W`.
5. **`broadcast_all_weights(src_rank=0)`** — sync engine 0's weights to all
   peers via NCCL (`PyNcclCommunicator.broadcast`).

### 2.2 The "never store ε" trick

Every `ε_i` is regenerated wherever it's needed via:

```python
gen = torch.Generator(device=p.device)
gen.manual_seed(int(seed))
noise = torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=gen)
```

so the RPC payload to perturb/restore/update is just `(seed: int, scale: float)`
plus the integer list of seeds for the update. No tensor of size `#params`
ever crosses Ray.

### 2.3 ES knobs

From `verl/verl/trainer/config/es_trainer.yaml` (every value can be overridden
on the command line as `es.<key>=...`):

| Knob | Default | Meaning |
|---|---|---|
| `es.sigma` | `0.001` | Noise scale `σ` |
| `es.alpha` | `0.0005` | Learning rate `α` |
| `es.population_size` | `30` | Number of seeds per iteration |
| `es.num_engines` | `4` | Parallel vLLM engines |
| `es.num_iterations` | `800` | Total iterations |
| `es.precision` | `bfloat16` | Model dtype |
| `es.temperature` | `0.0` | Rollout sampling temperature (greedy by default) |
| `es.max_tokens` | `1024` | Generation budget per rollout |
| `es.eval_interval` | `25` | Eval every N iterations |
| `es.gpu_memory_utilization` | `0.9` | vLLM KV-cache budget |
| `es.global_seed` | `42` | Deterministic seed for the iteration RNG |
| `es.worker_extension_cls` | `...es_worker_extension.WorkerExtension` | Worker class registered with vLLM |

### 2.4 ES usage

```bash
# Smallest end-to-end check.
ACTOR_MODEL_PATH=model/Qwen3-1.7B \
TRAIN_DATASET=datasets/dapo-math-17k.parquet \
EVAL_DATASET=datasets/test_data/AIME24/test.parquet \
SIGMA=0.001 ALPHA=0.0005 POPULATION_SIZE=30 NUM_ITERATIONS=200 \
N_GPUS_PER_NODE=8 NUM_ENGINES=8 \
bash opd_es.sh
```

The launcher tee's stdout/stderr into `logs/opd_es_<timestamp>.log` when not
running under SLURM, writes checkpoints to `${SAVE_DIR}` (auto-derived from
`PROJECT_PATH`, model name, and hyperparameters), and logs to wandb/console
(`ES_LOGGER`).

Direct invocation (bypassing the launcher) is also fine — useful for ad-hoc
overrides:

```bash
python3 -m verl.trainer.main_es \
    es.sigma=0.001 es.alpha=0.0005 es.population_size=30 \
    es.num_engines=8 es.num_iterations=200 \
    model.path=model/Qwen3-1.7B \
    data.task_type=opd_math \
    data.train_files=datasets/dapo-math-17k.parquet \
    data.val_files=datasets/test_data/AIME24/test.parquet \
    trainer.n_gpus_per_node=8 trainer.nnodes=1 \
    trainer.default_local_dir=/tmp/es_smoke
```

### 2.5 ES costs

- **Compute per step:** `population_size · F`. Population is the dominant cost.
- **Memory:** ES regenerates one `ε` per parameter tensor at a time and frees
  it immediately — peak overhead during update is `~2 · |W_largest_tensor|`.
- **RPC payload:** integer-only across the entire step.

---

## 3. NP (node perturbation) — design

NP perturbs the **output** of one (or many) linear layer(s):

```
y' = Wx + sign · σ · δy,    δy ~ N(0, I_{d_out})
```

The clean upstream input `x` is recovered by a second (clean) forward through
the same layer with a capture hook — because perturbation is added *after* `Wx`,
`x` is identical in the clean and perturbed passes.

The update is **ANP-normalized** (Dalm 2024) with antithetic sampling:

```
δW = Σ_t (L_+,t − L_-,t)/2 · (δy_t / ‖δy_t‖²) ⊗ x_t       (token tape)
δW = (L_+ − L_-)/2 · (δy / ‖δy‖²) ⊗ x̄                     (sequence tape)
W ← W + lr · δW
```

`L_+`/`L_-` come from antithetic `±δy` rollouts; `δy` is regenerated from the
same seed in both the perturb hook and `apply_node_update`, so antithetic
cancellation is exact up to floating-point noise.

### 3.1 Per-step loop

`verl/verl/trainer/np/ray_trainer.py` (`RayNPTrainer.fit()`):

```
for step in range(num_iterations):
    prompts, datas = next_batch()
    active_layers = round_robin_or_all(perturb_rules, step)

    for layer_name in active_layers:
        # 1) Clean forward — capture x at this layer.
        clean_outputs, x_clean = run_clean_pass_capture(layer_name, prompts)

        # 2) Perturbed +δy.
        plus_outputs  = run_perturbed_pass(layer_name, +1, sigma, tape_kind)
        L_plus  = loss_fn(plus_outputs, datas)

        # 3) Perturbed -δy (antithetic).
        minus_outputs = run_perturbed_pass(layer_name, -1, sigma, tape_kind)
        L_minus = loss_fn(minus_outputs, datas)

        # 4) Apply NP update on engine 0.
        apply_node_update(layer_name, step, sigma, tape_kind,
                          L_plus - L_minus, x_clean, lr)

        # 5) Broadcast updated layer weights to peer engines.
        broadcast_layer_weights(layer_name, src_rank=0)
```

### 3.2 Forward hooks (the engineering core)

`np_worker_extension.WorkerExtension.install_perturb_hooks(rules)` is called
**once** after `load_model`, before any rollout. It:

1. Walks `self.model_runner.model.named_modules()`.
2. Matches each module name against every regex (Python `re.fullmatch`).
3. Registers a single forward hook on each match.
4. Stores the matched-module dict on `self.np_state["modules"]`.

The hook body reads `self.np_state` — a worker-local dict — to decide whether
to no-op, capture `x`, or inject `δy`. This means installing hooks **once** is
enough; per-RPC mode switches just write to `np_state`.

vLLM's `ColumnParallelLinear` / `QKVParallelLinear` return either a bare
`Tensor` (when `return_bias=False`) or a `(Tensor, Optional[Param])` tuple
(default). The hook unpacks-and-repacks so both contracts are preserved.

### 3.3 The "never store δy" trick

```python
seed = _np_noise_seed(global_seed, step, layer_name, sign=1, position=t)
gen  = torch.Generator(device).manual_seed(int(seed))
δy   = torch.randn(d_out, ..., generator=gen)
```

`_np_noise_seed` is an FNV-1a 64-bit hash of `(global_seed, step, layer_name,
position)`. Both the perturb hook and `apply_node_update` derive the same `δy`
from the same `(step, layer, position)` tuple — so the trainer side regenerates
δy on demand when forming `δy ⊗ x`. RPC payloads are `(layer_name: str, step: int,
global_seed: int, sigma: float, ...)` — no `δy` ever crosses Ray, no `δy`
persists past the hook call.

### 3.4 Tape kinds

| `tape_kind` | δy shape per hook call | Update shape |
|---|---|---|
| `sequence` | `[d_out]` broadcast across all tokens | `(L_+ − L_-)/2 · (δy / ‖δy‖²) ⊗ x̄`, one update per layer |
| `token` | `[num_tokens, d_out]` block, one row per generated position | `Σ_t (L_+,t − L_-,t)/2 · (δy_t / ‖δy_t‖²) ⊗ x_t` |

Per-token tape memory: `T · d_out · #active_layers` bytes per engine (bf16 → 2×
that). For `T=4096, d_out=4096`, one active layer: ~64 MB.

### 3.5 `perturb_rules` regex semantics

A single regex space covers the three user-stated granularities:

```yaml
np:
  perturb_rules:
    - "model\\.layers\\.\\d+\\.mlp\\.up_proj"          # layer-type
    - "model\\.layers\\.\\d+\\.self_attn\\.q_proj"     # all q_projs
    - "model\\.layers\\.0\\.mlp\\.down_proj"           # specific layer
```

Modules are selected by `re.fullmatch`, so be precise about anchors. The
launcher converts a newline-separated `PERTURB_RULES` env var into the JSON
list Hydra expects.

### 3.6 Loss signal plug-in

`verl/verl/trainer/np/loss_fns.py` provides two factories:

- **`make_grpo_loss(reward_fn, grpo_n=4)`** — runs `K=grpo_n` rollouts per
  prompt, computes the rule reward via `reward_fn`, standardizes within each
  prompt's `K` rollouts, and returns the **negative** mean as `L` (so gradient
  ascent on reward becomes gradient descent on `L`). v1 is the GRPO path; this
  is what `opd_np.sh` exercises.
- **`make_opd_kl_loss(teacher_engines, tokenizer, log_prob_top_k, top_k_strategy,
  reward_weight_mode, teacher_temperature)`** — per-token reverse-KL against a
  teacher engine, restricted to OPD's K-token set. The factory implements the
  KL math but `RayNPTrainer.fit()` currently raises `NotImplementedError` for
  `loss_type=opd` (deferred: needs a teacher engine pool wired into
  `init_workers`).

### 3.7 NP knobs

From `verl/verl/trainer/config/np_trainer.yaml`:

| Knob | Default | Meaning |
|---|---|---|
| `np.perturb_rules` | `["model\\.layers\\.\\d+\\.mlp\\.up_proj"]` | Regex list (fullmatch). |
| `np.layer_schedule` | `one_per_step` | `one_per_step` (round-robin) or `all_per_step`. |
| `np.tape_kind` | `token` | `token` (per-position δy) or `sequence` (broadcast). |
| `np.sigma` | `0.01` | δy noise scale. |
| `np.lr` | `1.0e-4` | Update learning rate. |
| `np.antithetic` | `true` | If false, uses (perturbed − clean) instead of (+δy − −δy). |
| `np.update_clip` | `null` | Optional ‖δW‖ clip. |
| `np.loss_type` | `grpo` | `grpo` (rule reward) or `opd` (not yet wired). |
| `np.grpo_n` | `4` | Rollouts per prompt for GRPO standardization. |
| `np.num_engines` | `1` | Data-parallel engines (not a population dimension). |
| `np.num_iterations` | `200` | Total training steps. |
| `np.precision` | `bfloat16` | Model dtype. |
| `np.temperature` | `1.0` | Must be > 0 so the K GRPO rollouts vary. |
| `np.max_tokens` | `1024` | Generation budget. |
| `np.batch_size` | `16` | Prompts per step. |
| `np.eval_interval` | `25` | Eval frequency. |
| `np.gpu_memory_utilization` | `0.7` | vLLM KV-cache budget. |
| `np.worker_extension_cls` | `...np_worker_extension.WorkerExtension` | NP worker class. |

### 3.8 NP usage

```bash
ACTOR_MODEL_PATH=model/Qwen3-1.7B \
PERTURB_RULES=$'model\\.layers\\.\\d+\\.mlp\\.up_proj\nmodel\\.layers\\.\\d+\\.self_attn\\.q_proj' \
LAYER_SCHEDULE=one_per_step TAPE_KIND=token \
SIGMA=0.01 LR=1e-4 LOSS_TYPE=grpo GRPO_N=4 \
NUM_ITERATIONS=200 N_GPUS_PER_NODE=8 \
bash opd_np.sh
```

`PERTURB_RULES` is newline-separated; the launcher converts it to a JSON list
for Hydra. Same `PROJECT_PATH` / `SAVE_DIR` / `*_LOGGER` conventions as
`opd_es.sh`.

Direct invocation:

```bash
python3 -m verl.trainer.main_np \
    'np.perturb_rules=["model\\.layers\\.\\d+\\.mlp\\.up_proj"]' \
    np.layer_schedule=one_per_step np.tape_kind=token \
    np.sigma=0.01 np.lr=1e-4 np.loss_type=grpo np.grpo_n=4 \
    model.path=model/Qwen3-1.7B \
    data.task_type=opd_math \
    data.train_files=datasets/dapo-math-17k.parquet \
    data.val_files=datasets/test_data/AIME24/test.parquet \
    trainer.n_gpus_per_node=1 trainer.nnodes=1 \
    trainer.default_local_dir=/tmp/np_smoke np.num_iterations=1
```

### 3.9 NP costs

- **Compute per step:** `2·F` antithetic (or `3·F` if non-antithetic with a
  baseline clean pass), times `#active_layers`. `enforce_eager=True` adds
  another ~1.5–3× wall-clock penalty over `enforce_eager=False`.
- **Memory:** `T · d_out · #active_layers · dtype_bytes` per engine for the
  token tape, plus the captured `x` of shape `[T, d_in]`.
- **RPC payload:** integers + the captured `x` tensor (one per active layer per
  step; goes from worker → driver for the update).

### 3.10 NP-specific verification

Three quick checks before believing a long NP run:

1. **Hook-fires smoke test.** At `np.sigma=0`, NP is a no-op; clean and
   perturbed rollouts must be token-identical. Log
   `get_hook_call_count(layer_name)` at step 0 — must be `>0`; if it's exactly
   0 the hook was erased by `@support_torch_compile` (check `enforce_eager`).
2. **Antithetic cancellation.** At any `σ > 0`, if you symmetrize the loss
   (`(L_+ + L_-) / 2`) it should equal the clean loss up to floating-point
   noise — same `δy` magnitude, opposite sign.
3. **Gradient cosine similarity (offline).** Compute the true `∂L/∂W` for one
   layer/batch via an eager HF backward; compare NP's `δW` estimate over 100
   antithetic samples. Cosine should be ≥ 0.1 for token-tape NP within a few
   hundred samples; negative or stuck-near-zero means a sign bug.

---

## 4. Cross-cutting topics

### 4.1 vLLM compatibility

| | ES | NP |
|---|---|---|
| `enforce_eager` | Off (default) | **On (mandatory)** |
| `enable_prefix_caching` | Off | Off (identical-prompt forwards must not share KV cache across clean/perturbed) |
| vLLM version | 0.11.0 (pinned by `install_vllm_sglang_mcore.sh`) | 0.11.0 (same) |

ES does not need hooks, so it can ride the default CUDA-graph compile path.
NP needs forward hooks on `nn.Linear` submodules; vLLM 0.11.0 decorates
`Qwen3Model` / `LlamaModel` with `@support_torch_compile`, which inlines
submodules under Dynamo and erases hooks before CUDA-graph capture. The only
clean fix in 0.11.0 is `enforce_eager=True` (`NPNcclLLM.__init__` forces it).
vLLM RFC #36998 may eventually expose a hook-safe compile mode; revisit then.

### 4.2 Multi-engine semantics

- **ES `num_engines`** = parallel seed evaluators. With `population_size=30` and
  `num_engines=8`, ES does 4 batches of 8 seeds each (last batch may be
  partial). Engines write to engine 0 via NCCL broadcast at the end of each
  iteration.
- **NP `num_engines`** = data-parallel prompt shards. Each engine sees the same
  hooks installed; the trainer drives the update from engine 0 and broadcasts
  the updated layer's weights afterwards. No population dimension.

### 4.3 What ES and NP *don't* share

- **Checkpoint hot-swap:** both save standard HF state dicts, but their
  trajectories are not interchangeable mid-run (ES moves all of `θ` in concert;
  NP moves one layer at a time). Don't switch trainer types on a saved
  checkpoint mid-experiment.
- **Population sampling:** ES has it (`population_size`), NP doesn't (variance
  is over tokens).
- **Compile mode:** see §4.1.
- **Attention-internals support:** NP hooks on `q_proj` / `k_proj` / `v_proj` /
  `o_proj` fire fine, but the paged-attention fused op bypasses Python hooks —
  so attention itself is not perturbable in v1. ES, perturbing weights, is
  unaffected.

### 4.4 Task plumbing (shared)

Both trainers route built-in tasks through
`verl.trainer.es.task_utils.get_task_components`. Supported `data.task_type`:

```
countdown   gsm8k         math       math500       olympiadbench
uspto50k    common_gen    mbpp       rocstories    opd_math
```

`opd_math` is the OPD-paper math route — it pairs `create_math_prompt_processor`
with `create_opd_math_reward_fn`, which wraps `verl.utils.reward_score.ttrl_math.reward_func`
(the same backend `on_policy_distillation.sh` uses for RL).

Custom tasks: set `data.task_type=custom` and provide
`data.reward_fn_path`/`data.reward_fn_name` plus
`data.prompt_processor_path`/`data.prompt_processor_name`. Both are loaded
through `verl.utils.import_utils.load_extern_object`.

### 4.5 Hardware notes

The repo defaults assume 8× 80GB GPUs. For a small smoke test:

- ES with `Qwen3-1.7B`, `population_size=4`, `num_engines=2` fits on 2× 24GB.
- NP with `Qwen3-1.7B`, `np.num_engines=1`, one active layer fits on 1× 24GB
  with `np.gpu_memory_utilization=0.6` (because `enforce_eager` raises the
  per-step working set).

---

## 5. Pointers

- ES paper line: OpenAI ES (Salimans et al. 2017) → MeZO (Malladi et al. 2023)
  for the weight-perturbation language-model variant.
- NP / ANP: Dalm (2024) on Activity (Node) Perturbation — the source of the
  `δy / ‖δy‖²` normalization used in `apply_node_update`.
- Existing OPD entry points: `verl/verl/trainer/main_ppo.py` (the PPO/OPD
  trainer), `on_policy_distillation.sh`, `grpo.sh` — none of which are touched
  by the ES/NP code paths.
