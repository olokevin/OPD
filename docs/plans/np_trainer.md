# Node Perturbation (NP) trainer for OPD's verl fork

## Config interfaces (.yaml))

```
sigma: 0.01                      # Perturbation magnitude
n_sample: 8                      # Number of random samples per rollout
n_rollout: 8			 # Number of rollouts for grad estimation
sample_method: bernoulli         # 'gaussian', 'bernoulli', 'uniform'

en_layerwise_perturbation: true  # false = perturb all layers at once

perturb_method: forward      # 'forward' (only L+)) or 'antithetic' (L+, L-))
perturb_granularity: token # rollout: same perturbation for every generated perturbed token; token: independent perturbations
grad_estimate_sample: grpo # for n_sample, each independent sample u_n gives a token loss (opd teacher-student) L_n  average: 1/n_sample * sum_n((L_n-L)/2/sigma) * u_n)  grpo: 1/n_sample * sum((L_n-mean)/std * u_n)
grad_estimate_sequence: grpo # for n_rollout, each rollout t has a reward Rt average: 1/n_rollout* sum_n(R_t/sigma) * u_t)  grpo: 1/n_rollout* sum((R_t-mean)/std * u_t)  in rollout gradnularity u_t is the perturbation for rollout; in token granulairy just apply the scale for the grad estimation component of the rollout t

perturb_rules:
  # use regexp to find the layers to add perturbation & grad estimation. should adjust according to actual name pattern
  ### layer_norm:
    # name_pattern: 'model.layers.4.input_layernorm'

  ### single_layer:
    # name_pattern: 'model.layers.7.mlp.down_proj'
    # name_pattern: 'model.layers.4.mlp.up_proj'
    # name_pattern: 'model.layers.4.self_attn.o_proj'
    # name_pattern: 'model.layers.4.self_attn.v_proj'
    # name_pattern: '^model.layers.4.self_mlp$'

  ### single_type:
    # name_pattern: 'model.layers.\d.self_attn.q_proj'
    # name_pattern: 'model.layers.\d.mlp.up_proj'
  
  ### single decoder
    # name_pattern: '^model.layers.4$'
  
  ### all decoder layers:
    name_pattern: '^model.layers.\d$'
  
  ### single decoder block
    # name_pattern: 'model\.layers\.4\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(up_proj|down_proj|gate_proj))'

  ### all_linear_layers:
    # name_pattern: 'model\.layers\.\d\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(up_proj|down_proj|gate_proj))'
```

## Context

The repo already has a Weight-Perturbation trainer (`verl/trainer/main_es.py` + `verl/trainer/es/` + `workers/rollout/vllm_rollout/es_worker_extension.py`) that ports RandOpt's OpenAI-ES on top of vLLM. It is committed as `301bf5c feat(es): port RandOpt's evolution-strategy trainer to OPD verl` on the `feat/blocktt-svd-llamafactory` branch (it landed there during the port flow; intentionally left in place — do not touch BlockTT work). The same paper line on zeroth-order gradient estimation has a sibling family, **node / activity perturbation**: instead of $\theta \leftarrow \theta + \alpha \cdot \sum_i F_i \cdot \varepsilon_i$ over the entire weight tensor, perturb the **output of a linear layer** $y = Wx$ with $\delta y \sim \mathcal{N}(0, \sigma^2 I)$, then recover $\delta W \approx (L_{\text{perturbed}} - L_{\text{clean}}) \cdot \delta y \cdot x^\top / \sigma^2$. The student's input $x$ to the perturbed layer is **not cached** during the perturbed forward; it's recovered by a second (clean) forward — because perturbation is added *after* $Wx$, the upstream $x$ is identical across the two passes.

Why it's worth doing here:

- Per-layer variance scales $O(d_{\text{out}})$ (NP) vs $O(d_{\text{in}} \cdot d_{\text{out}})$ (MeZO weight-perturb) — orders of magnitude tighter per query.
- Compute per step is $2F$ ($3F$ with antithetic) vs $3F$ for SGD — same as MeZO, no backward pass.
- We get a richer update shape (rank-1 outer product $\delta y \otimes x$ per token, summed over tokens) rather than a single scalar-times-Gaussian like MeZO.
- It plugs into the two reward signals OPD already has plumbing for: a GRPO-style normalized rule reward, and OPD's teacher reverse-KL per token.

The headline risk is **vLLM compatibility**. NP requires forward hooks on internal linear layers; vLLM 0.11.0 (pinned in `verl/scripts/install_vllm_sglang_mcore.sh:12`) decorates `Qwen3Model` / `LlamaModel` with `@support_torch_compile`, which causes Dynamo to inline submodules and erase hooks before CUDA graph capture. Fix is `enforce_eager=True` — viable, ~1.5–3× throughput cost. Two production projects do exactly this on vLLM (UK AISI's `vllm-lens`, `nnsight` 0.5.x VLLMBatcher).

## Invariant: never store $\delta y$ — store the seed

Same trick the existing ES uses for weights: every $\delta y$ tensor is regenerated on demand from a deterministic seed via `torch.Generator(device).manual_seed(seed)` + `torch.randn(...)`. No $\delta y$ ever crosses a Ray boundary, no $\delta y$ is persisted, no $\delta y$ lives longer than the forward hook that consumes it. Concretely:

- The trainer side picks a **rollout_seed** per (step, layer, antithetic sign) tuple and sends only that integer in the RPC. Identity:
  `noise_seed(step, layer_name, sign, position=None) = hash((global_seed, step, layer_name, sign, position))`
  → fits in 64 bits; passed to `torch.Generator.manual_seed`.
- **Sequence tape**: one `seed = noise_seed(step, layer_name, sign)`. The hook generates one $\delta y \in \mathbb{R}^{d_{\text{out}}}$ per forward and broadcasts across positions.
- **Token tape**: per-position seed `seed_t = noise_seed(step, layer_name, sign, t)`, regenerated inside the hook from the current decode position. A small per-rollout `position_counter` on `self.np_state` advances each hook call.
- **Antithetic $L_-$**: same `step`, same `layer_name`, same `position`, opposite `sign`. The hook's RNG draws the identical `randn` and multiplies by $-\sigma$ instead of $+\sigma$ — guaranteeing exact cancellation up to floating-point noise.
- **`apply_node_update`** regenerates $\delta y$ from the same seed sequence when forming $\delta y \otimes x$. Memory cost of $\delta y$ is one $[d_{\text{out}}]$ (sequence) or $[d_{\text{out}}]$ per-token (transient inside the hook + accumulated into $\delta W$ in the update). The tape itself is **zero bytes on disk and zero bytes in RPC payloads**.

This mirrors `es_worker_extension.py:47-60` (perturb) and `:78-119` (update_from_seeds) verbatim in spirit. The only difference is the seed namespace now includes `(layer_name, position)`, and the RNG is consumed inside a forward hook rather than at the top of a perturb method.

## Design at a glance (sibling trainer, mirrors ES structure)

```
verl/trainer/main_np.py                      Hydra entry (mirrors main_es.py)
verl/trainer/np/__init__.py
verl/trainer/np/ray_trainer.py               RayNPTrainer + NPNcclLLM
verl/trainer/np/task_utils.py                reused from ES (import, no copy)
verl/trainer/config/np_trainer.yaml          new Hydra config
verl/workers/rollout/vllm_rollout/np_worker_extension.py
opd_np.sh                                    top-level launcher
```

Zero modifications to: `main_ppo.py`, `ppo/core_algos.py`, `dp_actor.py`, `fsdp_workers.py`, `vllm_rollout_spmd.py`, the just-merged `main_es.py` / `es/` / `es_worker_extension.py`.

## Worker extension — the engineering core

`np_worker_extension.py` is the only file with real algorithmic content. It owns the perturbation tape, the forward hooks, and the activation capture. Lifecycle on each vLLM worker:

1. **`install_perturb_hooks(perturb_rules: list[str])`** — called once after `load_model`, before any rollout. Walks `self.model_runner.model.named_modules()`, matches each module name against the regex list, and registers a **forward hook** on each match. Stores the active-module list as `self.np_modules: dict[str, nn.Module]`. The hook reads worker-local state (`self.np_state`) to decide whether to no-op, capture activations, or inject noise — so installing once and switching modes per RPC is cheap.

   Hook body unpacks vLLM's `(output, bias)` tuples (returned by `ColumnParallelLinear` / `QKVParallelLinear` when `return_bias=True`, which is the default — `vllm/model_executor/layers/linear.py:548-568`) and writes back the same shape.
2. **`run_clean_pass_capture(prompts, sampling_params, layer_name)`** — sets `np_state = {mode: "capture", layer: layer_name}`, calls `self.generate(prompts, sampling_params)`, the hook on `layer_name` records $x$ (the layer's *input* — `args[0]` in the forward hook signature) per generated token into `self.np_state.captured_x: dict[token_idx → tensor]`. Returns rollout tokens + captured $x$ tensor stacked as $[B, T, d_{\text{in}}]$. Also returns clean-rollout $L_{\text{clean}}$ via the configured loss function.
3. **`run_perturbed_pass(prompts, sampling_params, layer_name, rollout_seed, sigma, tape_kind, sign=+1)`** — sets `np_state = {mode: "perturb", layer, seed, sigma, tape_kind, sign}`. The hook regenerates $\delta y$ deterministically from `(rollout_seed, layer_name, position)` using `torch.Generator`, multiplies by $\text{sign} \cdot \sigma$, adds to the layer output. `tape_kind ∈ {"sequence", "token"}`: sequence broadcasts one $\delta y \in \mathbb{R}^{d_{\text{out}}}$ across all positions; token uses a per-position seed offset. Returns rollout tokens + $L_{\text{perturbed}}$ (which may be a per-token vector for the OPD reverse-KL path).
4. **`apply_node_update(layer_name, step, sigma, tape_kind, delta_L_signal, x_clean, lr, antithetic_pair=True)`** — implements the ANP-normalized update (Dalm 2024). With antithetic $\pm \delta y$:
   $\delta \hat{W} = N \cdot (L_+ - L_-) \cdot (\delta y / \lVert \delta y \rVert^2) \otimes x / 2$
   $W \leftarrow W + \text{lr} \cdot \delta \hat{W}$
   When `delta_L_signal` is a vector $L_t$ (per-token, from the OPD reverse-KL path with per-token noise tape), the update is $\delta \hat{W} = \sum_t (L_{+,t} - L_{-,t}) \cdot (\delta y_t / \lVert \delta y_t \rVert^2) \otimes x_t / 2$ — the richer per-token credit assignment branch. $\delta y$ is **regenerated on the fly** from `noise_seed(step, layer_name, +1, t)` (see "Invariant: never store $\delta y$" above); the only state crossing the RPC boundary is `step` and `layer_name`.
5. **`broadcast_updated_weights(layer_name, src_rank=0)`** — reuses the existing NCCL pattern from `es_worker_extension.py:132-137` but broadcasts only the updated layer's parameters, not the full model.

The hook closure is intentionally minimal — just `if self.np_state["mode"] == "capture": ...; elif "perturb": ...`. No `torch.compile`-incompatible Python (the model is loaded with `enforce_eager=True` anyway; see vLLM kwargs below).

## RayNPTrainer.fit() — per-step loop

```
for step in range(num_iterations):
    # Pick prompts for this step (mini-batch from train_data)
    prompts, gts = sample_batch(train_data, batch_size)

    # Resolve perturbation targets for this step
    matched = regex_match(perturb_rules, all_module_names)
    if layer_schedule == "one_per_step":
        active_layers = [matched[step % len(matched)]]   # round-robin
    elif layer_schedule == "all_per_step":
        active_layers = matched                          # all-at-once with independent noise

    for layer_name in active_layers:
        # 1) Clean forward — capture x at this layer
        clean_rollout, x_clean, L_clean = ray.get(
            engine.collective_rpc.remote("run_clean_pass_capture",
                                         args=(prompts, sp, layer_name)))

        # 2) Perturbed +δy
        rollout_seed = derive_seed(step, layer_name, sign=+1)
        roll_p, L_plus = ray.get(
            engine.collective_rpc.remote("run_perturbed_pass",
                                         args=(prompts, sp, layer_name,
                                               rollout_seed, sigma, tape_kind, +1)))

        # 3) Perturbed −δy (antithetic)
        roll_m, L_minus = ray.get(
            engine.collective_rpc.remote("run_perturbed_pass",
                                         args=(prompts, sp, layer_name,
                                               rollout_seed, sigma, tape_kind, -1)))

        # 4) Apply NP update to that layer's weights
        ray.get(engine.collective_rpc.remote(
            "apply_node_update",
            args=(layer_name, rollout_seed, sigma, tape_kind,
                  L_plus - L_minus, x_clean, lr, True)))

    # 5) Broadcast updated weights across engines
    ray.get([e.collective_rpc.remote("broadcast_updated_weights",
                                     args=(layer_name, 0))
             for e in engines for layer_name in active_layers])

    if step % eval_interval == 0:
        run_eval(...)
    if step % save_freq == 0:
        save_checkpoint(...)
```

Population-style parallelism (multiple seeds simultaneously) is **not** the v1 design — NP gets variance reduction from $\sum_t \delta y_t \otimes x_t$ over tokens, not from population averaging. Multi-engine parallelism is used for data-parallel rollout (split prompts across engines), not population sampling. This is the structural break from ES; the launcher exposes `num_engines` for prompt sharding, not `population_size`.

## Population aggregation across $N$ perturbation directions (v2, planned not built)

v1 draws one antithetic pair $(+\delta y, -\delta y)$ per (step, layer) and uses $(L_+ - L_-)/2$ as the scalar fitness in $\delta \hat{W} \propto (L_+ - L_-) \cdot (\delta y / \lVert \delta y \rVert^2) \otimes x$. Two complementary v2 extensions sample $N > 1$ independent perturbation directions $\{\delta y_n\}$ per (step, layer) and average their contributions. Both reduce variance along the **perturbation axis** (orthogonal to the token-axis reduction v1 already does); pick one — or none — once token-axis reduction is shown insufficient on real OPD runs.

**Option A — plain mean of antithetic-difference estimators.** Each direction $n$ produces its own $(L_{+,n} - L_{-,n})$. Form $\delta \hat{W} = (1/N) \cdot \sum_n (L_{+,n} - L_{-,n}) \cdot (\delta y_n / \lVert \delta y_n \rVert^2) \otimes x_n / 2$.

This is the natural drop-in: same per-direction kernel as v1, just averaged. Variance falls as $1/N$ for independent $\delta y_n$. Compute cost: $2N$ forwards per step per active layer (or $2N + 1$ if we keep a clean pass for the OPD baseline). Memory: one $[d_{\text{out}}]$ per active direction at a time — fully sequential is $O(d_{\text{out}})$, parallel batched needs $O(N \cdot d_{\text{out}})$.

**Option B — z-scored fitness across $N$ directions (ES-style ranking).** Instead of antithetic differencing per direction, draw $N$ one-sided perturbations $\{+\delta y_n\}$, score each with the loss $L_n$ (scalar, e.g. mean of per-token OPD KL over the rollout), and z-score across the population before weighting: $\mu_L = (1/N) \sum_n L_n$, $\sigma_L = \text{std}_n(L_n) + \epsilon$, then $\delta \hat{W} = (1/N) \cdot \sum_n ((L_n - \mu_L) / \sigma_L) \cdot (\delta y_n / \lVert \delta y_n \rVert^2) \otimes x_n$.

This is the OpenAI-ES / GRPO-style baseline: z-scoring removes the population mean (a free baseline, no critic) and normalizes scale so `lr` is invariant to reward-magnitude drift across steps. Sign: higher $L_n$ ⇒ direction $n$ is *worse* ($L$ is minimization-oriented), so the update steps **against** $\delta y_n$ — same sign as Option A's $-(L_+ - L_-)$ after the antithetic difference. With per-token signal $L_t$, the z-scoring is done per-token across $N$ ($\mu_{L,t}$, $\sigma_{L,t}$ are length-$T$ vectors), preserving token-axis variance reduction.

|                         | v1 (antithetic only)      | A (plain mean of$N$ antithetic)             | B (z-score$N$ one-sided)                        |
| ----------------------- | ------------------------- | --------------------------------------------- | ------------------------------------------------- |
| Forwards per step       | 2 (+1 clean)              | $2N$ (+1 clean)                             | $N$ (+1 clean for OPD baseline)                 |
| Variance reduction axis | token                     | token +$1/N$                                | token +$1/N$                                    |
| Baseline                | antithetic cancellation   | antithetic cancellation                       | population mean (free critic)                     |
| Sign-bug risk           | low (symmetric pair)      | low (per-direction symmetric)                 | medium (z-scoring inverts when$\sigma_L$ small) |
| Best when               | token-axis SNR sufficient | token-axis insufficient, sym. noise tolerable | reward scale drifts across steps                  |

**Knobs (v2):**

```yaml
np:
  population_size: 1                 # N; 1 = v1 antithetic only
  population_mode: "antithetic"      # "antithetic" (Option A) or "zscore" (Option B)
  zscore_axis: "token"               # "token" (per-position) or "scalar" (sequence-mean L first)
```

Not implemented in v1. The fit-loop wrapper is: outer `for n in range(N)` around the existing inner (clean + $\pm \delta y$) block, accumulate per-direction $(L_n, \delta y_n\text{ seed})$ tuples, fold into `apply_node_update` via a new `apply_node_update_population` entry that takes a list of seeds and per-direction signals. The existing v1 path is `population_size=1, population_mode=antithetic`.

## Loss-signal plug-in

Loss is a pluggable callable `loss_fn(rollout_tokens, prompts, sampling_outputs) -> Union[float, Tensor]`:

- **`grpo` mode (v1, no teacher).** Rule-based score per rollout via `verl.utils.reward_score.ttrl_math.reward_func` (already used by `opd_math` task type in `task_utils.py`). For each step, run $K$ rollouts per prompt (configurable `np.grpo_n`, default 4). Compute $(L_i - \mu) / \sigma$ over the $K$ rollouts as the per-rollout signal. $L_+$ and $L_-$ are each a scalar mean over the $K$ normalized rollouts.
- **`opd` mode (teacher reverse-KL, per-token).** Launches a second vLLM engine per node hosting the teacher model (pattern: `REWARD_MODEL_PATH` from `on_policy_distillation.sh`). Scores the student's perturbed rollout token-by-token via teacher log-probs. Per-token reverse-KL: $\mathrm{KL}_t = \sum_v p_{\text{stu}}(v \mid h_t) \cdot [\log p_{\text{stu}}(v \mid h_t) - \log p_{\text{tch}}(v \mid h_t)]$, restricted to OPD's top-k token set (reuse `log_prob_top_k`, `top_k_strategy`, `teacher_temperature` from `verl/workers/config/rollout.py:142` — read directly, do not modify the config class). Returns the per-token vector $L_t$ to `apply_node_update`, which uses the richer per-token credit-assignment update.

**Sign convention.** `loss_fn` returns a quantity to be **minimized**: lower $L_t$ = student closer to teacher. The plan's $\mathrm{KL}_t$ is positive reverse-KL (already minimization-oriented). OPD's existing `dp_actor.compute_distillation_reward` returns `rm_scores` $= -\mathrm{kl\_val} \cdot w$ which is the **negated, maximization-oriented** form (higher = better). If we ever reuse `compute_distillation_reward` to source the per-token signal, **negate it before passing to `apply_node_update`** so $(L_+ - L_-)$ carries the correct sign in $\delta \hat{W} \propto (L_+ - L_-) \cdot (\delta y / \lVert \delta y \rVert^2) \otimes x$. Document this in the `loss_fn` docstring.

Both modes implement the same `loss_fn` contract; the trainer doesn't know which is active. The teacher engine in OPD mode is launched once at `init_workers` time alongside the student engines.

### Per-token aggregation knob

`apply_node_update` sums per-token outer products: $\delta \hat{W} = \sum_t (L_{+,t} - L_{-,t}) \cdot (\delta y_t / \lVert \delta y_t \rVert^2) \otimes x_t / 2$. With variable-length sequences, **sum** weights longer rollouts more heavily; **mean** ($\sum_t / T_{\text{resp}}$) equalizes per-prompt contribution. Expose:

```yaml
np:
  token_agg: "sum"    # or "mean"; affects how per-token signals are combined into delta_W_hat
```

`sum` is the literal Dalm-2024 form and the default. `mean` folds a $1/T_{\text{resp}}$ into the update — equivalent to absorbing it into `lr`, but explicit so per-prompt weighting is uniform regardless of generation length. Apply the mask from `response_mask` either way (skip prompt + padding tokens).

## `perturb_rules` semantics

YAML:

```yaml
np:
  perturb_rules:
    - "model\\.layers\\.\\d+\\.mlp\\.up_proj"
    - "model\\.layers\\.\\d+\\.self_attn\\.q_proj"
    - "model\\.layers\\.0\\.mlp\\.down_proj"          # specific layer
  layer_schedule: "one_per_step"                      # or "all_per_step"
  tape_kind: "token"                                  # or "sequence"
  sigma: 0.01
  lr: 1.0e-4
  antithetic: true
  loss_type: "grpo"                                   # or "opd"
  grpo_n: 4
  token_agg: "sum"                                    # or "mean"; see §"Per-token aggregation knob"
  population_size: 1                                  # v2; 1 = v1 antithetic-only
  population_mode: "antithetic"                       # v2; "antithetic" | "zscore"
```

Resolution rule: `active_modules = {m for m in model.named_modules() if any(re.fullmatch(r, m_name) for r in perturb_rules)}`. The three user-stated granularities (layer-type, layer-number, specific-layer) all collapse into a single regex space — `mlp\.up_proj$` matches type, `layers\.5\..*` matches number, fully-qualified matches specific.

## vLLM kwargs (the compat dance)

`NPNcclLLM(LLM)` mirrors `ESNcclLLM` but force-overrides three engine kwargs:

```python
class NPNcclLLM(LLM):
    def __init__(self, *args, **kwargs):
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        kwargs["enforce_eager"] = True          # mandatory: hooks die under torch.compile
        kwargs["enable_prefix_caching"] = False # already off in ES, keep off
        super().__init__(*args, **kwargs)
```

Throughput drop vs ES: rough order-of-magnitude **3–6× slower wallclock per equivalent compute** (~2× from doubled forwards, ~1.5–3× from `enforce_eager=True`, partially offset by skipping the population dimension). This is the cost of doing this on vLLM; no clean alternative exists in 0.11.0 (vLLM RFC #36998 for an "observation plugin" is open but unmerged).

## Files to add (sibling layout)

| Path                                                              | Purpose                                                                                                                                                                                                                                                      | Approx LOC |
| ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------- |
| `verl/verl/trainer/main_np.py`                                  | Hydra entry. Clone `main_es.py:101–235` structure; swap to `RayNPTrainer`; keep the same task_type dispatch (reuse `verl.trainer.es.task_utils.get_task_components`, including the `opd_math` branch we already added).                             | ~240       |
| `verl/verl/trainer/np/__init__.py`                              | empty package marker                                                                                                                                                                                                                                         | —         |
| `verl/verl/trainer/np/ray_trainer.py`                           | `RayNPTrainer` + `NPNcclLLM`. Reuse `_launch_engines` / `_init_inter_engine_group` / `_evaluate_model` / `_evaluate_with_engine` patterns verbatim from `es/ray_trainer.py`. The only material rewrite is `fit()`.                           | ~600       |
| `verl/verl/trainer/np/loss_fns.py`                              | `make_grpo_loss(config, reward_fn)` and `make_opd_kl_loss(config, teacher_engines, tokenizer)` returning the `loss_fn` callable.                                                                                                                       | ~180       |
| `verl/verl/trainer/config/np_trainer.yaml`                      | Mirror `es_trainer.yaml`; replace ES knobs with the `np.*` block above.                                                                                                                                                                                  | —         |
| `verl/verl/workers/rollout/vllm_rollout/np_worker_extension.py` | The hook-installer + capture/perturb/update/broadcast methods described above.                                                                                                                                                                               | ~450       |
| `opd_np.sh`                                                     | Top-level launcher. Clone `opd_es.sh` env-var style; expose `PERTURB_RULES` (newline-separated regex list), `LAYER_SCHEDULE`, `TAPE_KIND`, `SIGMA`, `LR`, `LOSS_TYPE`, `GRPO_N`, `TEACHER_MODEL_PATH` (for OPD mode), `N_GPUS_PER_NODE`. | ~110       |

No edits to any existing file. The just-merged ES path is untouched.

## Existing functions/utilities to reuse

- `verl.trainer.es.task_utils.get_task_components` — including the `opd_math` branch added in the prior plan. NP imports it from the ES module, no copy.
- `verl.utils.reward_score.ttrl_math.reward_func` — rule reward, already used by ES `opd_math` task type. The `grpo` loss wraps this.
- `verl.workers.config.rollout.RolloutConfig` (read-only) — for `log_prob_top_k`, `top_k_strategy`, `reward_weight_mode`, `teacher_temperature` defaults in OPD-loss mode. Don't modify; just read fields off `OmegaConf` overrides.
- `verl.utils.import_utils.load_extern_object` and `verl.utils.device.auto_set_device` — both added in the prior plan; main_np.py uses them the same way main_es.py does.
- vLLM-side NCCL primitives (`PyNcclCommunicator`, `StatelessProcessGroup`) — already wired in `es_worker_extension.py:13-19`; copy the same import block and the `init_inter_engine_group` method body verbatim.

## What we are explicitly NOT doing in v1

- No population-style sampling (`population_size`) — NP variance reduction is over tokens, not over independent samples. Add later if a need surfaces.
- No support for `enforce_eager=False`. vLLM RFC #36998 may eventually expose a hook-safe compile mode; revisit then.
- No backward pass through the captured `x` — `x` is detached at capture; we use it only as a rank-1 outer-product factor.
- No NP on attention internals (paged-attention fused op bypasses Python hooks). Hooks on `q_proj` / `k_proj` / `v_proj` / `o_proj` fire fine; attention itself doesn't. Document in the launcher comments.
- No checkpoint hot-swap with the ES trainer — they save into different `default_local_dir` paths and are not weight-compatible mid-run (NP modifies one layer at a time; the saved checkpoint format is standard HF, but conceptually they don't share a training trajectory).

## Risks

| Risk                                                                                                         | Why it matters                                                                                                                                                                                                                               | Mitigation                                                                                                                                                                                                                                         |
| ------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Forward hooks erased by `@support_torch_compile` on the outer model class                                  | Silent — no warning; hooks just don't fire, NP looks like it's running but the update is zero.                                                                                                                                              | Mandatory `enforce_eager=True` in `NPNcclLLM.__init__`. **Verification**: at step 0, log the number of times each hook was called over a known-batch rollout; assert it's $B \cdot T$ (or $B$, in sequence mode). If 0, fail loudly. |
| Fused linear layer output is a `(tensor, bias)` tuple                                                      | If the hook returns a tensor when vLLM expected a tuple, the next layer crashes with a cryptic shape error.                                                                                                                                  | Unpack-and-repack in the hook; cover with a unit test that perturbs `qkv_proj` end-to-end.                                                                                                                                                       |
| Per-token noise tape memory for long rollouts                                                                | $\delta y \in \mathbb{R}^{T \times d_{\text{out}}}$ per layer per rollout × population becomes significant if we ever go population. v1 has no population so this is bounded by $T \cdot d_{\text{out}} \cdot \#\text{active\_layers}$. | For 7B,$T=4096$, $d_{\text{out}}=4096$, 1 active layer: ~64 MB — fine. Document the formula in `opd_np.sh`.                                                                                                                                 |
| ANP normalization$\delta y / \lVert \delta y \rVert^2$ can explode when $\lVert \delta y \rVert$ is tiny | Numerical instability at small$\sigma$.                                                                                                                                                                                                    | Use Dalm's normalized form; clamp$\lVert \delta y \rVert^2 \geq \epsilon$; add a `np.update_clip` safety knob.                                                                                                                                 |
| OPD-mode teacher engine doubles GPU memory                                                                   | Two vLLM engines per GPU is generally not possible.                                                                                                                                                                                          | Teacher gets dedicated GPUs (e.g. 4 student engines + 4 teacher engines on an 8-GPU node), like `on_policy_distillation.sh` already does. Expose via `STUDENT_GPUS` / `TEACHER_GPUS` env vars in `opd_np.sh`.                              |
| Vendor lock to vLLM 0.11.0                                                                                   | An upgrade may change `@support_torch_compile` behavior or the `(output, bias)` contract.                                                                                                                                                | Same as ES: pin the version, add a `# vLLM internal: model_runner.model` comment, list in install script's known-good versions.                                                                                                                  |

## Verification

1. **Hook-fires smoke test (single GPU, no training).**

   ```bash
   python3 -m verl.trainer.main_np \
     np.perturb_rules='["model\\.layers\\.0\\.mlp\\.up_proj"]' \
     np.layer_schedule=one_per_step np.tape_kind=token \
     np.sigma=0.0 np.loss_type=grpo np.grpo_n=1 \
     model.path=$ACTOR_MODEL_PATH \
     data.task_type=opd_math \
     data.train_files=datasets/dapo-math-17k-1percent.parquet \
     data.val_files=datasets/test_data/math500/test.parquet \
     data.train_max_samples=4 data.val_max_samples=4 \
     trainer.n_gpus_per_node=1 trainer.nnodes=1 \
     trainer.default_local_dir=/tmp/np_smoke trainer.num_iterations=1
   ```
   With $\sigma=0$, NP is a no-op; rollouts must match the clean rollouts byte-for-byte. The hook-call counter we log at step 0 must be $B \cdot T$ (token mode) or $B$ (sequence mode). If 0, `enforce_eager` didn't take effect or the regex matched nothing.
2. **Gradient cosine-similarity check (offline, 1 GPU).** Pick one layer (e.g. `layers.0.mlp.up_proj`), one mini-batch, compute the true $\partial L / \partial W$ via a one-off PyTorch backward through an eager HF forward of the same model on the same batch, then compute NP's $\delta \hat{W}$ estimate over 100 antithetic ANP samples. Report $\cos(\delta \hat{W}, \nabla_W L)$ per layer — should converge to $\geq 0.1$ within a few hundred samples for token mode, lower for sequence mode. This is the standard health metric from the literature (Dalm 2024 §4); fail criterion if it's negative or stuck near 0.
3. **End-to-end GRPO-loss run** (small).

   ```bash
   ACTOR_MODEL_PATH=model/Qwen3-1.7B \
   PERTURB_RULES=$'model\\.layers\\.\\d+\\.mlp\\.up_proj\nmodel\\.layers\\.\\d+\\.self_attn\\.q_proj' \
   LAYER_SCHEDULE=one_per_step TAPE_KIND=token \
   SIGMA=0.01 LR=1e-4 LOSS_TYPE=grpo GRPO_N=4 \
   NUM_ITERATIONS=200 N_GPUS_PER_NODE=8 \
   bash opd_np.sh
   ```
   Watch wandb/SwanLab for monotonic `train/L_clean` improvement over ~100 steps. Loss curves will be noisier than PPO — that's expected — but should trend up.
4. **End-to-end OPD-loss run** (small). Same as above but `LOSS_TYPE=opd TEACHER_MODEL_PATH=model/Qwen3-4B-Non-Thinking-RL-Math STUDENT_GPUS=0,1,2,3 TEACHER_GPUS=4,5,6,7`. Validate that teacher engine starts, per-token KL is logged, and the per-token credit-assignment update path is exercised (verify by logging `‖δW‖` per active layer and confirming non-zero).
5. **ES regression check.** `bash opd_es.sh` (the existing weight-perturb ES launcher, committed at `301bf5c` on `feat/blocktt-svd-llamafactory`) on the 1% slice with prior known-good config must produce identical loss curves. NP code is in separate files on a separate branch; this is a belt-and-suspenders check that nothing was incidentally edited.
