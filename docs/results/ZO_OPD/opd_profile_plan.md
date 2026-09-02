# `es_token` Parallel-Rail Efficiency Plan

## Exploiting Decode Compute Headroom on an 8×A100 Node

### 1. Objective

This document evaluates the systems hypothesis behind `es_token`:

> During autoregressive student rollout, evaluate one clean decode rail plus multiple token-local perturbed rails while sharing the same model weights and historical KV cache. Because decode is often memory-bandwidth-bound, the additional arithmetic from perturbed rails may use otherwise idle GPU compute with little or no increase in clean-token latency.

The goal is not to assume that a fixed number such as `N=30` is free. The goal is to determine, analytically and experimentally,

$$
N_{\text{free}}

=

\max N

\quad\text{s.t.}\quad

\frac{T_{\text{decode}}(N)-T_{\text{decode}}(0)}

{T_{\text{decode}}(0)}

\le\epsilon,
$$

where the primary definition of **near-lossless** is $\epsilon=5\%$, with a secondary threshold of $10\%$.

The target platform is one node with **8 NVIDIA A100 GPUs**.

---

## 2. Hardware Assumptions

Primary assumption:

- 8× NVIDIA A100 80GB SXM GPUs
- NVLink/NVSwitch within the node
- BF16 student inference
- Dense or GQA Transformer as the first target
- CUDA Graph-compatible vLLM decode path
- All 8 GPUs are on one high-bandwidth scale-up domain

Per A100 80GB SXM, NVIDIA specifies approximately:

- Dense BF16 Tensor Core peak: **312 TFLOP/s**
- HBM2e bandwidth: **2.039 TB/s**
- GPU memory: **80 GB**
- NVLink bandwidth: **600 GB/s per GPU**

The 8-GPU aggregate is therefore approximately:

$$
C_{\text{node}}\approx8\times312

=2.50\ \text{PFLOP/s},
$$

$$
BW_{\text{HBM,node}}

\approx8\times2.039

=16.3\ \text{TB/s}.
$$

The aggregate numbers do **not** imply that one request can pool all HBM bandwidth without distributed communication. The relevant roofline is still primarily per GPU.

> If the actual machine uses A100 40GB, PCIe A100s, or a different topology, rerun the hardware calibration in Phase 0 before using the numerical rail limits below.

Official hardware reference: NVIDIA A100 specifications, https://www.nvidia.com/en-us/data-center/a100/

---

# Part I — Efficiency Analysis

## 3. What Is Actually Shared Across Rails?

At token $t$, let there be one clean rail and $N$ perturbed rails:

$$
R=N+1.
$$

The clean rail determines the sampled token and is the only rail whose current-token KV state is committed to the autoregressive cache.

The perturbed rails are ephemeral evaluations of the same clean autoregressive state.

### Shared across rails

If implemented with fused kernels:

1.**Base model weights**$W$ should be loaded from HBM once per tile and reused across rail activations.

2.**Historical KV cache**$K_{<t},V_{<t}$ is identical across rails under the state-detached `es_token` formulation and should be loaded once per attention tile.

3.**Sequence metadata** such as positions, page tables, masks, and clean-prefix metadata can largely be shared.

### Not shared

Each rail has its own current hidden state $h_t^{(r)}$, and therefore rail-specific

$$
q_t^{(r)},\quad k_t^{(r)},\quad v_t^{(r)}.
$$

Each rail also performs its own arithmetic through every subsequent layer after perturbation.

Therefore the correct statement is:

> Parallel rails can avoid approximately $R\times$ replication of persistent weight and historical-KV HBM traffic, but they do **not** avoid $R\times$ arithmetic.

This is exactly the desired trade:

$$
\text{unused arithmetic capacity}

\longrightarrow

\text{additional ES samples}.
$$

---

## 4. Linear-Layer Roofline Model

Consider a BF16 linear layer:

$$
Y=XW^\top.
$$

Let the local clean decode batch on one GPU be $B$. With $R$ rails, the effective GEMM $M$-dimension becomes

$$
M_{\text{eff}}=BR.
$$

If all rails are fused into one GEMM, the same weight tile can be reused across all $BR$ token/rail vectors.

Ignoring activation traffic for the first-order model:

- one BF16 weight costs about 2 bytes from HBM;
- each token/rail performs about 2 FLOPs per weight.

Therefore

$$
AI_{\text{linear}}

\approx BR

\quad\text{FLOP/byte}.
$$

For A100 80GB SXM,

$$
AI_{\text{ridge}}

=

\frac{312\ \text{TFLOP/s}}

{2.039\ \text{TB/s}}

\approx153\ \text{FLOP/byte}.
$$

The ideal linear-layer memory-bound region is therefore approximately

$$
\boxed{BR\lesssim153}.
$$

Equivalently,

$$
\boxed{

N_{\text{free,ideal}}^{\text{linear}}

\approx

\left\lfloor\frac{153}{B}\right\rfloor-1.

}
$$

### Ideal upper bound on A100 80GB SXM

| Local clean batch $B$ | Maximum total rails $R$ from ideal roofline | Maximum extra rails $N$ |

|---:|---:|---:|

| 1 | 153 | 152 |

| 2 | 76 | 75 |

| 4 | 38 | 37 |

| 8 | 19 |**18**|

| 16 | 9 |**8**|

| 32 | 4 |**3**|

| 64 | 2 |**1**|

These are **upper bounds, not expected measured values**.

Peak Tensor Core throughput is not achieved at every GEMM shape, activation traffic is not zero, and several non-GEMM kernels scale with $R$. The measured transition to compute-bound execution can therefore occur earlier.

A reasonable first engineering target is:

| Local clean batch | Initial near-lossless search range |

|---:|---:|

| $B=4$ | $N=4,8,16,24,32$ |

| $B=8$ | $N=4,8,12,16,20$ |

| $B=16$ | $N=2,4,6,8,12$ |

| $B=32$ | $N=1,2,3,4,8$ |

For the expected `es_token` regime, **$B_{\rm local}=8$, $N=8$–16** is the most promising initial operating point on A100.

---

## 5. Why Weight Sharing Helps Even Though Every Rail Has Different Hidden States

For a rank-1 perturbation,

$$
W^{(r)}

=

W+\mu u_rv_r^\top,
$$

the rail output is

$$
y_r

=

Wx_r

+

\mu u_r(v_r^\top x_r).
$$

The perturbation correction costs only

$$
O(d_{\rm in}+d_{\rm out}),
$$

while the base matrix multiplication costs

$$
O(d_{\rm in}d_{\rm out}).
$$

The rank-1 correction is therefore cheap.

However, after the first perturbed operation,

$$
x_r\ne x_{r'},
$$

so the base term $Wx_r$ must still be computed independently for every rail.

The opportunity is **not** to reuse the matrix multiplication result. It is to compute

$$
W

\begin{bmatrix}

x_0 & x_1 & \cdots & x_N

\end{bmatrix}
$$

as one larger GEMM, loading each $W$ tile once and performing more Tensor Core work on it.

This is why the rail dimension should be internal to the model runner rather than represented as $N$ independent vLLM requests.

---

## 6. Attention Roofline with Shared Historical KV

The same principle applies more strongly to decode attention.

All rails share

$$
K_{<t},V_{<t},
$$

while each rail has a different query

$$
Q_t^{(r)}.
$$

A rail-aware attention kernel should conceptually operate on

$$
Q\in

\mathbb R^{B\times R\times H_q\times d_h}
$$

against one historical KV cache.

A KV tile should be loaded once and reused by all rail queries.

For GQA, define the native query reuse factor

$$
g=\frac{H_q}{H_{kv}}.
$$

A rough first-order attention arithmetic intensity model is then

$$
AI_{\text{attn}}

\propto gR.
$$

Using the same A100 ridge point,

$$
R_{\text{attn,ideal}}

\sim

\frac{153}{g}.
$$

Illustrative limits:

| GQA ratio $g$ | Ideal total rails $R$ | Extra rails $N$ |

|---:|---:|---:|

| 1 | 153 | 152 |

| 4 | 38 | 37 |

| 8 | 19 | 18 |

| 16 | 9 | 8 |

This estimate is intentionally approximate because actual decode-attention arithmetic intensity depends on context length, head dimension, paging, kernel tiling, and reduction implementation.

The key systems requirement is much more important than the exact formula:

> **Do not execute attention once per rail.** A useful implementation must reuse each historical KV tile across multiple rail queries within one kernel.

Otherwise the historical KV cache is effectively reread $R$ times and the main efficiency argument disappears.

---

## 7. Persistent Memory vs. Transient Rail Memory

The design avoids $R\times$ persistent copies of model parameters and historical KV cache.

But it does increase transient activation storage.

For hidden dimension $d$, BF16 rail activations are approximately

$$
M_{\text{rail}}

\sim

2BRd\ \text{bytes}
$$

per live tensor.

For example,

$$
B=8,\quad R=16,\quad d=4096
$$

gives only

$$
8\times16\times4096\times2

\approx1\ \text{MiB}
$$

for one hidden-state tensor.

This is small relative to multi-billion-parameter model weights and long-context KV caches, but multiple simultaneously live tensors and workspace allocations still need profiling.

---

## 8. Non-GEMM Kernels Can Become the Hidden Bottleneck

The linear-layer roofline alone is insufficient.

The following operations have traffic or execution work that grows approximately with $R$:

- RMSNorm
- LayerNorm if present
- residual adds
- SiLU/GELU
- elementwise gating
- RoPE on current-token states
- routing operations for MoE
- logits processing
- sampling-side statistics

They may be a small fraction of clean decode latency but become significant after adding rails.

Therefore the implementation should aggressively fuse

$$
\text{residual}

+

\text{norm}

+

\text{perturbation}
$$

and, where practical,

$$
\text{activation}

+

\text{gating}.
$$

The end-to-end free-rail budget is

$$
N_{\text{free}}

=

\min

\left(

N_{\text{linear}},

N_{\text{attn}},

N_{\text{elementwise}},

N_{\text{LM-head}},

N_{\text{communication}}

\right).
$$

---

## 9. LM Head Must Not Materialize All Rail Logits

For vocabulary size $V$, naively producing

$$
[B,R,V]
$$

BF16 logits can create substantial rail-dependent memory traffic.

This is unnecessary for `es_token`.

### Clean rail

The clean rail needs sufficient logits/statistics to perform normal sampling.

### Perturbed rails

For Sampled-Token OPD, each rail primarily needs:

- the sampled-token logit;
- the log-normalizer / log-probability.

For top-$k$ OPD, each rail needs only the selected support.

The preferred LM-head kernel is therefore streaming:

1. load one vocabulary-weight tile;
2. compute clean and perturbed rail dot products;
3. update clean sampling statistics;
4. update rail-wise streaming `logsumexp`;
5. gather only selected token logits;
6. never materialize the full $[B,R,V]$ tensor.

This kernel is important because otherwise the LM head can erase much of the bandwidth advantage gained in the Transformer blocks.

---

# Part II — One-Node Distributed Analysis

## 10. 8-Way Data Parallelism

If the entire student model fits on one A100, 8-way DP is the cleanest configuration.

For global clean batch $B_{\rm global}$,

$$
B_{\rm local}

=

\frac{B_{\rm global}}{8}.
$$

Each GPU independently has:

- one model replica;
- independent prompts;
- independent historical KV;
- independent perturbation rails.

No decode-time cross-GPU collective is required.

Therefore DP preserves the single-GPU rail roofline directly.

Example:

$$
B_{\rm global}=64
$$

with DP=8 gives

$$
B_{\rm local}=8.
$$

The ideal A100 linear roofline then permits up to roughly

$$
R\approx19,\qquad N\approx18.
$$

This is substantially more favorable than putting all 64 clean sequences on one GPU, where little compute headroom remains.

### Preferred configuration when the model fits

$$
\boxed{

\text{DP}=8,\quad

B_{\rm local}\approx4\text{--}16

}
$$

is the best initial distributed target for `es_token`.

---

## 11. Tensor Parallelism

Suppose TP degree is $P$.

Each GPU stores roughly $1/P$ of the relevant linear weights and performs roughly $1/P$ of the corresponding GEMM FLOPs.

Therefore the local compute-to-weight-bandwidth ratio remains approximately unchanged:

$$
\frac{F/P}{M/P}

=

\frac{F}{M}.
$$

Thus:

$$
\boxed{

\text{TP does not multiply the per-GPU free-rail budget.}

}
$$

The same approximate $BR\lesssim153$ compute/HBM bound still applies.

### New bottleneck: collectives

TP introduces all-reduce / reduce-scatter communication whose payload increases with rail count.

For hidden size $d$ and BF16 activations, the basic rail-dependent tensor size is approximately

$$
S_{\rm collective}

\sim

2BRd\ \text{bytes}.
$$

Example:

$$
B=8,\quad R=16,\quad d=4096
$$

gives roughly

$$
1\ \text{MiB}
$$

per such activation tensor.

On an 8×A100 SXM/NVSwitch node, the physical NVLink bandwidth is high enough that this is plausible, but actual NCCL latency, collective algorithm, topology, and the number of collectives per layer must be measured.

The communication model is

$$
T_{\rm comm}(R)

\approx

\alpha+\beta S(R).
$$

At small clean decode payloads, $\alpha$ can dominate. Increasing $R$ can initially improve communication efficiency by amortizing fixed collective latency, but eventually the $\beta S(R)$ term grows linearly with rails.

Therefore TP can remain compatible with multiple rails, but

$$
N_{\rm comm}
$$

may become smaller than the local HBM roofline limit.

---

## 12. Distributed Configurations to Benchmark

Use the same 8-GPU node and compare:

### A. DP8

$$
DP=8,\quad TP=1.
$$

Best case when model replication fits.

### B. DP4 × TP2

$$
DP=4,\quad TP=2.
$$

Useful for larger models while retaining multiple independent rollout groups.

### C. DP2 × TP4

$$
DP=2,\quad TP=4.
$$

More communication-sensitive.

### D. TP8

$$
DP=1,\quad TP=8.
$$

Necessary for models that require full-node sharding; likely the most communication-sensitive configuration.

The important quantity is always **local clean batch per TP group**, not just global batch.

---

## 13. Pipeline Parallelism

PP is not required for the first implementation but offers a second source of slack.

Ordinary autoregressive decoding can leave pipeline stages idle while a clean token moves through the pipeline.

Perturbation rails do not determine the clean sampled token, so they can potentially fill pipeline bubbles as low-priority microbatches.

Conceptually:

```text

time →

stage 0: clean(t)   rail(t,1) rail(t,2) clean(t+1) ...

stage 1:            clean(t)  rail(t,1) rail(t,2) ...

stage 2:                      clean(t)  rail(t,1) ...

```

This is a different opportunity from the per-GPU HBM roofline:

- rail batching exploits unused arithmetic intensity **within a stage**;
- PP scheduling exploits idle time **between stages**.

Do not include PP in the first milestone. Establish the single-GPU and TP/DP behavior first.

---

## 14. Context Parallelism

Context parallelism shards the historical KV state.

Within each context-parallel rank, multiple rail queries can still reuse the local KV shard.

However, partial attention outputs and/or normalization statistics must be communicated across ranks, and this communication scales with $R$.

Therefore CP should also be treated as a later-stage configuration. The rail idea remains valid locally, but the communication roofline can dominate.

---

## 15. MoE / Expert Parallelism Is a Separate Case

For a dense Transformer, all rails access the same weights.

For MoE, a perturbation may alter router decisions:

$$
E_t^{(r)}

\ne

E_t^{(0)}.
$$

Different rails can therefore request different experts, causing:

- additional expert-weight HBM traffic;
- lower weight reuse;
- larger EP all-to-all traffic.

This breaks one of the central assumptions of the dense/GQA rail model.

Two possible variants exist:

1.**Exact perturbed routing** — correct but potentially expensive.

2.**Clean-route sharing** — all rails use the clean rail's expert set, preserving weight reuse but introducing a stop-gradient / fixed-routing approximation.

The first `es_token` efficiency study should use a dense or fixed-routing model. MoE should be evaluated separately.

---

# Part III — Required Runtime Design

## 16. Rail as an Internal Tensor Dimension

Do **not** represent perturbation rails as independent vLLM requests.

Scheduler-visible requests should remain the clean batch:

$$
B.
$$

Inside the model runner, introduce a rail dimension:

$$
[B,R,d].
$$

The scheduler owns only the clean trajectories and KV pages.

This is required to enable real HBM reuse.

---

## 17. Linear Kernel

For each linear operation:

1. flatten

$$
[B,R,d_{\rm in}]

\rightarrow

[BR,d_{\rm in}];
$$

2. perform one base GEMM;
3. fuse or cheaply apply the rail-specific rank-1/node perturbation correction;
4. reshape back to

$$
[B,R,d_{\rm out}].
$$

Benchmark both:

- standard cuBLAS/CUTLASS GEMM with flattened rails;
- a customized fused perturbation epilogue.

The standard flattened GEMM is the first baseline and may already capture most of the weight-reuse benefit.

---

## 18. Shared-KV Attention Kernel

Required semantics:

- one clean historical KV cache;

-$R$ rail-specific queries;

- KV tile loaded once;
- all rail queries processed before the tile is discarded;
- only clean current-token KV is appended to the persistent cache.

The first prototype can compare:

### Naive reference

Run existing decode attention $R$ times against the same KV pointer.

### Desired fused implementation

Treat the rail dimension similarly to an additional query-group dimension and reuse KV tiles explicitly.

The ratio between these two implementations directly measures the value of true shared-KV execution.

---

## 19. Perturbation Buffer and CUDA Graphs

Avoid dynamic Python-side sampling during each decode step.

Use persistent graph-stable buffers or deterministic counter-based perturbation generation.

A perturbation should be reproducible from identifiers such as

$$
(\text{global step},b,t,r,l).
$$

Preferred properties:

- fixed memory addresses;
- no graph recapture per token;
- cheap regeneration during later gradient assembly;
- double buffering if the next token's perturbation is prepared while the current token is decoding.

---

## 20. Clean-Rail Priority

The clean rail is on the critical autoregressive path.

Perturbed rails are auxiliary work.

Therefore the scheduling objective is

$$
\min T_{\rm clean-token}
$$

subject to maximizing completed perturbation evaluations.

Two implementations should be compared:

### Synchronous fused rails

Clean and perturb rails are executed in the same GEMM/attention kernel.

Advantages:

- strongest HBM reuse;
- simple accounting.

Risk:

- once $R$ is too large, the clean rail waits for all extra arithmetic.

### Slack-filling asynchronous rails

Clean work receives priority; perturbation work consumes idle resources.

Advantages:

- potentially preserves clean critical path more aggressively.

Risks:

- CUDA stream priority alone does not guarantee fine-grained preemption of already-running GPU kernels;
- perturbation backlog can grow if average work exceeds available slack;
- weaker weight/KV tile reuse if work is separated into different kernels.

The synchronous fused design should be implemented first. Asynchronous slack stealing is a second optimization.

---

# Part IV — Experimental Test Plan

## 21. Phase 0 — Hardware and Baseline Calibration

Before modifying vLLM:

1. Record exact GPU SKU:

- A100 40GB vs 80GB;
- SXM vs PCIe.

2. Record topology:

```bash

nvidia-smi topo -m

```

3. Record clocks, power limits, CUDA version, driver version, NCCL version.
4. Verify NVLink health and topology.
5. Measure sustainable HBM bandwidth rather than relying only on the specification.
6. Measure GEMM throughput for the exact model dimensions.
7. Establish a clean vLLM decode baseline.

### Output

A hardware calibration table containing:

- measured HBM bandwidth;
- GEMM TFLOP/s by shape;
- clean decode latency;
- Tensor Core utilization;
- DRAM bandwidth utilization;
- NVLink/NCCL baseline.

---

## 22. Phase 1 — Linear Rail Microbenchmark

### Purpose

Test the core hypothesis:

> Can increasing $R$ reuse one weight load and increase compute without increasing latency until the A100 roofline is approached?

### Dimensions

Use actual model shapes, including:

- Q/K/V projections;
- output projection;
- MLP up/gate;
- MLP down.

Example dimensions should include $d=4096$ and the target model's actual intermediate size.

### Sweep

Clean batch:

$$
B\in\{1,2,4,8,16,32,64\}.
$$

Total rails:

$$
R\in\{1,2,4,8,12,16,20,24,32\}.
$$

Skip obviously impossible combinations after the transition has been observed.

### Implementations

1. Serial $R$ separate GEMMs.
2. Flattened $[BR,d]$ GEMM.
3. Flattened GEMM + perturbation epilogue.

### Metrics

For every $(B,R)$:

- kernel latency;
- achieved TFLOP/s;
- HBM bandwidth;
- SM utilization;
- Tensor Core utilization;
- bytes read/written;
- arithmetic intensity;
- relative latency:

$$
T(B,R)/T(B,1).
$$

### Main output

Plot

$$
R

\rightarrow

\text{latency overhead}
$$

for each $B$.

Extract

$$
N_{\rm free}^{5\%}(B),

\qquad

N_{\rm free}^{10\%}(B).
$$

### Hypothesis

For A100 80GB SXM:

-$B=8$: transition should occur before or around $R\approx19$;

-$B=16$: before or around $R\approx9$;

-$B=32$: before or around $R\approx4$.

Measured limits smaller than these are expected; measured limits substantially larger would indicate that the relevant GEMM compute ceiling is below nominal peak or that the simple roofline model is missing important effects.

---

## 23. Phase 2 — Shared-KV Attention Microbenchmark

### Purpose

Determine whether the same rail reuse exists for historical KV traffic.

### Sweep

Clean batch:

$$
B\in\{1,4,8,16\}.
$$

Rails:

$$
R\in\{1,2,4,8,16,24,32\}.
$$

Context lengths:

$$
L\in\{128,512,2048,8192,32768\},
$$

restricted by model/configuration memory.

Include the actual target model's:

- number of query heads;
- number of KV heads;
- head dimension.

### Compare

1. Existing attention called $R$ times.
2. Batched queries if supported by an existing kernel.
3. Custom rail-aware shared-KV attention.

### Metrics

- latency;
- historical KV bytes read;
- achieved HBM bandwidth;
- attention FLOPs;
- Tensor Core / CUDA core utilization;
- scratch/workspace bytes;
- output bytes.

### Success criterion

A fused rail-aware kernel should show substantially sublinear historical KV traffic growth with $R$.

If HBM traffic still scales approximately linearly with $R$, the intended shared-KV optimization has not been achieved.

---

## 24. Phase 3 — Non-GEMM and Fusion Audit

Profile one decoder layer with

$$
R=1,2,4,8,16.
$$

Break latency into:

- linear GEMMs;
- attention;
- RMSNorm;
- residual operations;
- activation/gating;
- RoPE;
- perturbation operations;
- memory copies;
- kernel launch gaps.

Determine which operations grow into bottlenecks as GEMMs become more compute-efficient.

Implement fusion only where profiling demonstrates material benefit.

---

## 25. Phase 4 — LM-Head Benchmark

Compare:

### Baseline A

Materialize full

$$
[B,R,V].
$$

### Baseline B

Run $R$ independent LM heads.

### Desired

One rail-batched LM head with:

- clean sampling;
- rail-wise streaming `logsumexp`;
- sampled-token gather;
- optional fixed top-$k$ support gather.

Measure:

- latency;
- HBM traffic;
- output memory;
- maximum useful $R$.

The desired implementation should avoid an $O(BRV)$ stored-logit tensor.

---

## 26. Phase 5 — Full Single-GPU Decoder

Integrate:

- rail-batched linears;
- shared-KV attention;
- perturbation application;
- rail-efficient LM head.

### Sweep

$$
B\in\{1,2,4,8,16,32\}
$$

and

$$
N\in\{0,1,2,4,8,12,16,20,24,32\}.
$$

### Context

Test at least:

- short context: 512;
- medium context: 2K;
- long context: 8K+.

### Primary metric

Clean token latency:

$$
\text{overhead}(N)

=

\frac{T_{\text{token}}(N)}

{T_{\text{token}}(0)}

-1.
$$

### Secondary metrics

- perturbation evaluations/sec;
- total rail-token evaluations/sec;
- HBM utilization;
- Tensor Core utilization;
- memory footprint.

### Main result

Produce a heat map:

$$
(B,N)

\rightarrow

\text{clean-token latency overhead}.
$$

Mark contours at:

- 5%;
- 10%;
- 25%.

This directly defines the usable operating region.

---

## 27. Phase 6 — 8-GPU Distributed Benchmark

Test four configurations:

| Configuration | DP | TP |

|---|---:|---:|

| DP8 | 8 | 1 |

| DP4-TP2 | 4 | 2 |

| DP2-TP4 | 2 | 4 |

| TP8 | 1 | 8 |

For each configuration, hold model, global workload, and context constant as much as possible.

### Sweep

Local clean batch per TP group:

$$
B_{\rm local}\in\{1,2,4,8,16\}.
$$

Rails:

$$
N\in\{0,2,4,8,12,16\},
$$

extending higher only where the measured single-GPU roofline permits.

### Distributed metrics

In addition to Phase 5:

- NCCL collective latency per layer;
- percentage of decoder time in communication;
- NVLink TX/RX;
- collective message size;
- communication/computation overlap;
- p50/p95 clean-token latency.

### Questions to answer

1. Does DP reproduce the single-GPU rail curve?
2. At what $R$ does TP communication become the bottleneck?
3. Does increasing rail payload initially amortize collective fixed latency?
4. Is TP2/TP4 materially better than TP8 for `es_token`?
5. For a fixed global batch, is it better to spend GPUs on DP to reduce $B_{\rm local}$, thereby creating more rail headroom?

---

## 28. Phase 7 — End-to-End `es_token` Runtime

Only after the previous phases validate the hardware opportunity, integrate the full training path:

1. clean student rollout;
2. token-local perturbation rails;
3. clean-only KV commit;
4. saved sampled-token / top-$k$ statistics;
5. teacher prefill/scoring;
6. vectorized ES assembly;
7. parameter update.

Measure separately:

$$
T_{\rm rollout},

\quad

T_{\rm teacher},

\quad

T_{\rm assemble},

\quad

T_{\rm update}.
$$

The central end-to-end quantity is

$$
\boxed{

\text{gradient samples per wall-clock second}

}
$$

rather than raw decode FLOPs.

Compare against:

1. clean OPD rollout;
2. conventional serial ES perturbation evaluations;
3. independent perturbed rollouts;

4.`es_token` parallel rails.

---

# Part V — Decision Criteria

## 29. Go / No-Go Gates

### Gate 1 — Linear reuse

**Go** if at $B_{\rm local}=8$,

$$
N\ge8
$$

adds less than 5–10% linear-layer latency for the dominant GEMMs.

If even $N=4$ causes large overhead, the main compute-headroom hypothesis is weak on A100.

### Gate 2 — Shared-KV attention

**Go** if a rail-aware attention kernel keeps historical KV traffic approximately constant or strongly sublinear in $R$, and $N=8$ remains near-lossless for representative context lengths.

### Gate 3 — Full single-GPU decoder

Strong target:

$$
B_{\rm local}=8,\quad N=8\text{--}16
$$

with

$$
<10\%
$$

clean-token latency increase.

Excellent result:

$$
B_{\rm local}=8,\quad N\ge8
$$

with

$$
<5\%
$$

increase.

### Gate 4 — Distributed scaling

For intra-node TP, the method remains compelling if TP communication does not reduce $N_{\rm free}$ by more than roughly 2× relative to the single-GPU bound.

### Gate 5 — End-to-end value

The final criterion is not whether rails are literally free. It is whether `es_token` produces substantially more useful gradient estimates per second than conventional ES.

A practical target is

$$
\frac{\text{useful ES samples/sec}_{\rm es\_token}}

{\text{useful ES samples/sec}_{\rm serial}}

\gg1
$$

while keeping the clean rollout trajectory unchanged.

---

# Part VI — Expected Outcome on 8×A100

## 30. Working Prior Before Measurement

For A100 80GB SXM, the ideal dense-BF16 linear ridge point is only about

$$
153\ \text{FLOP/byte},
$$

so the free-rail budget is smaller than on GPUs with a higher compute-to-HBM ratio.

For a dense/GQA model with properly fused kernels:

### Single GPU / DP rank

At

$$
B_{\rm local}=8,
$$

the ideal linear limit is about

$$
N\le18.
$$

A realistic initial target is

$$
\boxed{N=8\text{--}16}.
$$

At

$$
B_{\rm local}=16,
$$

the ideal limit is about

$$
N\le8,
$$

so the realistic target is

$$
\boxed{N=4\text{--}8}.
$$

At

$$
B_{\rm local}=32,
$$

only a few rails are expected to be near-free:

$$
\boxed{N\approx1\text{--}3}.
$$

Therefore `es_token` should prefer **moderate local clean batch plus extra rail dimension**, rather than first maximizing the conventional clean decode batch.

---

## 31. Recommended First System Configuration

### If the model fits on one GPU

Use

$$
\boxed{

DP=8,\quad

B_{\rm local}=8,\quad

N\in\{4,8,12,16\}.

}
$$

This configuration avoids decode-time communication and directly tests the central hardware hypothesis.

### If the model requires sharding

Start with the smallest TP degree that fits the model:

$$
\boxed{TP=2\ \text{or}\ 4}
$$

and use remaining GPUs for DP.

Avoid TP8 as the first experiment unless required by model size, because it introduces the largest communication surface.

---

# 32. Core Research Claim to Test

The systems claim should not be:

> Decode is memory-bound, therefore arbitrary extra perturbation compute is free.

The precise claim is:

> `es_token` creates a perturbation rail dimension that reuses the same base weights and historical KV cache. Increasing the rail dimension raises arithmetic intensity without proportionally increasing persistent HBM traffic. Up to a hardware- and batch-dependent roofline, this converts otherwise unused decode compute capacity into additional zeroth-order gradient samples.

For A100, the first-order model is

$$
\boxed{

BR

\lesssim

\frac{C_{\rm BF16}}

{BW_{\rm HBM}}

\approx153

}
$$

for weight-dominated linears, followed by additional limits from attention, pointwise kernels, LM-head processing, and distributed communication.

The final system should therefore select $N$ from **measured runtime headroom**, not treat it as a fixed algorithmic hyperparameter.

A later optimization can make $N_t$ adaptive:

$$
N_t

=

f(

B_{\rm local},

L_t,

\text{GPU utilization},

\text{communication load}

),
$$

allowing `es_token` to consume available compute slack dynamically while protecting clean-token latency.

---

## 33. Deliverables

1.**Roofline microbenchmark report**

- measured $N_{\rm free}(B)$ for A100.

2.**Rail-batched linear prototype**

- flattened GEMM;
- perturbation epilogue.

3.**Shared-KV rail attention prototype**

- direct comparison with repeated attention calls.

4.**Streaming rail LM head**

- Sampled-Token OPD first;
- fixed-support top-$k$ second.

5.**Single-GPU decode heat map**

$$
(B,N)\rightarrow\text{latency overhead}.
$$

6.**8-GPU DP/TP scaling table**

- DP8;
- DP4×TP2;
- DP2×TP4;
- TP8.

7.**End-to-end `es_token` throughput comparison**

- useful perturbation samples/sec;
- clean token latency;
- total training-step wall time.

8.**Adaptive rail-budget policy**

- optional after fixed-$N$ behavior is understood.

---

## 34. Bottom Line

On an 8×A100 80GB SXM node, the strongest initial opportunity is not an unconditional `N=30`.

The more defensible target is

$$
\boxed{

B_{\rm local}=8

\quad\Rightarrow\quad

N\approx8\text{--}16

\text{ worth testing as near-free}

}
$$

with an ideal linear roofline ceiling around $N=18$.

For

$$
B_{\rm local}=16,
$$

the useful range is more likely

$$
N\approx4\text{--}8.
$$

Eight-way DP is the cleanest execution mode when the model fits per GPU. TP does not increase the compute/HBM rail budget, but intra-node A100 NVLink/NVSwitch may still support several rails before collective communication becomes dominant.

The project should therefore first establish the **measured free-rail frontier**

$$
\boxed{

N_{\rm free}

=

f(

B_{\rm local},

L,

\text{model shape},

TP,

\text{kernel design}

)

}
$$

and only then choose the `es_token` perturbation count used for training.
