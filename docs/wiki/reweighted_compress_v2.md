# reweighted_compress_v2: damage-aware compression done correctly (curvature-weighted error-feedback refinement)

> The v1 idea — "spend the compression budget on the tokens compression damaged"
> — is right, but [reweighted_compress](reweighted_compress.md) implemented it in
> the **wrong space with the wrong signal** and lost 5pp MATH. This page re-derives
> it from the end-to-end objective and arrives at a different, *correct* setting:
> reweight the per-layer **output reconstruction error** by the **loss-curvature
> (Fisher) metric** of the teacher–student KL, inside an **initialize → measure
> realized gap → refit → iterate** loop. The per-token weight that falls out is
> *task-leverage* (how much a layer's output error moves the prediction), not
> *teacher-uncertainty* (which is what v1's logit-KL actually measured).

Reads on top of [reasoning_aware_compress_calib](reasoning_aware_compress_calib.md)
(operating point, eval contract, the M2-objective null and M3-accumulation
mechanisms) and [reweighted_compress](reweighted_compress.md) (the v1 negative
result this corrects). Grounded entirely in primitives that **already exist** in
`src/compress/`: `svd_compress_layer_combined` (doubly-whitened SVD),
`collect_backward_covariances_from_loader` (output-grad covariance),
`sequential/relinearized.py` (depth-ordered re-linearization loop),
`hybrid/lr_sparse.py` (`refine_passes` error-feedback loop),
`calibration_opd_loss.py` (teacher–student KL gradient).

---

## 1. Why v1 failed — the two category errors

v1 put a per-token weight $w_t = \exp(\beta\,\tilde\delta_t)$ (from the global
logit KL $\delta_t = D_{\mathrm{KL}}(p^T_t \,\|\, p^S_t)$) into the **input
activation covariance** $C_x = \sum_t w_t\, x_t x_t^\top$, then recompressed. It
cost $-5$pp MATH. Two distinct mistakes, both visible in hindsight:

1. **Wrong space.** Upweighting $x_t$ in $C_x$ makes the layer preserve $W$
   *along the input direction $x_t$*. But the damage is a property of the
   **output/logit prediction**, layers downstream. Preserving the map along a
   high-KL token's *input* direction has no clean relation to fixing that token's
   *prediction*. The lever must act on the **output error**
   $\varepsilon_t = (W-\hat W)x_t$, in $d_{\text{out}}$ space — a *left* (output)
   metric, not a *right* (input) one.

2. **Wrong signal.** Forward KL $D_{\mathrm{KL}}(p^T \,\|\, p^S)$ is large wherever
   the student lost teacher probability mass — and the teacher spreads mass on
   intrinsically high-entropy, hard-to-predict tokens. So $\delta_t$ correlates
   with **teacher uncertainty**, not **task leverage**. v1 chased the
   hardest-to-predict tokens (high-variance, less-structured activations) and
   pulled rank/neuron budget toward noise. Confirmed by the data: PPL
   flat-to-worse while MATH dropped and `n_reached` fell — a *reasoning* loss, the
   budget went to the wrong tokens.

The fix for both falls out of writing down the actual end-to-end objective and
expanding it once.

## 2. The objective we actually care about

The thing to minimize is the **prediction damage** on the calibration trace —
the teacher–student KL summed over tokens, with special concern that no single
token's damage blows up (the "most degraded" tokens you name):

$$
\mathcal{L}_{\text{end}} \;=\; \sum_t \delta_t,
\qquad
\delta_t \;=\; D_{\mathrm{KL}}\!\big(p^T_t \,\|\, p^S_t\big).
$$

What every activation-aware method *actually* minimizes instead is a sum of
**local, per-layer, Euclidean** output-reconstruction errors:

$$
\mathcal{L}_{\text{local}} \;=\; \sum_\ell \sum_t \big\| (W_\ell - \hat W_\ell)\, x^\ell_t \big\|_2^2 .
$$

$\mathcal{L}_{\text{local}}$ is a **surrogate** for $\mathcal{L}_{\text{end}}$.
The v1/M2 results say the surrogate is already near-optimal *for the average
token* — which is why a naive objective tweak (D-block) was null. The leverage is
in the **mismatch between the two for the worst tokens**. We make that mismatch
explicit.

## 3. Linking local error → end damage (one Taylor step)

Let $\varepsilon^\ell_t = (W_\ell - \hat W_\ell)\, x^\ell_t \in \mathbb{R}^{d_{\text{out}}}$
be the output perturbation compressing layer $\ell$ injects at token $t$. To first
order it propagates through the rest of the frozen network to perturb the final
logits:

$$
\Delta z_t \;\approx\; \sum_\ell J^{\ell\to z}_t \, \varepsilon^\ell_t ,
\qquad
J^{\ell\to z}_t \;=\; \frac{\partial z_t}{\partial\,(\text{layer-}\ell\ \text{output})} .
$$

KL is locally a quadratic in the logit shift, with the **Fisher information** of
the teacher distribution as its Hessian:

$$
\delta_t \;\approx\; \tfrac12\, \Delta z_t^\top F_t\, \Delta z_t,
\qquad
F_t \;=\; \operatorname{diag}(p^T_t) - p^T_t\, {p^T_t}^{\!\top}
\;=\; \nabla^2_{z}\, D_{\mathrm{KL}}(p^T_t \,\|\, \cdot)\big|_{z=z^T_t}.
$$

Substituting and keeping the per-layer diagonal (cross-layer terms are
second-order in the perturbation):

$$
\boxed{\;
\delta_t \;\approx\; \tfrac12 \sum_\ell\; {\varepsilon^{\ell}_t}^{\!\top}\, H^\ell_t\, \varepsilon^\ell_t,
\qquad
H^\ell_t \;=\; {J^{\ell\to z}_t}^{\!\top}\, F_t\, J^{\ell\to z}_t \;\succeq\; 0.
\;}
$$

$H^\ell_t$ is the **output-error metric**: a PSD matrix in $d_{\text{out}}$ space
saying *how much an output error at layer $\ell$, token $t$, turns into KL damage
at the head*. This is the object v1 was missing. Three readings of the box:

- **The correct per-layer objective** is
  $\sum_t {\varepsilon^\ell_t}^{\!\top} H^\ell_t \varepsilon^\ell_t$, **not**
  $\sum_t \|\varepsilon^\ell_t\|_2^2$. It weights the output error by
  task-relevant curvature.
- **The correct per-token importance** is the *scale* of $H^\ell_t$ (e.g.
  $\operatorname{tr} H^\ell_t$), which is large for **leverage** tokens (small
  output error $\Rightarrow$ large prediction change), *independent of teacher
  entropy*. Opposite of v1's $\delta_t$.
- **The correct per-direction importance**: $H^\ell_t$ is not isotropic — it
  upweights the *output subspace* that the head is sensitive to. A plain scalar
  weight cannot express this; a **left whitening** can.

## 4. From the box to a computable compression step

$H^\ell_t$ per token is a $d_{\text{out}}\times d_{\text{out}}$ object we never
form. Two reductions make it the existing **doubly-whitened SVD**:

**(a) Aggregate over tokens into one left covariance.** Define the
**KL-curvature output covariance**

$$
G_\ell \;=\; \sum_t H^\ell_t
\;=\; \sum_t {J^{\ell\to z}_t}^{\!\top}\, F_t\, J^{\ell\to z}_t .
$$

This is *exactly* the second moment of the **back-propagated gradient of the
teacher–student KL** w.r.t. layer $\ell$'s output. Proof sketch: the gradient of
$\delta_t$ w.r.t. the layer-$\ell$ output is
$g^\ell_t = {J^{\ell\to z}_t}^{\!\top} F_t\, \Delta z_t$, and near the operating
point its second moment under the Fisher metric is $J^\top F J$. So

$$
G_\ell \;\approx\; \mathbb{E}_t\big[\, g^\ell_t\, {g^\ell_t}^{\!\top} \,\big]
\quad\text{with}\quad
g^\ell_t \;=\; \nabla_{(\text{layer-}\ell\ \text{out})}\, D_{\mathrm{KL}}(p^T_t \,\|\, p^S_t).
$$

**$G_\ell$ is the backward (output-gradient) covariance — already collected by
`collect_backward_covariances_from_loader` — but driven by the teacher–student
KL loss (`calibration_opd_loss`), not next-token CE.** This is the single most
important correction: the D-block's "backward objective" used CE on the *dense*
model (so it measured generic next-token saliency, found null/destructive); here
the backward loss is $D_{\mathrm{KL}}(p^T \,\|\, p^S)$ on the *compressed* student
vs the teacher, so it measures **realized compression damage curvature**.

**(b) The weighted layer objective is doubly-whitened SVD.** With input
covariance $C_x^\ell = \sum_t x^\ell_t\, {x^\ell_t}^{\!\top}$ (the standard right
whitening) and the KL-curvature output covariance $G_\ell$ (the left whitening),
the per-layer problem

$$
\hat W_\ell
\;=\; \arg\min_{\hat W}\ \sum_t {\varepsilon^\ell_t}^{\!\top} H^\ell_t\, \varepsilon^\ell_t
\;\approx\; \arg\min_{\hat W}\ \big\| G_\ell^{1/2}\, (W_\ell-\hat W)\, {C_x^\ell}^{1/2} \big\|_F^2
$$

is solved by truncated SVD of $M = \Phi_g^\top W \Phi_x$ with
$\Phi_g \Phi_g^\top = G_\ell$ and $\Phi_x \Phi_x^\top = C_x^\ell$ —
**`svd_compress_layer_combined`, already implemented** (svd_llm_v2.py:124). For
the MLP/Nystrom path the analogue is the existing joint fwd+bwd kernel
(`collect_nystrom_combined_statistics` $\to (C_f, C_b)$), with $C_b$ now the
KL-curvature instead of CE.

So **no new compression math is needed** — only (i) the *right loss* behind the
backward covariance and (ii) the loop that keeps it *realized* (next section).
The per-token "balance" you ask for is not a scalar $w_t$; it is the **left
metric $G_\ell$** that the doubly-whitened SVD already consumes — the principled
generalization of a per-token weight to a per-(token, direction) curvature.

### 4.1 Optional explicit worst-token balancing (minimax)
The box minimizes the *sum* $\sum_t \delta_t$. To explicitly protect the **most
degraded** tokens (the minimax flavor you emphasize), add an outer reweight on the
curvature aggregation:

$$
G_\ell^{(\alpha)} \;=\; \sum_t \big(\,\delta_t^{(k)}\,\big)^{\alpha}\; H^\ell_t,
\qquad \alpha \ge 0,
$$

where $\delta_t^{(k)}$ is the *realized* damage at the current refinement iterate
$k$. $\alpha = 0$ is the plain sum (Gauss–Newton). $\alpha > 0$ tilts the left
metric toward tokens that are *currently* most damaged — a reweighted-least-squares
step toward the minimax $\min \max_t \delta_t$. Unlike v1's $\exp(\beta\,\delta_t)$
on the **input** covariance, here $\delta_t$ multiplies the **curvature** (output,
task-leveraged) and is **recomputed each pass** against the realized student — so
it tracks *which tokens are still broken after the last refit*, not a stale
one-shot guess. $\alpha$ is the knob that was *morally* intended in v1 but applied
to the wrong object.

## 5. The structure: initialize → measure realized gap → refit → iterate

The Taylor expansion of §3 holds only at the operating point where $x^\ell_t$,
$J^{\ell\to z}_t$, and $\delta_t$ are evaluated. After compression those all move —
which is why a one-shot estimate is wrong (the v1 lesson, and mechanism **M3**,
cross-depth accumulation). The fix is to make the gap **realized** and close it by
**error feedback**, the structure you name ("initialize a compression, then refine
to mitigate the gap"):

```
initialize:   Ŵ⁽⁰⁾ ← uniform forward-only compress (current recipe)        [§5 baseline]
repeat k = 0,1,...:
  measure:    run teacher T and student S⁽ᵏ⁾ on the calib trace together;
              backprop  δ_t = D_KL(p^T_t ‖ p^S_t)  into every layer
              → realized KL-curvature output cov  G_ℓ⁽ᵏ⁾  (and δ_t⁽ᵏ⁾ for α)
              → re-collect input cov  C_x^ℓ⁽ᵏ⁾  against the COMPRESSED prefix
  refit:      Ŵ⁽ᵏ⁺¹⁾ ← doubly-whitened SVD / joint-Nystrom with (C_x⁽ᵏ⁾, G⁽ᵏ⁾)
              [optionally α-tilted, §4.1]
  stop when   Σ_t δ_t⁽ᵏ⁾  stops decreasing (or max passes)
```

Two realizations already exist as harnesses and just need the KL-driven $G$:

- **Depth-ordered (re-linearization).** `sequential_relinearized_compress`
  (relinearized.py) already compresses layer-by-layer, re-collecting each layer's
  covariance against the **already-compressed prefix** — the exact "measure the
  realized gap before refitting this layer" loop, single forward+backward per
  layer. It currently supports `objective="combined"` with an `opd_loss_fn`; pass
  a teacher–student **KL** loss as `opd_loss_fn` and it computes $G_\ell$ from
  realized damage. **This is the minimal correct implementation** — the v2 method
  with $\alpha = 0$, one depth pass.
  *Loss caveat (the one genuinely new piece):* `calibration_opd_loss` is a top-K
  **policy-gradient surrogate** (matches the OPD trainer's update); its backward
  second moment is *related to* but **not equal to** the Fisher-curvature $G_\ell$
  of §4. The faithful driver is the **plain forward KL** as a differentiable
  scalar — a ~10-line `kl_calibration_loss` computing
  $\sum_t \text{mask}_t\,\big(p^T_t \cdot (\log p^T_t - \log p^S_t)\big)$ with the
  teacher detached — which is the exact loss whose layer-output gradient covariance
  is $G_\ell$. The OPD surrogate is a worthwhile *second* driver (it weights by the
  curvature the trainer will actually optimize downstream), but the KL form is what
  §3 derives.
- **Whole-model iterate (error feedback).** `lr_sparse`'s `refine_passes` is the
  same loop for a residual term: re-fit against the **true deployed** activations
  each pass. The v2 analogue re-fits the *low-rank factor itself* (not a residual)
  against $(C_x^{(k)}, G^{(k)})$ each pass — a few global passes instead of a depth
  sweep.

### 5.1 Why this should beat v1 and the D-block null
- vs **v1**: weight is on the **output curvature** (task leverage), recomputed
  against the **realized** student each pass — not a one-shot input reweight by
  teacher-uncertainty. Fixes both §1 errors.
- vs **D-block (objective null)**: D used CE backward on the **dense** model with
  **no reweighting and no refit** $\to$ it measured generic saliency and tied D0.
  Here the backward loss is **teacher–student KL on the compressed student**, and
  it is **re-measured after each refit** $\to$ it measures *residual realized
  damage*, which is exactly the signal D never had. The null does **not** transfer.
- vs **M3/SRC**: SRC fixes the *input* distribution shift across depth; v2 adds
  the *output* curvature metric (the missing left whitening) on top of the same
  loop. Complementary — SRC is v2 with an isotropic left metric ($G_\ell = I$).

## 6. What to actually run (proposed; not yet executed)

Same cell as before: **Qwen3-4B non-thinking, sequence-reweight, full length,
retain 0.7, last layer dense, MATH-500(100) + C4 PPL**, so it is directly
comparable to v1 (67% anchor / 62% v1) and §5 (69%).

| cell | init | left metric $G$ | loop | $\alpha$ | hypothesis |
| ---- | ---- | --------------- | ---- | -------- | ---------- |
| **V0** | uniform fwd-only | — (none) | — | — | $=$ §5 anchor (67–69%) |
| **V1-repro** | — | — | — | — | the v1 negative (62%), for context |
| **C1** | uniform | KL-curvature $G$ (combined SVD), **1 global refit** | 1 pass | 0 | does the *output* metric + 1 refit help? |
| **C2** | uniform | KL-curvature $G$, **depth-ordered (SRC)** | per-layer | 0 | realized per-layer gap (the strong form) |
| **C3** | uniform | KL-curvature $G$, depth-ordered | per-layer | 1 | + worst-token (minimax) tilt |

Success: **C2 or C3 $>$ V0** on strict MATH at fixed budget, PPL not regressing.
Pre-registered falsifiers: (i) if C1 $\approx$ V0, the one-shot output metric is as
null as v1's input metric $\to$ the *loop* is what matters, look at C2; (ii) if
C2 $\approx$ V0 too, then the realized-curvature objective genuinely adds nothing
over uniform forward-only at 0.7 (a real finding: the surrogate
$\mathcal{L}_{\text{local}}$ is tight here and the whole
damage-aware-calibration thesis is dead at this ratio — move to lower ratios where
the surrogate gap is larger, per §4's "leverage grows past the cliff"); (iii) if
C3 $<$ C2, the minimax tilt over-focuses the tail (same failure shape as v1's
$\beta$), cap or lower $\alpha$.

Cost: C1 $=$ 1 extra fwd+bwd KL pass + 1 recompress. C2 $= O(N_{\text{layers}})$
fwd+bwd passes (SRC is already this cost). Backward over the 4B at full-seq is the
memory ceiling — reuse the `max_seq_len=4096` truncation the combined path already
uses.

## 7. Status

**Derivation + design only (2026-06-08). Not implemented, not run.** The math
points at primitives that already exist — the implementation is "wire
`calibration_opd_loss` (KL form) into `sequential_relinearized_compress`'s
`opd_loss_fn` with `objective='combined'`, add the optional $\delta_t^{\alpha}$
curvature tilt, run C1/C2/C3." Open question this resolves vs leaves: it explains
the v1 failure and gives the correct lever, but whether the corrected lever clears
the surrogate-tightness bar **at retain 0.7** is exactly what C2 tests — the honest
prior (from M2-null + v1-negative) is that the gain, if any, lives **below the
cliff**, so a 0.6/0.5 arm should be ready to run if 0.7 is flat.

**One-line takeaway**: v1 reweighted the *input* by *teacher-uncertainty*
one-shot; the correct method reweights the *output reconstruction* by *realized
KL-curvature (task-leverage)* inside an *iterative error-feedback loop* — which is
the doubly-whitened SVD the repo already has, driven by a teacher–student-KL
backward and re-measured after each refit.
