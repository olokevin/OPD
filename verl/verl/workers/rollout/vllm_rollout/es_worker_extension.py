import gc
import time
import random
import numpy as np
import torch
import os
import inspect
try:
    from vllm.forward_context import set_forward_context
except ImportError:
    set_forward_context = None

def _stateless_init_process_group(master_address, master_port, rank, world_size, device):
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup
    pg = StatelessProcessGroup.create(
        host=master_address, port=master_port, rank=rank, world_size=world_size
    )
    return PyNcclCommunicator(pg, device=device)

class WorkerExtension:
    """
    Methods used by the ES trainer:
    - perturb_self_weights(seed, sigma_or_scale, coeff=1.0, negate=False)
    - restore_self_weights(seed, SIGMA)
    - update_weights_from_seeds(seeds, coeffs)  <-- NEW METHOD
    - init_inter_engine_group(master_address, master_port, rank, world_size)
    - broadcast_all_weights(src_rank)
    - save_self_weights_to_disk(filepath)
    
    Ensemble methods:
    - store_base_weights()
    - apply_perturbation(seed, sigma)
    - reset_to_base_weights()
    - get_next_token_logits(input_ids)
    """
    def _set_seed(self, seed):
        # set a seed locally on the worker extension for reproducibility
        self.local_seed = seed

        # seeding
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def perturb_self_weights(self, seed, noise_scale, negate=False):
        self._set_seed(seed)
        scale = float(noise_scale)
        sign = -1.0 if negate else 1.0
        for _, p in self.model_runner.model.named_parameters():
            gen = torch.Generator(device=p.device)
            gen.manual_seed(int(seed))
            noise = torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=gen)
            p.data.add_(sign * scale * noise)
            del noise
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return True

    def restore_self_weights(self, seed, SIGMA, negate=False):
        """Undo perturbation. Must use same negate value as perturb_self_weights."""
        self._set_seed(seed)
        sign = -1.0 if negate else 1.0  # Same sign as perturb
        for _, p in self.model_runner.model.named_parameters():
            gen = torch.Generator(device=p.device)
            gen.manual_seed(int(seed))
            noise = torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=gen)
            # Undo: subtract what we added (sign * sigma * noise)
            p.data.add_(-sign * float(SIGMA) * noise)
            del noise
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return True

    def update_weights_from_seeds(self, seeds, coeffs, alpha, population_size):
        """
        Mimics the Original implementation's update loop structure:
        Iterate Param -> Iterate Seeds -> Accumulate -> Single Update.
        """
        # seeds and coeffs should be lists of equal length
        # coeffs[i] should be: (alpha / population_size) * normalized_reward
        
        for _, p in self.model_runner.model.named_parameters():
            # Use model's native dtype for accumulator to save memory
            # Scale coefficients to avoid precision issues
            update_accumulator = torch.zeros_like(p.data)
            
            for i, seed in enumerate(seeds):
                self._set_seed(seed)
                gen = torch.Generator(device=p.device)
                gen.manual_seed(int(seed))
                
                # Generate noise in native precision
                noise = torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=gen)
                
                # Scale and accumulate in-place (memory efficient)
                # Use float32 coefficient to maintain precision
                update_accumulator.add_(noise, alpha=float(coeffs[i]))
                
                # Immediately free noise tensor
                del noise
            
            # div by population_size multiply by alpha (scalar)
            # Apply update in-place
            update_accumulator.mul_(alpha / population_size)
            p.data.add_(update_accumulator)
            
            del update_accumulator
            
            # Periodically clear cache to prevent fragmentation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return True

    def get_worker_ip(self):
        """Return the IP address of this worker's node."""
        from vllm.utils import get_ip
        return get_ip()

    def init_inter_engine_group(self, master_address: str, master_port: int, rank: int, world_size: int):
        self.inter_pg = _stateless_init_process_group(
            master_address, master_port, rank, world_size, self.device
        )
        return True

    def broadcast_all_weights(self, src_rank: int):
        for _, p in self.model_runner.model.named_parameters():
            self.inter_pg.broadcast(p, src=int(src_rank), stream=torch.cuda.current_stream())
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return True

    def save_self_weights_to_disk(self, filepath):
        state_dict_to_save = {}
        for name, p in self.model_runner.model.named_parameters():
            state_dict_to_save[name] = p.detach().cpu()
        torch.save(state_dict_to_save, filepath)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(0.1)
        return True
    
    def dump_noise_for_seed(self, seed: int, out_dir: str):
        """
        Generate per-parameter noise using the same method as perturb/restore
        and save them to disk for determinism comparison.
        """
        os.makedirs(out_dir, exist_ok=True)
        noise_state = {}
        for name, p in self.model_runner.model.named_parameters():
            gen = torch.Generator(device=p.device)
            gen.manual_seed(int(seed))
            noise = torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=gen)
            noise_state[name] = noise.detach().cpu()
            del noise
        torch.save(noise_state, os.path.join(out_dir, f"noise_seed_{int(seed)}.pt"))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        return True
    
    # debug
    def print_model_weights_stats(self):
        for name, p in self.model_runner.model.named_parameters():
            print(f"Param: {name}, Shape: {p.shape}")
        return True
    
    # ==================== Ensemble Methods ====================
    
    def store_base_weights(self):
        """Store a copy of current weights as base weights for ensemble."""
        self._base_weights = {}
        for name, p in self.model_runner.model.named_parameters():
            self._base_weights[name] = p.data.clone()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return True
    
    def apply_perturbation(self, seed, sigma):
        """Apply perturbation from base weights (not current weights)."""
        if not hasattr(self, '_base_weights'):
            raise RuntimeError("Must call store_base_weights first")
        
        self._set_seed(seed)
        for name, p in self.model_runner.model.named_parameters():
            # Restore base weights first
            p.data.copy_(self._base_weights[name])
            # Then apply perturbation
            gen = torch.Generator(device=p.device)
            gen.manual_seed(int(seed))
            noise = torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=gen)
            p.data.add_(float(sigma) * noise)
            del noise
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return True
    
    def reset_to_base_weights(self):
        """Reset model weights to stored base weights."""
        if not hasattr(self, '_base_weights'):
            raise RuntimeError("Must call store_base_weights first")
        for name, p in self.model_runner.model.named_parameters():
            p.data.copy_(self._base_weights[name])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return True
    
    def clear_base_weights(self):
        """Free memory used by stored base weights."""
        if hasattr(self, '_base_weights'):
            del self._base_weights
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True
    
    def apply_averaged_perturbations(self, seeds_sigmas, weights=None):
        """
        Apply the weighted average of multiple perturbations from base weights.
        This creates a single weight-averaged model from K perturbed models.
        
        Args:
            seeds_sigmas: List of (seed, sigma) tuples
            weights: Optional list of weights for each perturbation (default: equal weights)
        
        The averaged model is: W_base + sum(w_i * sigma_i * noise_i) / sum(w_i)
        """
        if not hasattr(self, '_base_weights'):
            raise RuntimeError("Must call store_base_weights first")
        
        K = len(seeds_sigmas)
        if weights is None:
            weights = [1.0 / K] * K  # Equal weights, normalized
        else:
            # Normalize weights
            total = sum(weights)
            weights = [w / total for w in weights]
        
        for name, p in self.model_runner.model.named_parameters():
            # Start with base weights
            p.data.copy_(self._base_weights[name])
            
            # Accumulate weighted perturbations in float32 for precision
            perturbation = torch.zeros_like(p.data, dtype=torch.float32)
            
            for (seed, sigma), weight in zip(seeds_sigmas, weights):
                gen = torch.Generator(device=p.device)
                gen.manual_seed(int(seed))
                noise = torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=gen)
                perturbation.add_(weight * float(sigma) * noise.to(torch.float32))
                del noise
            
            # Apply averaged perturbation
            p.data.add_(perturbation.to(p.dtype))
            del perturbation
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return True
    
    def get_logits_for_prompt(self, input_ids_list):
        """
        Get logits for the last token position for a batch of prompts.
        Returns logits as CPU tensors for ensemble averaging.
        
        Args:
            input_ids_list: List of input_ids (each is a list of token ids)
        
        Returns:
            List of logits tensors (vocab_size,) for each prompt
        """
        model = self.model_runner.model
        model.eval()
        
        results = []
        with torch.no_grad():
            for input_ids in input_ids_list:
                seq_len = len(input_ids)
                # vLLM V1 expects flattened tensors (not batched)
                # input_ids: (seq_len,), positions: (seq_len,)
                ids_tensor = torch.tensor(input_ids, dtype=torch.long, device=self.device)
                positions = torch.arange(seq_len, dtype=torch.long, device=self.device)
                
                # Forward pass - get logits
                # vLLM v0.11+ requires forward context
                if set_forward_context is not None and hasattr(self.model_runner, "vllm_config"):
                    with set_forward_context(attn_metadata=None, 
                                           vllm_config=self.model_runner.vllm_config):
                        outputs = model(input_ids=ids_tensor, positions=positions)
                else:
                    # Fallback for older vLLM versions
                    if 'positions' in inspect.signature(model.forward).parameters:
                        outputs = model(input_ids=ids_tensor, positions=positions)
                    else:
                        outputs = model(input_ids=ids_tensor.unsqueeze(0))
                
                # Get logits for the last position
                # outputs may have .logits attribute or be the logits tensor directly
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                # vLLM V1: logits shape is (seq_len, vocab_size) for flattened input
                # or (batch, seq_len, vocab_size) for batched input
                if logits.ndim == 2:
                    # Flattened: (seq_len, vocab_size)
                    last_logits = logits[-1, :].cpu()
                else:
                    # Batched: (batch, seq_len, vocab_size)
                    last_logits = logits[0, -1, :].cpu()
                results.append(last_logits)
                
                del ids_tensor, positions, outputs, logits
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.cuda.empty_cache()
        
        return results
    
    def generate_with_logits_callback(self, input_ids, max_new_tokens, temperature=1.0):
        """
        Generate tokens step by step and return the logits at each step.
        This is for debugging/analysis - actual ensemble should use get_logits_for_prompt.
        
        Returns: (generated_ids, list_of_logits_at_each_step)
        """
        model = self.model_runner.model
        model.eval()
        
        current_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        all_logits = []
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = model(input_ids=current_ids)
                last_logits = outputs.logits[0, -1, :]
                all_logits.append(last_logits.cpu())
                
                # Sample next token
                if temperature > 0:
                    probs = torch.softmax(last_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = last_logits.argmax(dim=-1, keepdim=True)
                
                current_ids = torch.cat([current_ids, next_token.unsqueeze(0)], dim=-1)
                
                del outputs
        
        generated = current_ids[0].cpu().tolist()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        torch.cuda.empty_cache()
        
        return generated, all_logits


# =====================================================================================
# Structured ES: subspace-restricted perturbations + fp32 coefficient master weights
# =====================================================================================
#
# All four experiment variants share one abstraction: the perturbed weight is
#
#       W = W_base + P(C)                     C = trainable coefficients (fp32)
#
# and ES perturbs / updates ``C`` instead of ``W``.  ``P`` is a fixed linear map:
#
#   dense    P(C) = C                          C = every parameter (paper baseline)
#   zoact    P(C) = C @ V_r                    V_r = top-r right singular vectors of
#                                              the layer's input activations
#                                              (ZO-Act, arXiv:2607.01125; W_eff = W + V_r B)
#   insparse P(C)[:, idx] = C                  idx = top-k input channels by activation RMS
#   fura     P(C)[:, blk_j] = A_j @ C_j        W_j = A_j R_j is the full-rank BTT
#                                              (output_one_block) factorization of input
#                                              block j, A_j = U_j diag(S_j); only the small
#                                              core R is perturbed (train_position=small,
#                                              s_merged_to=keep_frozen)
#
# Keeping C in fp32 matters: one ES step moves a coefficient by ~alpha/sqrt(N) ~ 1e-4,
# which is at or below one bf16 ULP of a typical LLM weight (~8e-5 at |w|~0.02).  With a
# bf16-only accumulator most of the update is silently rounded away, and the structured
# modes -- whose weight-space footprint is spread over a low-dimensional subspace -- lose
# it entirely.  W itself is always written back as bf16 for the vLLM forward, and
# perturb/restore are exact (both recompute W from base+C rather than add/subtract).

_ES_SKIP_SUBSTR = ("embed_tokens", "lm_head")
# vLLM fuses these; the fused matrix shares its input with the HF module we calibrated.
_ES_FUSED_ALIAS = (("qkv_proj", "q_proj"), ("gate_up_proj", "gate_proj"))


def _es_calib_key(vllm_param_name):
    """`model.layers.0.self_attn.qkv_proj.weight` -> `model.layers.0.self_attn.q_proj`."""
    name = vllm_param_name
    if name.endswith(".weight"):
        name = name[: -len(".weight")]
    for fused, src in _ES_FUSED_ALIAS:
        if name.endswith(fused):
            return name[: -len(fused)] + src
    return name


def _es_target(st):
    """The tensor ES actually optimises for a given mode."""
    if st["kind"] == "dense":
        return st["master"]
    if st["kind"] in ("iso", "isobtt"):
        return st["state"]
    return st["coef"]


def _es_closest_factor_pair(n):
    """Factor `n` into (p, q) with p*q == n and p as close to sqrt(n) as possible.

    Matches `btt_layer._closest_factor_pair`: returns (n_blocks, block_size).
    """
    root = int(n ** 0.5)
    for p in range(root, 0, -1):
        if n % p == 0:
            return p, n // p
    return 1, n


# =====================================================================================
# ISO-ES: fixed-spectrum ES on the bi-orthogonal orbit
# =====================================================================================
#
# ISO (arXiv:2607.19331) constrains RLVR updates to the *fixed-spectrum family*
#
#       F(W0) = { U Sigma_0 V^T : U in St(m,q), V in St(n,q) },   q = min(m,n)
#
# and trains the frames U, V with a base optimizer plus a polar retraction (their
# Eq. 15/30/31/34/35).  For ES we need something the retraction cannot give us: the
# perturbed weight must already be feasible, N times per iteration, without an SVD.
#
# Lemma (orbit form).  O(m) acts transitively on St(m,q), so
#
#       F(W0) = { C_L W0 C_R^T : C_L in O(m), C_R in O(n) }.
#
# So we never store U, Sigma_0, V at all: *any* bi-orthogonal transform of W stays in
# the ISO family, and it preserves the singular values **exactly** (not merely to
# first order as the polar retraction does).  ||W||_F is then an exact invariant --
# a free online check that the constraint still holds (see `es_get_metrics`).
#
# Perturbation.  For a skew Omega, the Cayley transform
#
#       Cay(X) = (I - X/2)^{-1} (I + X/2)
#
# is exactly orthogonal (I - X/2 is always invertible for skew X), so
#
#       W(eps) = Cay(s*Omega_L) W Cay(s*Omega_R)^T   in F(W0)  exactly.
#
# To first order W(eps) = W + s (Omega_L W - W Omega_R), and since U^T Omega_L U and
# V^T Omega_R V are skew, diag(U^T dW V) = 0 -- ISO's Prop. 4.1 / Eq. (16).
#
# Tractable generator.  A dense Omega costs O(m^3 + m^2 n) per perturbation, and ES
# pays that N=30 times per iteration; infeasible at 7B.  We use a *block-diagonal*
# skew in a seed-dependent permuted basis: Cay is then block-diagonal (a batched b x b
# solve) and applying it to W is one batched matmul costing b*|W| flops instead of
# m*|W|.  Re-drawing the permutation every seed means the group generated across
# iterations is still the full O(m) x O(n), not a fixed block-diagonal subgroup.
# Row permutations are drawn *within* each fused vLLM output segment (qkv ->
# [q,k,v], gate_up -> [gate,up]) so a rotation never mixes q/k/v or gate/up channels.
#
# Scale convention.  With Omega_j entries ~ N(0, 1/b) one has E||Omega_j F||_F ~
# ||F||_F, hence ||dW||_F / ||W||_F ~= sigma.  `sigma` is therefore the *relative
# weight-space displacement*, directly comparable to the ||dW||/||W|| column measured
# for the dense / zoact / insparse / fura modes.  Both sides share sigma/sqrt(2).
#
# Update.  The ES estimator lives in the Lie algebra, Omega_bar = (1/N) sum_n Z_n
# Omega_n, and the update is W <- Cay(alpha*Omega_bar) W Cay(alpha*Omega_bar_R)^T.
# Because each seed uses its own permuted block basis, we realise it as the ordered
# product of the N individual Cayley factors at scale (alpha/N) Z_n: exactly
# orthogonal (a product of orthogonals) and equal to the single Cayley up to
# O(alpha^2/N) ~ 2e-5 relative, far below the ES noise floor.  The per-iteration
# relative motion is alpha/sqrt(N), which at alpha = sigma/2 matches the dense-ES
# baseline exactly.
#
# `isobtt` applies the same construction to the *block-wise* SVD already used by the
# `fura` mode.  Per input block j, W[:, blk_j] = A_j R_j with A_j = U_j diag(S_j)
# frozen (bf16) and R_j = Vh_j in O(b) trained (fp32).  Perturbing R_j <- Cay(s
# Omega_j) R_j keeps R_j orthogonal and preserves *each block's* spectrum exactly,
# because sigma(Sigma_j C_j) = sigma(Sigma_j).  Trainable state drops from |W| fp32
# (iso) to n_blk * b^2 fp32 ~ 1.3% of the model, and the perturbation touches one
# side only.


def _iso_segments(name, out_features, hf_cfg):
    """Output-row segments of a (possibly vLLM-fused) 2-D weight."""
    base = name[: -len(".weight")] if name.endswith(".weight") else name
    if hf_cfg is not None:
        try:
            if base.endswith("qkv_proj"):
                hd = getattr(hf_cfg, "head_dim", None) or (
                    hf_cfg.hidden_size // hf_cfg.num_attention_heads
                )
                segs = [
                    hf_cfg.num_attention_heads * hd,
                    hf_cfg.num_key_value_heads * hd,
                    hf_cfg.num_key_value_heads * hd,
                ]
                if sum(segs) == out_features:
                    return segs
            if base.endswith("gate_up_proj") and 2 * hf_cfg.intermediate_size == out_features:
                return [hf_cfg.intermediate_size] * 2
        except Exception:
            pass
    return [int(out_features)]


def _iso_block_size(dims, b_req):
    """Largest b <= b_req dividing every dim (so no block straddles a segment)."""
    import math

    g = 0
    for d in dims:
        g = math.gcd(g, int(d))
    b = min(int(b_req), g)
    while b > 1 and g % b:
        b -= 1
    return b


def _iso_skew(n_blk, b, gen, device):
    """Block-diagonal skew generator, entries ~ N(0, 1/b) so ||Omega F|| ~ ||F||."""
    e = torch.randn(n_blk, b, b, dtype=torch.float32, device=device, generator=gen)
    return (e - e.transpose(1, 2)).mul_((2.0 * b) ** -0.5)


def _iso_cayley(omega, scale):
    """Cay(scale * Omega) = (I - X/2)^-1 (I + X/2), exactly orthogonal for skew Omega."""
    n_blk, b, _ = omega.shape
    a = omega * (0.5 * float(scale))
    eye = torch.eye(b, dtype=omega.dtype, device=omega.device).expand(n_blk, b, b)
    return torch.linalg.solve(eye - a, eye + a)


class StructuredESMixin:
    """Mixin adding subspace-restricted ES perturbation to the vLLM worker."""

    # ---------------------------------------------------------------- setup
    def init_es_state(self, mode, cfg=None):
        cfg = dict(cfg or {})
        self._es_mode = mode
        self._es_cfg = cfg
        self._es = {}

        if mode == "off":
            return {"mode": mode}

        calib = None
        if mode in ("zoact", "insparse"):
            blob = torch.load(cfg["calib_path"], map_location="cpu", weights_only=False)
            calib = blob["layers"]

        hf_cfg = None
        for holder in ("vllm_config", "model_config"):
            obj = getattr(self.model_runner, holder, None)
            hf_cfg = getattr(getattr(obj, "model_config", obj), "hf_config", None)
            if hf_cfg is not None:
                break

        model = self.model_runner.model
        n_coef = 0
        n_base = 0
        n_manifold = 0
        lid = 0
        for name, p in model.named_parameters():
            if mode == "dense":
                # Paper baseline: every parameter is perturbed, master copy in fp32.
                self._es[name] = {"kind": "dense", "master": p.data.detach().clone().float()}
                n_coef += p.numel()
                continue

            if p.ndim != 2 or any(s in name for s in _ES_SKIP_SUBSTR):
                continue  # structured modes touch linear weights only (ZO-Act convention)
            out_f, in_f = p.shape

            if mode == "zoact":
                v = calib[_es_calib_key(name)]["v"].to(p.device, torch.float32)  # (r, in)
                r = int(cfg.get("rank", v.shape[0]))
                v = v[:r].contiguous()
                self._es[name] = {
                    "kind": "zoact",
                    "base": p.data.detach().clone(),
                    "v": v,
                    "coef": torch.zeros(out_f, r, dtype=torch.float32, device=p.device),
                }
                n_coef += out_f * r
                n_base += p.numel()

            elif mode == "insparse":
                rms = calib[_es_calib_key(name)]["act_rms"].to(p.device, torch.float32)
                k = max(1, int(round(float(cfg.get("density", 0.01)) * in_f)))
                idx = torch.topk(rms, k).indices.sort().values.contiguous()
                self._es[name] = {
                    "kind": "insparse",
                    "idx": idx,
                    "base_cols": p.data[:, idx].detach().clone(),
                    "coef": torch.zeros(out_f, k, dtype=torch.float32, device=p.device),
                }
                n_coef += out_f * k
                n_base += out_f * k

            elif mode == "iso":
                st = self._iso_init(name, p, cfg, hf_cfg, lid)
                if st is None:
                    print(f"[ES] iso: skipping {name} (no usable block size)", flush=True)
                    continue
                self._es[name] = st
                lid += 1
                n_coef += p.numel()  # fp32 master W_t
                n_manifold += (out_f * (st["bl"] - 1) + in_f * (st["br"] - 1)) // 2

            elif mode == "isobtt":
                st = self._iso_init_btt(name, p, cfg, lid)
                if st is None:
                    continue
                self._es[name] = st
                lid += 1
                n_coef += st["state"].numel()
                n_base += st["A"].numel()
                n_manifold += st["n_blk"] * st["b"] * (st["b"] - 1) // 2
                # Overwrite W with its exact BTT reconstruction so step 0 is consistent.
                self._iso_write(p, st, None, 0.0)

            elif mode == "fura":
                n_blk, b = _es_closest_factor_pair(in_f)
                if cfg.get("swap_blocks", False):
                    n_blk, b = b, n_blk
                w = p.data.reshape(out_f, n_blk, b).permute(1, 0, 2).float()  # (n, out, b)
                U, S, Vh = torch.linalg.svd(w, full_matrices=False)  # k = min(out, b) = b
                A = (U * S.unsqueeze(1)).to(p.dtype).contiguous()  # (n, out, b)
                R = Vh.float().contiguous()  # (n, b, b)
                del w, U, S, Vh
                self._es[name] = {
                    "kind": "fura",
                    "A": A,
                    "R0": R,
                    "coef": torch.zeros_like(R),
                    "n_blk": n_blk,
                    "b": b,
                }
                n_coef += R.numel()
                n_base += A.numel()
                # Overwrite W with its exact BTT reconstruction so step 0 is consistent.
                self._es_write(name, p, self._es[name], None, 0.0)
            else:
                raise ValueError(f"unknown ES perturb mode: {mode}")

        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        info = {
            "mode": mode,
            "layers": len(self._es),
            "coef_params": int(n_coef),
            "base_params": int(n_base),
        }
        if n_manifold:
            info["manifold_dim"] = int(n_manifold)
        if mode == "iso":
            fused = sorted({f"{n.split('.')[-2]}{st['segs']}"
                            for n, st in self._es.items() if len(st["segs"]) > 1})
            info["fused_segments"] = fused or "NONE-DETECTED"
            info["block"] = sorted({(st["bl"], st["br"]) for st in self._es.values()})
        print(f"[ES] init_es_state {info}", flush=True)
        return info


    # ---------------------------------------------------------------- ISO modes
    def _iso_init(self, name, p, cfg, hf_cfg, lid):
        """State for `iso` (full-matrix bi-orthogonal orbit)."""
        m, n = p.shape
        segs = _iso_segments(name, m, hf_cfg)
        bl = _iso_block_size(segs, cfg.get("iso_block_size", 128))
        br = _iso_block_size([n], cfg.get("iso_block_size", 128))
        if bl < 2 or br < 2:
            return None
        return {
            "kind": "iso",
            "lid": lid,
            "state": p.data.detach().clone().float(),  # fp32 master W_t, always in F(W0)
            "segs": segs,
            "bl": bl,
            "br": br,
            "perm": bool(cfg.get("iso_perm", True)),
            "fro0": float(p.data.detach().float().norm()),
        }

    def _iso_init_btt(self, name, p, cfg, lid):
        """State for `isobtt` (block-wise SVD, frozen per-block spectrum)."""
        out_f, in_f = p.shape
        n_blk, b = _es_closest_factor_pair(in_f)
        if cfg.get("swap_blocks", False):
            n_blk, b = b, n_blk
        if b < 2:
            return None
        w = p.data.reshape(out_f, n_blk, b).permute(1, 0, 2).float()  # (n, out, b)
        U, S, Vh = torch.linalg.svd(w, full_matrices=False)  # k = min(out, b) = b
        A = (U * S.unsqueeze(1)).to(p.dtype).contiguous()  # (n, out, b), frozen
        R = Vh.float().contiguous()  # (n, b, b), orthogonal, trained
        del w, U, S, Vh
        return {
            "kind": "isobtt",
            "lid": lid,
            "A": A,
            "state": R,
            "n_blk": n_blk,
            "b": b,
            "fro0": float(torch.linalg.matrix_norm(A.float()).square().sum().sqrt()),
        }

    # -- noise -------------------------------------------------------------------
    def _iso_gen(self, st, seed):
        dev = st["state"].device
        gen = torch.Generator(device=dev)
        # Decorrelate layers: the legacy `_es_noise` reseeds with the bare seed, so
        # every layer with the same shape draws *identical* noise (all 28 q_proj get
        # one direction).  ISO mixes the layer id in.
        gen.manual_seed(int((int(seed) * 1000003 + 7 * st["lid"] + 11) & 0x7FFFFFFF))
        return gen, dev

    def _iso_noise(self, st, seed, scale):
        """(permL, C_L, permR, C_R) for `iso`; both sides share scale/sqrt(2)."""
        gen, dev = self._iso_gen(st, seed)
        m = st["state"].shape[0]
        n = st["state"].shape[1]
        s = float(scale) * (0.5 ** 0.5)
        if st["perm"]:
            parts, off = [], 0
            for seg in st["segs"]:
                parts.append(torch.randperm(seg, generator=gen, device=dev) + off)
                off += seg
            perm_l = parts[0] if len(parts) == 1 else torch.cat(parts)
            perm_r = torch.randperm(n, generator=gen, device=dev)
        else:
            perm_l = torch.arange(m, device=dev)
            perm_r = torch.arange(n, device=dev)
        c_l = _iso_cayley(_iso_skew(m // st["bl"], st["bl"], gen, dev), s)
        c_r = _iso_cayley(_iso_skew(n // st["br"], st["br"], gen, dev), s)
        return perm_l, c_l, perm_r, c_r

    def _iso_noise_btt(self, st, seed, scale):
        gen, dev = self._iso_gen(st, seed)
        return _iso_cayley(_iso_skew(st["n_blk"], st["b"], gen, dev), float(scale))

    # -- kernels -----------------------------------------------------------------
    @staticmethod
    def _iso_left(x, perm, c):
        """Cay(Omega_L) @ X, blocks over (permuted) rows."""
        m, n = x.shape
        n_blk, b, _ = c.shape
        y = torch.bmm(c, x.index_select(0, perm).view(n_blk, b, n)).view(m, n)
        out = torch.empty_like(x)
        out.index_copy_(0, perm, y)
        del y
        return out

    @staticmethod
    def _iso_right(x, perm, c):
        """X @ Cay(Omega_R)^T, blocks over (permuted) columns."""
        m, n = x.shape
        n_blk, b, _ = c.shape
        xp = x.index_select(1, perm).view(m, n_blk, b).transpose(0, 1).contiguous()
        y = torch.bmm(xp, c.transpose(1, 2)).transpose(0, 1).reshape(m, n)
        del xp
        out = torch.empty_like(x)
        out.index_copy_(1, perm, y)
        del y
        return out

    def _iso_rotate(self, st, x, seed, scale):
        """Apply one seed's exactly-orthogonal rotation to the stored state."""
        if st["kind"] == "iso":
            perm_l, c_l, perm_r, c_r = self._iso_noise(st, seed, scale)
            y = self._iso_left(x, perm_l, c_l)
            del c_l, perm_l
            out = self._iso_right(y, perm_r, c_r)
            del y, c_r, perm_r
            return out
        return torch.bmm(self._iso_noise_btt(st, seed, scale), x)  # R <- Cay @ R

    def _iso_render(self, st, x, p):
        """Materialise the bf16 vLLM weight from the (rotated) state."""
        if st["kind"] == "iso":
            p.data.copy_(x)
            return
        prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False  # TF32 mantissa < ES step size
        try:
            w = torch.bmm(st["A"].float(), x)  # (n_blk, out, b)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev
        p.data.copy_(w.permute(1, 0, 2).reshape(p.shape))
        del w

    def _iso_write(self, p, st, seed, scale):
        """W = rot(state) if seed is not None else W = state.  Never mutates state."""
        x = st["state"]
        if seed is not None:
            x = self._iso_rotate(st, x, seed, scale)
        self._iso_render(st, x, p)
        if seed is not None:
            del x

    def _iso_commit(self, p, st, seeds, scales):
        """state <- (prod_n Cay(scale_n Omega_n)) state ; then rewrite W."""
        x = st["state"]
        for seed, sc in zip(seeds, scales):
            if sc == 0.0:
                continue
            y = self._iso_rotate(st, x, seed, sc)
            if x is not st["state"]:
                del x
            x = y
        if x is not st["state"]:
            st["state"].copy_(x)
            del x
        drift = self._iso_recondition(st)
        self._iso_render(st, st["state"], p)
        return drift

    def _iso_recondition(self, st):
        """Remove fp32 accumulation drift; return the violation that was there.

        Repeated fp32 Cayley applications leave a per-matrix **isotropic gain**: every
        singular value scales by the same factor.  Measured at N=30, b=128: +5.9e-6 per
        iteration, compounding *linearly* (log-log slope 1.03), while the *shape* of the
        spectrum is preserved to 1.7e-7 over 40 iterations.  In fp64 the drift is absent
        (1e-16), so it is pure round-off -- but it is biased, not a random walk, so at 10k
        iterations it would reach 3.4e-2.  Because it is isotropic, one scalar per matrix
        removes it exactly (2.4e-4 -> 1.7e-7 measured).
        """
        if st["kind"] == "iso":
            fro = float(st["state"].norm())
            if fro <= 0.0:
                return 0.0
            st["state"].mul_(st["fro0"] / fro)
            return abs(fro / st["fro0"] - 1.0)
        # isobtt: the trained core must stay in O(b).  One Newton-Schulz step
        # R <- R(1.5 I - 0.5 R^T R) drives ||R^T R - I|| = E to -0.75 E^2, i.e. 1e-5 -> 1e-10.
        r = st["state"]
        g = torch.bmm(r.transpose(1, 2), r)
        err = float((g - torch.eye(g.shape[-1], device=g.device, dtype=g.dtype)).abs().max())
        r.copy_(torch.baddbmm(r, r, g, beta=1.5, alpha=-0.5))
        del g
        return err

    # ---------------------------------------------------------------- kernels
    def _es_noise(self, st, p, seed):
        gen = torch.Generator(device=p.device)
        gen.manual_seed(int(seed))
        kind = st["kind"]
        if kind == "dense":
            shape = p.shape
        elif kind == "zoact":
            shape = st["coef"].shape
        elif kind == "insparse":
            shape = st["coef"].shape
        else:  # fura
            shape = st["coef"].shape
        return torch.randn(shape, dtype=torch.float32, device=p.device, generator=gen)

    def _es_write(self, name, p, st, noise, scale):
        """Write W = W_base + P(coef + scale*noise) into the live vLLM parameter."""
        kind = st["kind"]
        if kind == "dense":
            m = st["master"]
            p.data.copy_(m if noise is None else torch.add(m, noise, alpha=scale))
            return
        if kind == "zoact":
            c = st["coef"] if noise is None else torch.add(st["coef"], noise, alpha=scale)
            delta = c @ st["v"]  # (out, r) @ (r, in)
            delta.add_(st["base"])
            p.data.copy_(delta)
            return
        if kind == "insparse":
            c = st["coef"] if noise is None else torch.add(st["coef"], noise, alpha=scale)
            p.data[:, st["idx"]] = (c + st["base_cols"]).to(p.dtype)
            return
        # fura: W[:, blk_j] = A_j @ (R0_j + coef_j + scale*noise_j)
        R = st["R0"] + st["coef"] if noise is None else torch.add(
            st["R0"] + st["coef"], noise, alpha=scale
        )
        prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False  # TF32 mantissa < ES step size
        try:
            w = torch.bmm(st["A"].float(), R)  # (n, out, b)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev
        p.data.copy_(w.permute(1, 0, 2).reshape(p.shape))
        del w, R

    # ---------------------------------------------------------------- ES API
    def es_perturb(self, seed, scale, negate=False):
        sign = -1.0 if negate else 1.0
        for name, p in self.model_runner.model.named_parameters():
            st = self._es.get(name)
            if st is None:
                continue
            if st["kind"] in ("iso", "isobtt"):
                self._iso_write(p, st, seed, sign * float(scale))
                continue
            noise = self._es_noise(st, p, seed)
            self._es_write(name, p, st, noise, sign * float(scale))
            del noise
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return True

    def es_restore(self, *_args, **_kwargs):
        for name, p in self.model_runner.model.named_parameters():
            st = self._es.get(name)
            if st is None:
                continue
            if st["kind"] in ("iso", "isobtt"):
                self._iso_write(p, st, None, 0.0)
                continue
            self._es_write(name, p, st, None, 0.0)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return True

    def es_update(self, seeds, coeffs, alpha, population_size):
        """coef <- coef + (alpha/N) * sum_i z_i * eps_i ; then rewrite W."""
        step = float(alpha) / float(population_size)
        drift = 0.0
        for name, p in self.model_runner.model.named_parameters():
            st = self._es.get(name)
            if st is None:
                continue
            if st["kind"] in ("iso", "isobtt"):
                # ES gradient lives in the Lie algebra: Omega_bar = (1/N) sum_n Z_n Omega_n.
                # Realised as the ordered product of the N Cayley factors, which is
                # exactly orthogonal and equals Cay(alpha*Omega_bar) to O(alpha^2/N).
                drift = max(drift, self._iso_commit(
                    p, st, seeds, [step * float(c) for c in coeffs]))
                torch.cuda.empty_cache()
                continue
            tgt = st["master"] if st["kind"] == "dense" else st["coef"]
            acc = torch.zeros_like(tgt)
            for i, seed in enumerate(seeds):
                noise = self._es_noise(st, p, seed)
                acc.add_(noise, alpha=float(coeffs[i]))
                del noise
            tgt.add_(acc, alpha=step)
            del acc
            self._es_write(name, p, st, None, 0.0)
            torch.cuda.empty_cache()
        torch.cuda.synchronize()
        self._es_metrics = {"iso/frob_drift": drift} if drift else {}
        return True

    def es_get_metrics(self):
        """Online proof that the fixed-spectrum constraint still holds.

        `iso/frob_drift` is the worst-layer constraint violation *seen at the last update*,
        i.e. before `_iso_recondition` removed it -- ||W||_F is an exact invariant of the
        bi-orthogonal orbit, so it measures exactly how far the spectrum had moved.
        `iso/orth_err` (isobtt) is the residual *after* the Newton-Schulz step.
        """
        m = dict(getattr(self, "_es_metrics", {}) or {})
        worst = 0.0
        for st in self._es.values():
            if st["kind"] != "isobtt":
                continue
            r = st["state"]
            eye = torch.eye(r.shape[-1], device=r.device, dtype=r.dtype)
            worst = max(worst, float((torch.bmm(r, r.transpose(1, 2)) - eye).abs().max()))
        if worst:
            m["iso/orth_err"] = worst
        return m

    def es_broadcast(self, src_rank):
        """Sync the coefficient state (not just W) across engines."""
        for name, _p in self.model_runner.model.named_parameters():
            st = self._es.get(name)
            if st is None:
                continue
            tgt = _es_target(st)
            self.inter_pg.broadcast(tgt, src=int(src_rank), stream=torch.cuda.current_stream())
        torch.cuda.synchronize()
        return True

    def es_save_coef(self, filepath):
        blob = {"mode": self._es_mode, "cfg": self._es_cfg, "layers": {}}
        for name, st in self._es.items():
            tgt = _es_target(st)
            big = st["kind"] in ("dense", "iso")  # full-size masters are stored bf16
            blob["layers"][name] = tgt.detach().to("cpu", torch.bfloat16 if big else torch.float32)
        torch.save(blob, filepath)
        gc.collect()
        torch.cuda.empty_cache()
        return True


# Fold the structured API into the extension class used by the ES trainer, and make the
# three trainer-facing entry points dispatch on whether structured state was installed.
for _n, _f in vars(StructuredESMixin).items():
    if not _n.startswith("__"):
        setattr(WorkerExtension, _n, _f)

WorkerExtension._es_mode = "off"
WorkerExtension._es = {}

_legacy_perturb = WorkerExtension.perturb_self_weights
_legacy_restore = WorkerExtension.restore_self_weights
_legacy_update = WorkerExtension.update_weights_from_seeds
_legacy_broadcast = WorkerExtension.broadcast_all_weights


def _perturb_self_weights(self, seed, noise_scale, negate=False):
    if getattr(self, "_es_mode", "off") == "off":
        return _legacy_perturb(self, seed, noise_scale, negate)
    return self.es_perturb(seed, noise_scale, negate)


def _restore_self_weights(self, seed, SIGMA, negate=False):
    if getattr(self, "_es_mode", "off") == "off":
        return _legacy_restore(self, seed, SIGMA, negate)
    return self.es_restore()


def _update_weights_from_seeds(self, seeds, coeffs, alpha, population_size):
    if getattr(self, "_es_mode", "off") == "off":
        return _legacy_update(self, seeds, coeffs, alpha, population_size)
    return self.es_update(seeds, coeffs, alpha, population_size)


def _broadcast_all_weights(self, src_rank):
    if getattr(self, "_es_mode", "off") == "off":
        return _legacy_broadcast(self, src_rank)
    return self.es_broadcast(src_rank)


WorkerExtension.perturb_self_weights = _perturb_self_weights
WorkerExtension.restore_self_weights = _restore_self_weights
WorkerExtension.update_weights_from_seeds = _update_weights_from_seeds
WorkerExtension.broadcast_all_weights = _broadcast_all_weights
