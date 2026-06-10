"""Per-token base-noise seeding for es_token.

Invariant (inherited from NP): noise is NEVER stored or shipped across RPC
boundaries -- only integer seeds. The decode worker draws (u_t, v_t) for token t
of rollout `rid` from es_token_seed(global_seed, t, rid); the assembly
regenerates the SAME bytes from the same key ON THE SAME DEVICE/dtype
(draw_noise's CUDA and CPU generators differ, so decode and assembly both run
on the worker GPU).

Layout: ONE flat draw of width d_total = sum_l (d_out_l + d_in_l) per
(slot, token); each layer reads its (u, v) as fixed slices via the layout
table built at install time.
"""
from typing import Dict, List, Tuple

import torch

from verl.trainer.np.seeding import draw_noise, noise_seed

# Reserved layer-namespace tag so es_token seeds can never collide with NP's
# (layer, q) keys even under the same global_seed.
_ES_TOKEN_TAG = "es_token_noise"


def es_token_seed(global_seed: int, step_t: int, rollout_id: int) -> int:
    """63-bit seed for the (token t, rollout) shared base-noise draw."""
    return noise_seed(int(global_seed), int(step_t), _ES_TOKEN_TAG,
                      int(rollout_id), 0)


def draw_token_noise(global_seed: int, step_t: int, rollout_id: int,
                     d_total: int, device: torch.device, dtype: torch.dtype,
                     method: str) -> torch.Tensor:
    """[d_total] base-noise vector for one (slot, token). ONE fused draw."""
    seed = es_token_seed(global_seed, step_t, rollout_id)
    return draw_noise(seed, (int(d_total),), device, dtype, method)


def build_noise_layout(layer_dims: List[Tuple[str, int, int]]
                       ) -> Tuple[Dict[str, Tuple[int, int, int, int]], int]:
    """Assign each layer contiguous (u, v) slices of the flat noise vector.

    layer_dims: [(layer_name, d_out, d_in), ...] in a FIXED order (the resolved
    perturb_rules order) -- the layout must be identical on every worker and at
    assembly, so callers always pass the same ordered list.

    Returns ({layer: (off_u, d_out, off_v, d_in)}, d_total).
    """
    layout: Dict[str, Tuple[int, int, int, int]] = {}
    off = 0
    for name, d_out, d_in in layer_dims:
        off_u = off
        off += int(d_out)
        off_v = off
        off += int(d_in)
        layout[name] = (off_u, int(d_out), off_v, int(d_in))
    return layout, off
