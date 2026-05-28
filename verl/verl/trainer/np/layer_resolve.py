"""Resolve perturb_rules regexes to module names and pick active layers per step.

Resolution uses re.fullmatch: a rule must match the WHOLE module name. Output
preserves the order modules appear in (model.named_modules() order), de-duped.
Names are vLLM-real (fused qkv_proj / gate_up_proj); see spec §4.
"""
import re
from typing import List


def resolve_modules(
    rules: List[str],
    module_names: List[str],
    error_if_empty: bool = False,
) -> List[str]:
    compiled = [re.compile(r) for r in rules]
    out, seen = [], set()
    for name in module_names:
        if name in seen:
            continue
        if any(c.fullmatch(name) for c in compiled):
            out.append(name)
            seen.add(name)
    if error_if_empty and not out:
        raise ValueError(
            f"perturb_rules {rules!r} matched no modules. "
            f"Note vLLM fuses qkv/gate_up: use self_attn.qkv_proj / mlp.gate_up_proj, "
            f"not q_proj / up_proj."
        )
    return out


def active_layers_for_step(matched: List[str], step: int, en_layerwise: bool) -> List[str]:
    """en_layerwise=True -> one layer per step (round-robin); False -> all matched."""
    if not matched:
        return []
    if en_layerwise:
        return [matched[step % len(matched)]]
    return list(matched)
