from verl.trainer.np.layer_resolve import resolve_modules, active_layers_for_step

# Representative vLLM-real module names (fused qkv_proj / gate_up_proj).
MODULES = [
    "model.embed_tokens",
    "model.layers.0.self_attn.qkv_proj",
    "model.layers.0.self_attn.o_proj",
    "model.layers.0.mlp.gate_up_proj",
    "model.layers.0.mlp.down_proj",
    "model.layers.1.self_attn.qkv_proj",
    "model.layers.1.mlp.down_proj",
    "model.norm",
    "lm_head",
]


def test_fullmatch_single_type():
    rules = [r"model\.layers\.\d+\.mlp\.down_proj"]
    assert resolve_modules(rules, MODULES) == [
        "model.layers.0.mlp.down_proj",
        "model.layers.1.mlp.down_proj",
    ]


def test_fullmatch_specific_layer():
    rules = [r"model\.layers\.0\.self_attn\.qkv_proj"]
    assert resolve_modules(rules, MODULES) == ["model.layers.0.self_attn.qkv_proj"]


def test_partial_substring_does_not_match():
    # fullmatch semantics: a rule must match the WHOLE name
    rules = [r"qkv_proj"]
    assert resolve_modules(rules, MODULES) == []


def test_multiple_rules_union_dedup_in_module_order():
    rules = [r"model\.layers\.\d+\.mlp\.down_proj", r"model\.layers\.0\.self_attn\.o_proj"]
    assert resolve_modules(rules, MODULES) == [
        "model.layers.0.self_attn.o_proj",
        "model.layers.0.mlp.down_proj",
        "model.layers.1.mlp.down_proj",
    ]


def test_no_match_raises():
    # HF-style name that never exists in vLLM -> empty -> caller must fail loudly
    import pytest
    with pytest.raises(ValueError, match="matched no modules"):
        resolve_modules([r"model\.layers\.\d+\.mlp\.up_proj"], MODULES, error_if_empty=True)


def test_layer_schedule_layerwise_roundrobin():
    matched = ["A", "B", "C"]
    assert active_layers_for_step(matched, step=0, en_layerwise=True) == ["A"]
    assert active_layers_for_step(matched, step=1, en_layerwise=True) == ["B"]
    assert active_layers_for_step(matched, step=3, en_layerwise=True) == ["A"]  # wraps


def test_layer_schedule_all_at_once():
    matched = ["A", "B", "C"]
    assert active_layers_for_step(matched, step=5, en_layerwise=False) == ["A", "B", "C"]


def test_all_layer_mode_returns_all_in_forward_order():
    matched = ["model.layers.2.mlp.down_proj", "model.layers.10.mlp.down_proj"]
    # en_layerwise=False already returns all matched, in the SAME order (NOT sorted)
    out = active_layers_for_step(matched, step=0, en_layerwise=False)
    assert out == matched  # forward/resolve order preserved; layers.2 before layers.10
