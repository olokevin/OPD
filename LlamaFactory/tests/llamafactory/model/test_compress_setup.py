"""Unit tests for compress_setup lazy import and dispatch."""
import sys


class _FakeFA:
    """Minimal FinetuningArguments stand-in for unit tests."""
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def test_ensure_compress_on_path_idempotent():
    from llamafactory.model import compress_setup
    # Drop any pre-existing entry
    src_dir = compress_setup._repo_src_dir()
    while str(src_dir) in sys.path:
        sys.path.remove(str(src_dir))
    compress_setup._ensure_compress_on_path()
    assert str(src_dir) in sys.path
    # Idempotent
    n = sys.path.count(str(src_dir))
    compress_setup._ensure_compress_on_path()
    assert sys.path.count(str(src_dir)) == n


def test_repo_src_dir_resolves_to_opd_src():
    from llamafactory.model import compress_setup
    src = compress_setup._repo_src_dir()
    assert src.name == "src"
    # ".../OPD/src" → parent is the repo root, which must contain LlamaFactory
    assert (src.parent / "LlamaFactory").exists()


def test_init_compress_model_skips_when_not_trainable():
    from llamafactory.model import compress_setup
    sentinel = object()
    out = compress_setup.init_compress_model(
        config=None, model=sentinel, model_args=None,
        finetuning_args=_FakeFA(finetuning_type="blocktt", calib_mode="none"),
        is_trainable=False,
    )
    assert out is sentinel


def test_to_namespace_covers_all_compress_fields():
    import dataclasses
    from llamafactory.hparams.finetuning_args import CompressArguments
    from llamafactory.model import compress_setup

    fa = _FakeFA(
        finetuning_type="blocktt",
        # populate every CompressArguments field with a recognizable value
        **{f.name: f.default for f in dataclasses.fields(CompressArguments)},
    )
    ns = compress_setup._to_namespace(fa)

    # train_mode is added explicitly
    assert ns.train_mode == "blocktt"
    # every CompressArguments field is on the namespace
    for f in dataclasses.fields(CompressArguments):
        assert hasattr(ns, f.name), f"_to_namespace missing field {f.name}"
        assert getattr(ns, f.name) == f.default


import torch
import torch.nn as nn


def _tiny_qwen_like_model():
    """Build a minimal Qwen-shaped nn.Module with the linear submodules
    compress.integration looks for (q/k/v/o/gate/up/down)."""
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(32, 32, bias=False)
            self.k_proj = nn.Linear(32, 32, bias=False)
            self.v_proj = nn.Linear(32, 32, bias=False)
            self.o_proj = nn.Linear(32, 32, bias=False)
            self.gate_proj = nn.Linear(32, 64, bias=False)
            self.up_proj = nn.Linear(32, 64, bias=False)
            self.down_proj = nn.Linear(64, 32, bias=False)
        def forward(self, x):
            x = self.q_proj(x) + self.k_proj(x) + self.v_proj(x) + self.o_proj(x)
            return self.down_proj(self.gate_proj(x).relu() * self.up_proj(x))
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Block() for _ in range(2)])
        def forward(self, x):
            for b in self.layers:
                x = b(x)
            return x
    return Model()


def test_plain_blocktt_converts_linear_modules():
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("compress conversion requires CUDA")
    from llamafactory.model import compress_setup
    compress_setup._ensure_compress_on_path()
    from compress.integration import BTTLinear

    model = _tiny_qwen_like_model().cuda()
    fa = _FakeFA(
        finetuning_type="blocktt", calib_mode="none",
        trainable_type="all", train_position="small",
        s_merged_to="frozen", decomp_mode="input_one_block",
        blocktt_rank="full", convert_mode="svd", train_bias=True,
        blocktt_normalize_after_update=False, blocktt_factorize_by_head=True,
    )
    out = compress_setup.init_compress_model(
        config=None, model=model, model_args=None,
        finetuning_args=fa, is_trainable=True,
    )

    btt_count = sum(1 for m in out.modules() if isinstance(m, BTTLinear))
    assert btt_count > 0, "expected some Linear modules converted to BTTLinear"

    trainable = [n for n, p in out.named_parameters() if p.requires_grad]
    assert any(".btt_l" in n or ".btt_r" in n for n in trainable), trainable


def test_plain_svd_converts_linear_modules():
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("compress conversion requires CUDA")
    from llamafactory.model import compress_setup
    compress_setup._ensure_compress_on_path()
    from compress.integration import SVDCompressedLinear

    model = _tiny_qwen_like_model().cuda()
    fa = _FakeFA(
        finetuning_type="svd", calib_mode="none",
        trainable_type="all", train_position="output",
        s_merged_to="frozen", decomp_mode="input_one_block",
        blocktt_rank="full", convert_mode="svd", train_bias=True,
        blocktt_normalize_after_update=False, blocktt_factorize_by_head=True,
    )
    out = compress_setup.init_compress_model(
        config=None, model=model, model_args=None,
        finetuning_args=fa, is_trainable=True,
    )

    svd_count = sum(1 for m in out.modules() if isinstance(m, SVDCompressedLinear))
    assert svd_count > 0

    trainable = [n for n, p in out.named_parameters() if p.requires_grad]
    assert any(".U_r" in n or ".V_r" in n for n in trainable), trainable
