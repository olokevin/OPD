"""Unit tests for compress_setup lazy import and dispatch."""
import sys
import pathlib

import pytest


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
