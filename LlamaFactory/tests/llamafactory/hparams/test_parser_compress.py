"""Unit tests for parser-level cross-validators added by the CompressArguments work.

These tests exercise the ZeRO-3 / blocktt+svd guard that lives in
``llamafactory.hparams.parser.get_train_args`` (see ``parser.py:385-396``).

The guard fires on two signals:
  1. A deepspeed config *path* whose filename contains the substring
     ``z3`` or ``zero3`` (covers the common ``ds_z3_config.json`` naming).
  2. ``transformers.integrations.is_deepspeed_zero3_enabled()`` returning True.

We exercise signal (1) via the public ``get_train_args`` entry-point with a
config dict — this avoids file-IO and argv-mocking, sidesteps the early
distributed-mode check (which would otherwise fire before the guard), and
keeps the test hermetic (no GPU / network / dataset download needed). The
``deepspeed`` field is the string path itself; HF's training_args validates
the path exists (we create a real stub file), and our guard only inspects
the path string before HF activates ZeRO-3, so no actual deepspeed runtime
is required.
"""

import json
import os

import pytest


def _make_ds_z3_path(tmp_path) -> str:
    """Write a stub deepspeed ZeRO-3 config file at a path containing 'z3'
    and return its absolute path. The substring 'z3' in the filename is what
    the parser-level guard inspects."""
    ds_path = tmp_path / "ds_z3_config.json"
    ds_path.write_text(json.dumps({"zero_optimization": {"stage": 3}}))
    return str(ds_path)


def _base_config(tmp_path, finetuning_type: str, **extra) -> dict:
    """Build a minimal training-args dict that passes all parser checks
    BEFORE reaching the blocktt/svd ZeRO-3 guard at parser.py:385.

    Notes:
      * ``model_name_or_path`` only needs to be a non-empty string — the
        parser does not download anything at this stage.
      * ``deepspeed`` is set to a real file path so HF's TrainingArguments
        post-init does not fail on file-existence checks.
      * ``parallel_mode`` is forced to ``DISTRIBUTED`` by setting
        ``LOCAL_RANK`` in the environment (handled by the caller via
        monkeypatch); otherwise the parser raises
        "Please launch distributed training" before our guard runs.
    """
    cfg: dict = {
        "model_name_or_path": "fake/model",
        "trust_remote_code": True,
        "stage": "sft",
        "do_train": True,
        "finetuning_type": finetuning_type,
        "deepspeed": _make_ds_z3_path(tmp_path),
        "dataset": "alpaca_en_demo",
        "template": "qwen3",
        "output_dir": str(tmp_path / "out"),
        "per_device_train_batch_size": 1,
        "max_steps": 1,
        "report_to": "none",
        "bf16": True,
    }
    cfg.update(extra)
    return cfg


@pytest.fixture
def _distributed_env(monkeypatch):
    """Make HF TrainingArguments think we are in a distributed launch so that
    the ``parallel_mode == NOT_DISTRIBUTED`` early-return check at
    ``parser.py:332`` passes and execution actually reaches the guard at 385."""
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29500")
    yield


def test_blocktt_rejects_deepspeed_zero3_path_substring(tmp_path, _distributed_env):
    """A deepspeed config path whose filename contains ``z3`` is rejected
    by the parser even before HF activates ZeRO-3 at runtime."""
    from llamafactory.hparams.parser import get_train_args

    cfg = _base_config(tmp_path, finetuning_type="blocktt", train_position="small")
    with pytest.raises(ValueError, match="DeepSpeed ZeRO-3"):
        get_train_args(cfg)


def test_svd_rejects_deepspeed_zero3_path_substring(tmp_path, _distributed_env):
    """Symmetric guard for ``finetuning_type=svd``."""
    from llamafactory.hparams.parser import get_train_args

    cfg = _base_config(tmp_path, finetuning_type="svd", train_position="output")
    with pytest.raises(ValueError, match="DeepSpeed ZeRO-3"):
        get_train_args(cfg)


def test_lora_with_deepspeed_zero3_path_still_allowed(tmp_path, _distributed_env):
    """Sanity: the guard is scoped to blocktt/svd and does not regress lora.

    We just need to ensure that *if* ``get_train_args`` raises, the message
    is NOT our specific ZeRO-3 error — any other downstream failure
    (dataset lookup, missing files, etc.) is fine for the purpose of
    asserting non-regression of the guard's scope.
    """
    from llamafactory.hparams.parser import get_train_args

    cfg = _base_config(tmp_path, finetuning_type="lora")
    try:
        get_train_args(cfg)
    except ValueError as exc:
        if "DeepSpeed ZeRO-3" in str(exc):
            pytest.fail(f"unexpectedly rejected lora+ZeRO-3 by compress guard: {exc}")
    except Exception:
        # Any other error (dataset download, etc.) is fine — we only assert
        # our guard isn't tripped.
        pass
