"""Shared pytest fixtures for llamafactory tests."""
import pytest


class _FakeFinetuningArgs:
    """Minimal FinetuningArguments stand-in for unit tests.

    Tests can populate only the fields they need; production code uses the
    real ``FinetuningArguments`` dataclass which sets every field via
    ``CompressArguments``/``LoraArguments``/etc. defaults.
    """

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


@pytest.fixture
def fake_finetuning_args():
    """Return the _FakeFinetuningArgs class so tests can call
    fake_finetuning_args(finetuning_type=..., ...). Returning the class
    (not an instance) keeps the call site explicit about which fields
    matter for the test."""
    return _FakeFinetuningArgs
