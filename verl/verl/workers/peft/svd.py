"""SVD adapter — filled in by Task 6."""
from verl.workers.peft.base import PEFTAdapter


class SVDAdapter(PEFTAdapter):
    mode = "svd"

    def apply(self, model, *, tokenizer, calib_loader_builder):
        raise NotImplementedError("SVDAdapter.apply implemented in Task 6")
