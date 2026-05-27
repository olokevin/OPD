"""BlockTT adapter — filled in by Task 5."""
from verl.workers.peft.base import PEFTAdapter


class BlockTTAdapter(PEFTAdapter):
    mode = "blocktt"

    def apply(self, model, *, tokenizer, calib_loader_builder):
        raise NotImplementedError("BlockTTAdapter.apply implemented in Task 5")
