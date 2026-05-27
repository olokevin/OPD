"""LoRA adapter — filled in by Task 3."""
from verl.workers.peft.base import PEFTAdapter


class LoRAAdapter(PEFTAdapter):
    mode = "lora"

    def apply(self, model, *, tokenizer, calib_loader_builder):
        raise NotImplementedError("LoRAAdapter.apply implemented in Task 3")
