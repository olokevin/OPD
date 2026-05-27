"""QLoRA adapter — filled in by Task 4."""
from verl.workers.peft.lora import LoRAAdapter


class QLoRAAdapter(LoRAAdapter):
    mode = "qlora"
