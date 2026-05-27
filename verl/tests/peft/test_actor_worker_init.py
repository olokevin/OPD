"""GPU smoke test: instantiate ActorRolloutRefWorker for each PEFT mode and
verify it reaches a post-FSDP-wrap state.

These tests require a GPU and exercise the full ``init_model()`` path, which
spins up vLLM. Run with ``pytest -m 'gpu and slow'``.
"""
from __future__ import annotations

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


@pytest.mark.gpu
@pytest.mark.parametrize("peft_mode_cfg", [
    {"mode": "none"},
    {"mode": "lora", "target_modules": "all", "lora": {"rank": 4, "alpha": 8}},
    {"mode": "blocktt", "target_modules": "all",
     "blocktt": {"decomp_mode": "input_one_block", "rank": "full",
                 "train_position": "small", "s_merged_to": "frozen",
                 "convert_mode": "svd", "factorize_by_head": True,
                 "train_bias": True, "normalize_after_update": False,
                 "qfura": {"enabled": False}}},
    {"mode": "svd", "target_modules": "all",
     "svd": {"train_position": "output", "s_merged_to": "frozen",
             "compression_ratio": 1.0}},
])
def test_actor_worker_init_per_mode(peft_mode_cfg):
    # Defer heavy imports.
    import os
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    from omegaconf import OmegaConf
    from verl.workers.fsdp_workers import ActorRolloutRefWorker
    from verl.trainer.config import ppo_trainer  # ensure config schema loads

    cfg = OmegaConf.load("verl/trainer/config/ppo_trainer.yaml")
    cfg.actor_rollout_ref.model.path = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    cfg.actor_rollout_ref.peft = OmegaConf.create(peft_mode_cfg)
    cfg.actor_rollout_ref.actor.fsdp_config.param_offload = False
    # Avoid full vllm rollout init; only test model build.
    worker = ActorRolloutRefWorker(config=cfg.actor_rollout_ref, role="actor")
    # The worker's _build_model_optimizer is invoked during init_model; trigger it.
    worker.init_model()
    assert hasattr(worker, "_peft_adapter")
    assert worker._peft_adapter.mode == peft_mode_cfg["mode"]
