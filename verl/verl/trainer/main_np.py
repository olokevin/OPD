"""Hydra entry point for Node-Perturbation (NP) training. Mirrors main_es.py."""
import os
import socket
import tempfile
import time

import hydra
import ray
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from verl.trainer.np.ray_trainer import RayNPTrainer
from verl.utils.device import auto_set_device


@hydra.main(config_path="config", config_name="np_trainer", version_base=None)
def main(config):
    auto_set_device(config)
    run_np(config)


def run_np(config) -> None:
    from pprint import pprint
    print(f"NP Training - hostname: {socket.gethostname()}, PID: {os.getpid()}")
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    if not ray.is_initialized():
        for k in ("RAY_ADDRESS", "RAY_HEAD_IP", "RAY_GCS_SERVER_ADDRESS"):
            os.environ.pop(k, None)
        unique_dir = tempfile.mkdtemp(prefix=f"ray_np_session_{int(time.time())}_")
        ray.init(address="local", include_dashboard=False, ignore_reinit_error=True,
                 _temp_dir=unique_dir, dashboard_port=None)

    model_path = config.model.path
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=config.model.get("trust_remote_code", False))

    from verl.trainer.main_es import load_data
    train_data, eval_data = [], []
    if config.data.get("train_files"):
        train_data = load_data(config.data.train_files)
        if config.data.get("train_max_samples", -1) > 0:
            train_data = train_data[: config.data.train_max_samples]
    if config.data.get("val_files"):
        eval_data = load_data(config.data.val_files)
        if config.data.get("val_max_samples", -1) > 0:
            eval_data = eval_data[: config.data.val_max_samples]

    task_type = config.data.get("task_type", "opd_math")
    if task_type in ["countdown", "gsm8k", "math", "math500", "olympiadbench",
                     "uspto50k", "common_gen", "mbpp", "rocstories", "opd_math"]:
        from verl.trainer.np.task_utils import get_task_components
        prompt_processor, reward_fn = get_task_components(
            task_type, OmegaConf.to_container(config.data, resolve=True))
    elif task_type == "custom":
        from verl.utils.import_utils import load_extern_object
        reward_fn = (load_extern_object(config.data.reward_fn_path, config.data.reward_fn_name)
                     if config.data.get("reward_fn_path") else None)
        prompt_processor = (load_extern_object(config.data.prompt_processor_path,
                                               config.data.prompt_processor_name)
                            if config.data.get("prompt_processor_path") else None)
    else:
        raise ValueError(f"Unknown task_type: {task_type}")

    if prompt_processor and eval_data:
        for d in eval_data:
            d["context"] = prompt_processor(d, tokenizer)

    trainer = RayNPTrainer(config=config, tokenizer=tokenizer, reward_fn=reward_fn,
                           val_reward_fn=reward_fn, train_data=train_data,
                           eval_data=eval_data, prompt_processor=prompt_processor)
    trainer.init_workers(model_path)
    trainer.fit()
    ray.shutdown()


if __name__ == "__main__":
    main()
