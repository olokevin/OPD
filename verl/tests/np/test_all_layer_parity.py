import torch
from verl.workers.rollout.vllm_rollout.np_worker_extension import (
    PerturbedLinear, WorkerExtension)


def test_all_layer_u_matches_single_layer_seed():
    we = WorkerExtension.__new__(WorkerExtension)
    layers = ["L0", "L1", "L2"]
    cfg = dict(global_seed=7, sample_method="bernoulli")
    # all-layer fill
    u_all = {n: torch.zeros(4, 8) for n in layers}
    we._np_fill_u_buf_all_layers(u_all, cfg, layers, step=3, rollout=1, n_sample=4)
    # single-layer fill, one layer at a time (V1 seed path)
    for n in layers:
        u_one = torch.zeros(4, 8)
        we._np_fill_u_buf_all_layers(u_one_dict := {n: u_one}, cfg, [n],
                                     step=3, rollout=1, n_sample=4)
        assert torch.equal(u_all[n], u_one_dict[n]), f"{n} u not bit-identical"
