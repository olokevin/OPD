"""Zeroth-order Node-Perturbation gradient-check harness.

This package holds an *offline, autograd-enabled* validation of the NP
(node-perturbation) gradient estimator that the production trainer
(verl/verl/trainer/np/) computes under vLLM no_grad.

The NP trainer can never produce a true backprop gradient itself: its custom
n_sample-wide decode runs inside vLLM under torch.no_grad, and the perturbation
is injected at a layer's *output* node. To know whether the NP estimate points
the right way (and at the right scale), we reproduce the *exact same* estimator
math here on an eager HuggingFace model where a real loss.backward() is available
as ground truth.

See docs/superpowers/specs/2026-05-28-np-trainer-design.md  test #2 ("Gradient
cosine-sim, offline, 1 GPU").

The estimator math is imported verbatim from the production code paths:
  - verl.trainer.np.seeding            (noise_seed, draw_noise)
  - verl.trainer.np.grad_estimator     (sample_scale, accumulate_delta_w)
  - verl.workers...np_worker_extension (assemble_layer_delta)
  - verl.trainer.np.teacher_scorer     (reverse_kl_topk)
so this is a faithful check of what actually ships, not a re-implementation.
"""
