"""Node-Perturbation (NP) zeroth-order trainer.

See docs/superpowers/specs/2026-05-28-np-trainer-design.md for the algorithm.
Mirrors the ES sibling trainer (verl/verl/trainer/es/) but perturbs linear-layer
*outputs* during a custom n_sample-wide decode rather than perturbing weights.
"""
