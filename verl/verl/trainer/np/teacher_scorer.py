"""Per-step teacher scoring -> per-token reverse-KL loss L_t (minimization-oriented).

This file holds two layers:
  1. reverse_kl_topk(...): pure-math kernel over a top-k token set (tested in
     test_teacher_scorer.py).
  2. TeacherScorer (added in Task 12): wraps a second vLLM engine, gathers the
     teacher's log-probs over the OPD top-k set, and calls the kernel per token.

Sign: L_t is positive reverse-KL, already minimization-oriented (lower = student
closer to teacher). If ever sourcing from dp_actor.compute_distillation_reward
(which returns -kl*w, maximization-oriented), negate before use. See spec §3.
"""
import torch


def reverse_kl_topk(
    student_logp: torch.Tensor,   # [k] student log-probs over the top-k token set
    teacher_logp: torch.Tensor,   # [k] teacher log-probs over the SAME k tokens
    weight_mode: str = "student_p",
) -> torch.Tensor:
    """Reverse KL: sum_v w_v * (log p_student - log p_teacher).

    weight_mode:
      - "student_p": w_v = softmax(student)_v  (standard reverse KL E_student[...])
      - "teacher_p": w_v = softmax(teacher)_v
      - "none":      w_v = 1 (unweighted sum of log-prob differences over the set)
    """
    diff = student_logp - teacher_logp
    if weight_mode == "student_p":
        w = student_logp.exp()
    elif weight_mode == "teacher_p":
        w = teacher_logp.exp()
    elif weight_mode == "none":
        w = torch.ones_like(diff)
    else:
        raise ValueError(f"unknown reward_weight_mode: {weight_mode!r}")
    return (w * diff).sum()
