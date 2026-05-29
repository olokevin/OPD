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


class TeacherScorer:
    """Wraps a teacher vLLM engine; scores per-step candidate logits into L_t^(q).

    Given the student's prefix tokens (committed) and the per-step candidate
    next-token distributions, query the teacher for its log-probs over the OPD
    top-k set and call reverse_kl_topk per (token, sample). Returns L_t^(q).
    """

    def __init__(self, teacher_engine, top_k, top_k_strategy, teacher_temperature, weight_mode):
        self.engine = teacher_engine
        self.top_k = int(top_k)
        self.top_k_strategy = top_k_strategy
        self.teacher_temperature = float(teacher_temperature)
        self.weight_mode = weight_mode

    def score_rollout(self, prefix_token_ids, candidate_logits):
        """candidate_logits: list over steps of [1+n_sample, vocab] (CPU tensors).

        Returns (L_q_per_step, L_clean_per_step):
          L_q_per_step[t]:   [n_sample] reverse-KL of each perturbed candidate vs teacher
          L_clean_per_step[t]: float reverse-KL of the clean (row 0) candidate vs teacher
        Implementation: feed [prefix + committed token sequence] to the teacher with
        prompt_logprobs=top_k to get teacher top-k log-probs per position (vLLM
        SamplingParams.prompt_logprobs, sampling_params.py:154-164). Align teacher
        position p with student step p, gather student log-probs over the same top-k
        token ids, call reverse_kl_topk. See spec §3.
        """
        L_q_per_step, L_clean_per_step = [], []
        teacher_logp_by_pos, topk_ids_by_pos = self._teacher_topk_logprobs(
            prefix_token_ids, len(candidate_logits))
        for t, cl in enumerate(candidate_logits):
            ids = topk_ids_by_pos[t]                       # [k] teacher's top-k token ids
            t_logp = teacher_logp_by_pos[t]                # [k]
            s_full_logp = torch.log_softmax(cl.float(), dim=-1)  # [1+n_sample, vocab]
            s_logp = s_full_logp[:, ids]                   # [1+n_sample, k]
            L_clean_per_step.append(
                float(reverse_kl_topk(s_logp[0], t_logp, self.weight_mode)))
            L_q_per_step.append(torch.stack([
                reverse_kl_topk(s_logp[1 + q], t_logp, self.weight_mode)
                for q in range(s_logp.shape[0] - 1)
            ]))
        return L_q_per_step, L_clean_per_step

    def _teacher_topk_logprobs(self, prefix_token_ids, num_steps):
        """Query the teacher engine for per-position top-k log-probs over the response.

        Returns (logp_by_pos: list[Tensor[k]], ids_by_pos: list[LongTensor[k]]).
        """
        import ray
        from vllm import SamplingParams

        sp = SamplingParams(temperature=self.teacher_temperature, max_tokens=1,
                            prompt_logprobs=self.top_k)
        out = self.engine.generate.remote({"prompt_token_ids": prefix_token_ids}, sp,
                                          use_tqdm=False)
        out = ray.get(out)[0]
        # prompt_logprobs: list[dict[token_id -> Logprob]] aligned to prompt positions.
        # The response region is the last num_steps positions of prefix_token_ids.
        plp = out.prompt_logprobs[-num_steps:]
        logp_by_pos, ids_by_pos = [], []
        for d in plp:
            ids = list(d.keys())
            logp_by_pos.append(torch.tensor([d[i].logprob for i in ids]))
            ids_by_pos.append(torch.tensor(ids, dtype=torch.long))
        return logp_by_pos, ids_by_pos
