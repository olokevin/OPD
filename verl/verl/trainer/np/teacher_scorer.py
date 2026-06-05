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
        """Score per-step candidate distributions into per-token reverse-KL losses.

        Two accepted input formats (auto-detected from candidate_logits[0]):

          1. LEGACY full-vocab: list over steps of FULL-vocab [1+n_sample, vocab]
             log-prob-able CPU tensors. Kept verbatim below; also the parity
             oracle for the new path.
          2. C1 TOP-K TUPLE: list over steps of (topk_logp [1+N, k], ids [k]),
             where `topk_logp` are the student rows' log-probs (already
             log_softmax'd over the FULL vocab, then sliced) over the stored
             student top-`k` `ids`, and `ids` are those student top-`k` vocab
             ids in DESCENDING student-logp order (torch.topk default). The
             stored window width k (topk_store_k, default 512) is wide enough
             that union/intersection still see teacher ids inside it.

        Returns (L_q_per_step, L_clean_per_step):
          L_q_per_step[t]:   [n_sample] reverse-KL of each perturbed candidate vs teacher
          L_clean_per_step[t]: float reverse-KL of the clean (row 0) candidate vs teacher

        Honors self.top_k_strategy when selecting the token set the KL is scored over:
          - only_tch: teacher's prompt_logprobs ids (legacy default).
          - only_stu: student-clean-row's top-k vocab ids.
          - intersection: ids in BOTH sets.
          - union: ids in EITHER set.
          - union-intersection: union, but tokens missing from one side use -inf logp
                                so the KL focuses on shared mass (same as union here,
                                kept as an alias since the spec's "richest form" is
                                a v2 refinement).
        Teacher logprobs for ids the teacher didn't surface are filled with the
        MIN of the teacher's surfaced logp at the same position (a calibrated
        lower bound that doesn't explode the KL — `-1e30` would blow up
        L_clean to ~1e25 since hundreds of student top-k tokens are off the
        teacher's k=256 surface). The MIN is conservative: tokens off the
        teacher's top-k have probability strictly less than the minimum it
        surfaced, but using the minimum prevents log-prob domination.
        See spec §3 and §4 (top_k_strategy).

        New-path approximation (TOP-K TUPLE only): for `union` (and
        `union-intersection`), a teacher id may fall OUTSIDE the stored student
        window — its true student log-prob is then unavailable (the full-vocab
        path had it). We use the per-row MIN of the stored student top-k logps
        as that id's student log-prob (a calibrated floor, symmetric to how
        teacher ids missing from the teacher surface are filled with the
        teacher MIN). For `only_stu`/`intersection`/`only_tch` no such id can
        occur (the scored set is a subset of the stored window, or the teacher
        set is scored verbatim), so those strategies are EXACT vs the legacy
        path. only_stu — the lossless production default — is exact.
        """
        if not candidate_logits:
            return [], []
        if isinstance(candidate_logits[0], tuple):
            return self._score_rollout_topk(prefix_token_ids, candidate_logits)

        L_q_per_step, L_clean_per_step = [], []
        teacher_logp_by_pos, teacher_ids_by_pos = self._teacher_topk_logprobs(
            prefix_token_ids, len(candidate_logits))
        for t, cl in enumerate(candidate_logits):
            s_full_logp = torch.log_softmax(cl.float(), dim=-1)  # [1+n_sample, vocab]
            t_ids = teacher_ids_by_pos[t]
            t_logp = teacher_logp_by_pos[t]
            fallback = float(t_logp.min().item()) if t_logp.numel() else -50.0
            ids, t_logp_aligned = self._select_ids(s_full_logp[0], t_ids, t_logp, fallback)
            s_logp = s_full_logp[:, ids]                   # [1+n_sample, k']
            L_clean_per_step.append(
                float(reverse_kl_topk(s_logp[0], t_logp_aligned, self.weight_mode)))
            L_q_per_step.append(torch.stack([
                reverse_kl_topk(s_logp[1 + q], t_logp_aligned, self.weight_mode)
                for q in range(s_logp.shape[0] - 1)
            ]))
        return L_q_per_step, L_clean_per_step

    def _score_rollout_topk(self, prefix_token_ids, candidate_logits):
        """New path: candidate_logits is a list of (topk_logp [1+N, k], ids [k]).

        Mirrors the legacy full-vocab loop but (a) sources the student top-k from
        the stored window and (b) computes reverse-KL for all 1+N rows in ONE
        vectorized reduce per step instead of a per-row Python loop. The reduce
        is the row-batched form of reverse_kl_topk; see score_rollout docstring
        for the union out-of-window student-logp approximation.
        """
        L_q_per_step, L_clean_per_step = [], []
        teacher_logp_by_pos, teacher_ids_by_pos = self._teacher_topk_logprobs(
            prefix_token_ids, len(candidate_logits))
        for t, (topk_logp, stored_ids) in enumerate(candidate_logits):
            # topk_logp: [1+N, k] student rows' full-vocab log-softmax over stored_ids.
            t_ids = teacher_ids_by_pos[t]
            t_logp = teacher_logp_by_pos[t]
            fallback = float(t_logp.min().item()) if t_logp.numel() else -50.0
            # Per-row student-logp floor for union ids outside the stored window.
            s_floor = topk_logp.min(dim=-1).values  # [1+N]

            # s_logp_sel: [1+N, k'] student log-probs over the selected id set;
            # t_aligned: [k'] teacher log-probs aligned to the same ids.
            s_logp_sel, t_aligned = self._select_ids_from_window(
                stored_ids, topk_logp, t_ids, t_logp, fallback, s_floor)

            losses = self._reverse_kl_rows(s_logp_sel, t_aligned)  # [1+N]
            L_clean_per_step.append(float(losses[0]))
            L_q_per_step.append(losses[1:].clone())
        return L_q_per_step, L_clean_per_step

    def _reverse_kl_rows(self, s_logp, t_logp):
        """Row-batched reverse_kl_topk: same formula, reduced over the last dim.

        s_logp: [R, k'] (per-row student log-probs over the scored set)
        t_logp: [k']    (teacher log-probs over the SAME set, shared across rows)
        Returns [R] reverse-KL, identical to calling reverse_kl_topk per row.
        """
        diff = s_logp - t_logp                       # [R, k'] (t broadcasts)
        if self.weight_mode == "student_p":
            w = s_logp.exp()                         # [R, k']
        elif self.weight_mode == "teacher_p":
            w = t_logp.exp().expand_as(diff)         # [k'] -> [R, k']
        elif self.weight_mode == "none":
            w = torch.ones_like(diff)
        else:
            raise ValueError(f"unknown reward_weight_mode: {self.weight_mode!r}")
        return (w * diff).sum(dim=-1)                 # [R]

    def _select_ids_from_window(self, stored_ids, topk_logp, t_ids, t_logp,
                                fallback, s_floor):
        """Windowed analogue of _select_ids for the C1 top-k tuple path.

        Returns (s_logp_sel [1+N, k'], t_aligned [k']) where the scored id set
        matches what _select_ids would produce, the teacher logps are aligned
        the same way (missing -> `fallback`), and the student logps are looked
        up from the stored window (missing -> per-row `s_floor`, union only).

        stored_ids are the student top-`topk_store_k` in DESCENDING student-logp
        order, so stored_ids[:top_k] IS the student top-`top_k` (== the
        torch.topk(s_full, top_k) set the legacy _select_ids computes).
        """
        n_rows = topk_logp.shape[0]
        # Map stored id -> its column in topk_logp (student log-probs per row).
        stored_list = stored_ids.tolist()
        col_of = {int(i): j for j, i in enumerate(stored_list)}
        t_map = {int(i): float(lp) for i, lp in zip(t_ids.tolist(), t_logp.tolist())}

        strat = self.top_k_strategy
        if strat == "only_tch":
            ids_list = [int(i) for i in t_ids.tolist()]
        else:
            s_top = stored_list[:self.top_k]          # student top-`top_k`
            s_set = set(s_top)
            t_set = set(int(i) for i in t_ids.tolist())
            if strat == "only_stu":
                ids_list = sorted(s_set)
            elif strat == "intersection":
                ids_list = sorted(s_set & t_set)
            elif strat in ("union", "union-intersection"):
                ids_list = sorted(s_set | t_set)
            else:
                raise ValueError(f"unknown top_k_strategy: {strat!r}")
            if not ids_list:
                # Defensive: empty intersection -> score the teacher set verbatim.
                ids_list = [int(i) for i in t_ids.tolist()]

        # Build student log-probs [1+N, k'] column-by-column, and teacher [k'].
        s_cols, t_vals = [], []
        for i in ids_list:
            j = col_of.get(i)
            if j is not None:
                s_cols.append(topk_logp[:, j])        # [1+N] exact student logp
            else:
                # union teacher id outside the stored window -> per-row floor.
                s_cols.append(s_floor)
            t_vals.append(t_map.get(i, fallback))
        s_logp_sel = torch.stack(s_cols, dim=1)       # [1+N, k']
        t_aligned = torch.tensor(t_vals, dtype=topk_logp.dtype)
        return s_logp_sel, t_aligned

    def _select_ids(self, s_clean_full_logp, t_ids, t_logp, fallback):
        """Compute the union/intersection/etc id set per self.top_k_strategy and
        return (ids[LongTensor], teacher_logp[FloatTensor]) aligned to ids.

        Teacher entries missing from t_ids are filled with the `fallback` logp
        (callers should pass the per-position min teacher logp to avoid blowing
        up the reverse-KL — see score_rollout).
        """
        strat = self.top_k_strategy
        if strat == "only_tch":
            return t_ids, t_logp

        # Student top-k from the clean-row full distribution.
        s_top = torch.topk(s_clean_full_logp, self.top_k).indices  # [top_k]
        s_set = set(s_top.tolist())
        t_set = set(t_ids.tolist())
        t_map = {int(i): float(lp) for i, lp in zip(t_ids.tolist(), t_logp.tolist())}

        if strat == "only_stu":
            ids_list = sorted(s_set)
        elif strat == "intersection":
            ids_list = sorted(s_set & t_set)
        elif strat in ("union", "union-intersection"):
            ids_list = sorted(s_set | t_set)
        else:
            raise ValueError(f"unknown top_k_strategy: {strat!r}")
        if not ids_list:
            # Defensive: empty intersection -> fall back to teacher's ids so the
            # KL is well-defined rather than degenerate.
            return t_ids, t_logp

        ids = torch.tensor(ids_list, dtype=torch.long)
        t_aligned = torch.tensor([t_map.get(int(i), fallback) for i in ids_list],
                                 dtype=t_logp.dtype)
        return ids, t_aligned

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
