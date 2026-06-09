import pytest
import torch
from verl.trainer.np.teacher_scorer import TeacherScorer, reverse_kl_topk


def test_reverse_kl_zero_when_identical():
    # student == teacher -> KL = 0
    logp = torch.log_softmax(torch.tensor([2.0, 1.0, 0.5, 0.0]), dim=-1)
    kl = reverse_kl_topk(student_logp=logp, teacher_logp=logp, weight_mode="none")
    assert torch.allclose(kl, torch.tensor(0.0), atol=1e-6)


def test_reverse_kl_positive_and_minimization_oriented():
    s = torch.log_softmax(torch.tensor([3.0, 0.0, 0.0]), dim=-1)   # peaked
    t = torch.log_softmax(torch.tensor([0.0, 0.0, 0.0]), dim=-1)   # uniform
    kl = reverse_kl_topk(student_logp=s, teacher_logp=t, weight_mode="student_p")
    assert kl.item() > 0.0   # reverse KL E_student[log p_s - log p_t] > 0 here


def test_reverse_kl_student_p_weighting_uses_student_probs():
    s = torch.log_softmax(torch.tensor([3.0, 0.0]), dim=-1)
    t = torch.log_softmax(torch.tensor([0.0, 0.0]), dim=-1)
    # student_p weighting weights each vocab term by softmax(student) over the k set
    kl_w = reverse_kl_topk(student_logp=s, teacher_logp=t, weight_mode="student_p")
    # equals sum_v softmax(s)_v * (s_v - t_v) == standard reverse KL
    p = s.exp()
    expected = (p * (s - t)).sum()
    assert torch.allclose(kl_w, expected, atol=1e-6)


def _make_scorer(strategy, top_k=3):
    return TeacherScorer(teacher_engine=None, top_k=top_k, top_k_strategy=strategy,
                         teacher_temperature=1.0, weight_mode="none")


def test_select_ids_only_tch_returns_teacher_ids_verbatim():
    sc = _make_scorer("only_tch", top_k=3)
    s_logp = torch.log_softmax(torch.arange(10).float(), dim=-1)  # vocab=10, peaked at 9
    t_ids = torch.tensor([9, 7, 5])
    t_logp = torch.tensor([-0.1, -2.0, -5.0])
    ids, t_aligned = sc._select_ids(s_logp, t_ids, t_logp, fallback=-1e30)
    assert torch.equal(ids, t_ids) and torch.equal(t_aligned, t_logp)


def test_select_ids_only_stu_uses_student_topk_and_fills_missing_teacher():
    sc = _make_scorer("only_stu", top_k=3)
    s_logp = torch.log_softmax(torch.arange(10).float(), dim=-1)  # top-3 = {9, 8, 7}
    t_ids = torch.tensor([9, 5, 4])         # teacher only has id 9 in student's top-3
    t_logp = torch.tensor([-0.1, -3.0, -4.0])
    ids, t_aligned = sc._select_ids(s_logp, t_ids, t_logp, fallback=-1e30)
    assert sorted(ids.tolist()) == [7, 8, 9]
    # Token 7, 8 not in teacher -> neg_inf; token 9 keeps -0.1
    aligned_map = dict(zip(ids.tolist(), t_aligned.tolist()))
    assert aligned_map[9] == pytest.approx(-0.1, abs=1e-5)
    assert aligned_map[7] == pytest.approx(-1e30) and aligned_map[8] == pytest.approx(-1e30)


def test_select_ids_intersection_keeps_only_shared_tokens():
    sc = _make_scorer("intersection", top_k=3)
    s_logp = torch.log_softmax(torch.arange(10).float(), dim=-1)  # top-3 = {9, 8, 7}
    t_ids = torch.tensor([9, 8, 5])
    t_logp = torch.tensor([-0.1, -1.0, -3.0])
    ids, t_aligned = sc._select_ids(s_logp, t_ids, t_logp, fallback=-1e30)
    assert sorted(ids.tolist()) == [8, 9]
    aligned_map = dict(zip(ids.tolist(), t_aligned.tolist()))
    assert aligned_map[9] == pytest.approx(-0.1, abs=1e-5)
    assert aligned_map[8] == pytest.approx(-1.0, abs=1e-5)


def test_select_ids_union_keeps_all_tokens():
    sc = _make_scorer("union", top_k=3)
    s_logp = torch.log_softmax(torch.arange(10).float(), dim=-1)  # top-3 = {9, 8, 7}
    t_ids = torch.tensor([5, 4])
    t_logp = torch.tensor([-2.0, -3.0])
    ids, t_aligned = sc._select_ids(s_logp, t_ids, t_logp, fallback=-1e30)
    assert sorted(ids.tolist()) == [4, 5, 7, 8, 9]


def test_select_ids_empty_intersection_falls_back_to_teacher():
    sc = _make_scorer("intersection", top_k=3)
    s_logp = torch.log_softmax(torch.arange(10).float(), dim=-1)  # top-3 = {9, 8, 7}
    t_ids = torch.tensor([1, 2, 3])         # disjoint from student top-k
    t_logp = torch.tensor([-2.0, -3.0, -4.0])
    ids, t_aligned = sc._select_ids(s_logp, t_ids, t_logp, fallback=-1e30)
    assert torch.equal(ids, t_ids) and torch.equal(t_aligned, t_logp)


def test_topk_window_preserves_teacher_id_outside_student_topk():
    # student top-1 is id 0; teacher's top id is 5 (outside student top-1).
    # With strategy="union", the scored id set MUST include id 5.
    # Pins the C-1 invariant: the decode-side top-k slice (C1) must keep a window
    # wide enough (topk_store_k = max(log_prob_top_k, 512)) that _select_ids can
    # still see teacher ids outside the student top-k, or union/intersection/
    # teacher_p silently degrade to only_stu.
    sc = TeacherScorer.__new__(TeacherScorer)
    sc.top_k = 1; sc.top_k_strategy = "union"; sc.weight_mode = "none"
    sc.teacher_temperature = 1.0
    s_clean_full = torch.tensor([10.0, -1, -1, -1, -1, -1])  # student argmax=0
    t_ids = torch.tensor([5, 0]); t_logp = torch.tensor([-0.1, -2.0])
    ids, t_aligned = sc._select_ids(s_clean_full, t_ids, t_logp, fallback=-50.0)
    assert 5 in ids.tolist() and 0 in ids.tolist()  # union keeps teacher id 5


def test_topk_slice_matches_full_vocab_kl_only_stu():
    import torch
    from verl.trainer.np.teacher_scorer import reverse_kl_topk
    vocab, n, k = 200, 4, 32
    logits = torch.randn(1 + n, vocab)
    # full-vocab reference: take student top-k ids, score
    ids = torch.topk(logits[0], k).indices
    s_full = torch.log_softmax(logits.float(), -1)
    ref = s_full[:, ids]
    # sliced storage: gather all rows on the same ids, store [1+n, k] + ids
    s_sliced = torch.log_softmax(logits.float(), -1)[:, ids]  # log_softmax over full vocab, then slice
    assert torch.allclose(ref, s_sliced, atol=1e-5)


def test_topk_store_helper_returns_logprobs_and_ids():
    import torch
    from verl.workers.rollout.vllm_rollout.np_worker_extension import WorkerExtension
    we = WorkerExtension.__new__(WorkerExtension)
    vocab, n, k = 200, 4, 32
    logits = torch.randn(1 + n, vocab)
    topk_logp, ids = we._topk_store(logits, k)
    assert topk_logp.shape == (1 + n, k)
    assert ids.shape == (k,)
    # ids are the student (row 0) top-k
    assert torch.equal(torch.sort(ids).values,
                       torch.sort(torch.topk(logits[0], k).indices).values)
    # stored values == log_softmax over full vocab, then sliced on those ids
    ref = torch.log_softmax(logits.float(), -1)[:, ids]
    assert torch.allclose(topk_logp, ref, atol=1e-5)
    # outputs are on CPU (D2H done inside the helper)
    assert topk_logp.device.type == "cpu" and ids.device.type == "cpu"


def _fake_teacher_scorer(strategy, weight_mode, top_k, teacher_ids_by_pos, teacher_logp_by_pos):
    from verl.trainer.np.teacher_scorer import TeacherScorer
    sc = TeacherScorer.__new__(TeacherScorer)
    sc.top_k = top_k; sc.top_k_strategy = strategy; sc.weight_mode = weight_mode
    sc.teacher_temperature = 1.0
    sc._teacher_topk_logprobs = lambda prefix, n_steps: (teacher_logp_by_pos, teacher_ids_by_pos)
    return sc


@pytest.mark.parametrize("weight_mode", ["student_p", "teacher_p", "none"])
def test_vectorized_score_matches_legacy_only_stu(weight_mode):
    # Parity oracle: the new tuple-fed (C1) vectorized path must reproduce the
    # legacy full-vocab per-token-loop result for only_stu (lossless production
    # default — the scored id set is the student top-k, a subset of the stored
    # window, so there is NO approximation).
    from verl.workers.rollout.vllm_rollout.np_worker_extension import WorkerExtension
    torch.manual_seed(0)
    T, n, vocab, top_k, store_k = 6, 4, 128, 8, 64
    we = WorkerExtension.__new__(WorkerExtension)
    logits_steps = [torch.randn(1 + n, vocab) for _ in range(T)]
    # teacher: random ids + logps per position
    t_ids = [torch.randperm(vocab)[:16] for _ in range(T)]
    t_logp = [torch.log_softmax(torch.randn(16), -1) for _ in range(T)]
    legacy_cl = [s.clone() for s in logits_steps]                  # full-vocab tensors
    tuple_cl = [we._topk_store(s, store_k) for s in logits_steps]  # C1 tuples

    sc_legacy = _fake_teacher_scorer("only_stu", weight_mode, top_k, t_ids, t_logp)
    sc_vec = _fake_teacher_scorer("only_stu", weight_mode, top_k, t_ids, t_logp)
    Lq_ref, Lc_ref = sc_legacy.score_rollout(None, legacy_cl)
    Lq_new, Lc_new = sc_vec.score_rollout(None, tuple_cl)

    for t in range(T):
        rel = (Lq_new[t] - Lq_ref[t]).norm() / (Lq_ref[t].norm() + 1e-9)
        assert rel < 1e-4, f"step {t} weight={weight_mode} rel={rel:.2e}"
        assert abs(Lc_new[t] - Lc_ref[t]) < 1e-4 * (abs(Lc_ref[t]) + 1e-6)


@pytest.mark.parametrize("strategy", ["only_stu", "only_tch", "intersection", "union"])
@pytest.mark.parametrize("weight_mode", ["student_p", "teacher_p", "none"])
def test_vectorized_score_matches_legacy_full_window_all_strategies(strategy, weight_mode):
    # FULL-WINDOW exactness: with store_k >= vocab the stored student window IS
    # the full vocab, so NO scored teacher id is ever out-of-window and the
    # s_floor approximation is never used. Therefore the vectorized tuple path
    # must reproduce the legacy full-vocab per-token-loop result for ALL
    # strategies (only_stu / only_tch / intersection / union) and all weight
    # modes. This guards the id-set selection + teacher-alignment logic against
    # future regressions. (At small store_k, only_stu stays exact while the
    # others deviate — that is documented, not tested here.)
    from verl.workers.rollout.vllm_rollout.np_worker_extension import WorkerExtension
    torch.manual_seed(0)
    T, n, vocab, top_k, store_k = 6, 4, 128, 8, 128  # store_k == vocab -> full window
    we = WorkerExtension.__new__(WorkerExtension)
    logits_steps = [torch.randn(1 + n, vocab) for _ in range(T)]

    # Build teacher ids that include some WITHIN the student top-`top_k` and some
    # genuinely OUTSIDE it (but inside store_k=vocab), so union/intersection have
    # non-trivial content. All ids are in-window since store_k == vocab.
    t_ids, t_logp = [], []
    for s in logits_steps:
        s_top = torch.topk(s[0], top_k).indices                  # student top-`top_k`
        not_top = torch.tensor([i for i in range(vocab) if i not in set(s_top.tolist())])
        inside = s_top[:4]                                        # 4 ids inside student top-k
        outside = not_top[torch.randperm(not_top.numel())[:12]]  # 12 ids outside it
        ids = torch.cat([inside, outside])
        t_ids.append(ids)
        t_logp.append(torch.log_softmax(torch.randn(ids.numel()), -1))

    legacy_cl = [s.clone() for s in logits_steps]                  # full-vocab tensors
    tuple_cl = [we._topk_store(s, store_k) for s in logits_steps]  # C1 tuples (full window)

    sc_legacy = _fake_teacher_scorer(strategy, weight_mode, top_k, t_ids, t_logp)
    sc_vec = _fake_teacher_scorer(strategy, weight_mode, top_k, t_ids, t_logp)
    Lq_ref, Lc_ref = sc_legacy.score_rollout(None, legacy_cl)
    Lq_new, Lc_new = sc_vec.score_rollout(None, tuple_cl)

    for t in range(T):
        rel = (Lq_new[t] - Lq_ref[t]).norm() / (Lq_ref[t].norm() + 1e-9)
        assert rel < 1e-4, f"step {t} strat={strategy} weight={weight_mode} rel={rel:.2e}"
        assert abs(Lc_new[t] - Lc_ref[t]) < 1e-4 * (abs(Lc_ref[t]) + 1e-6)


@pytest.mark.parametrize("strategy", ["only_stu", "only_tch", "intersection", "union"])
@pytest.mark.parametrize("weight_mode", ["student_p", "teacher_p", "none"])
def test_score_rollout_offwindow_teacher_id_survives_wide_window(strategy, weight_mode):
    # C-1 LANDMINE, end-to-end. The prior full-window test sets store_k == vocab so
    # nothing is ever out of window; the _select_ids window test exercises the
    # selection logic in isolation. This one closes the gap: a WIDE-but-FINITE
    # window (store_k=512, vocab=1024) with teacher ids that sit OUTSIDE the
    # student top-`top_k` (=8) but INSIDE the stored 512-window. The decision
    # `topk_store_k >= 512` exists precisely so union/intersection still SEE these
    # ids; if the stored window were narrowed below their student-rank they would
    # silently vanish and union/intersection would degrade to only_stu.
    #
    # We pick teacher ids by RANK in the student clean-row (row 0) distribution via
    # argsort, so the geometry is deterministic and explicit: rank 2 is inside the
    # top-8, ranks 20 and 100 are outside the top-8 but well inside the top-512.
    # Asserts:
    #   (a) the vectorized tuple path == legacy full-vocab path within rel-Frob
    #       1e-4 for ALL strategies/modes -- EXACT here because every scored id
    #       (incl. the rank-20/100 teacher ids) is inside the 512-window, so the
    #       s_floor out-of-window approximation never fires (no flooring).
    #   (b) union's per-step L_q DIFFERS from only_stu's on the SAME tuple inputs,
    #       proving the rank-20/100 teacher ids were actually scored (union didn't
    #       collapse to only_stu).
    from verl.workers.rollout.vllm_rollout.np_worker_extension import WorkerExtension
    torch.manual_seed(0)
    T, n, vocab, top_k, store_k = 6, 4, 1024, 8, 512  # top-8 < 512-window < vocab
    we = WorkerExtension.__new__(WorkerExtension)
    logits_steps = [torch.randn(1 + n, vocab) for _ in range(T)]

    # Teacher ids chosen by student clean-row RANK: {2 (in top-8), 20, 100}.
    # argsort(descending) maps rank -> vocab id deterministically. The off-top-8
    # ranks 20/100 are the C-1 ids: outside the student top-8, inside the 512-window.
    rank_picks = torch.tensor([2, 20, 100])
    t_ids, t_logp = [], []
    for s in logits_steps:
        order = torch.argsort(s[0], descending=True)   # order[r] = id at student-rank r
        ids = order[rank_picks]
        s_top8 = set(torch.topk(s[0], top_k).indices.tolist())
        s_win = set(torch.topk(s[0], store_k).indices.tolist())
        # Guard the geometry the test depends on: rank 2 in top-8; ranks 20/100
        # outside top-8 but inside the 512-window (so they are scored from the
        # window exactly, never floored).
        assert ids[0].item() in s_top8
        assert ids[1].item() not in s_top8 and ids[1].item() in s_win
        assert ids[2].item() not in s_top8 and ids[2].item() in s_win
        t_ids.append(ids)
        t_logp.append(torch.log_softmax(torch.randn(ids.numel()), -1))

    legacy_cl = [s.clone() for s in logits_steps]                  # full-vocab tensors
    tuple_cl = [we._topk_store(s, store_k) for s in logits_steps]  # C1 tuples (512-window)

    sc_legacy = _fake_teacher_scorer(strategy, weight_mode, top_k, t_ids, t_logp)
    sc_vec = _fake_teacher_scorer(strategy, weight_mode, top_k, t_ids, t_logp)
    Lq_ref, Lc_ref = sc_legacy.score_rollout(None, legacy_cl)
    Lq_new, Lc_new = sc_vec.score_rollout(None, tuple_cl)

    # (a) Exact parity: the rank-100 teacher id is inside the 512-window, so the
    # windowed lookup must reproduce the full-vocab result with no flooring.
    for t in range(T):
        rel = (Lq_new[t] - Lq_ref[t]).norm() / (Lq_ref[t].norm() + 1e-9)
        assert rel < 1e-4, f"step {t} strat={strategy} weight={weight_mode} rel={rel:.2e}"
        assert abs(Lc_new[t] - Lc_ref[t]) < 1e-4 * (abs(Lc_ref[t]) + 1e-6)

    # (b) Only when union ADDS the off-top-8 teacher ids does its loss diverge from
    # only_stu (whose scored set is exactly the student top-8). Run both on the
    # SAME tuple inputs; a non-trivial per-step difference proves the rank-20/100
    # teacher ids were scored, i.e. union did NOT degrade to only_stu.
    if strategy == "union":
        sc_stu = _fake_teacher_scorer("only_stu", weight_mode, top_k, t_ids, t_logp)
        Lq_stu, _ = sc_stu.score_rollout(None, tuple_cl)
        max_diff = max((Lq_new[t] - Lq_stu[t]).abs().max().item() for t in range(T))
        assert max_diff > 1e-4, (
            f"union collapsed to only_stu (weight={weight_mode} max|diff|={max_diff:.2e}) "
            "-- the off-window teacher ids were NOT scored")


# ---------------------------------------------------------------------------
# C3: batched teacher prefill (score_wave) parity with serial score_rollout.
# ---------------------------------------------------------------------------


class _FakeLogprob:
    def __init__(self, lp):
        self.logprob = lp


class _FakeOut:
    def __init__(self, prompt_logprobs):
        self.prompt_logprobs = prompt_logprobs


class _FakeGenerate:
    """Mimics engine.generate; .remote(prompts, sp, use_tqdm=...) returns a list
    of _FakeOut, one per prompt (dict input -> 1-element list, like vLLM).

    Keyed by the prompt_token_ids tuple so a serial dict call for prompt i and
    the batched list call see the SAME prompt_logprobs for prompt i.
    """

    def __init__(self, by_prefix):
        self._by_prefix = by_prefix  # {tuple(prefix): _FakeOut}

    def remote(self, prompts, sp, use_tqdm=False):
        if isinstance(prompts, dict):
            return [self._by_prefix[tuple(prompts["prompt_token_ids"])]]
        return [self._by_prefix[tuple(p["prompt_token_ids"])] for p in prompts]


class _FakeEngine:
    def __init__(self, by_prefix):
        self.generate = _FakeGenerate(by_prefix)


def _build_wave_fixture(monkeypatch, top_k=8, store_k=64, strategy="union",
                        weight_mode="student_p", B=3, vocab=128, n=4):
    """Build B prompts, each with its own candidate_logits (C1 tuples) and a
    deterministic per-prompt teacher prompt_logprobs surface. Returns
    (scorer, prefixes, cand_list). Patches the module's ray.get to identity so
    the fake .remote return value flows straight through.
    """
    from verl.workers.rollout.vllm_rollout.np_worker_extension import WorkerExtension

    # _teacher_topk_logprobs and score_wave do `import ray` locally, so patching
    # the real ray module's get makes the fake .remote return flow straight through.
    import ray
    monkeypatch.setattr(ray, "get", lambda x: x)

    we = WorkerExtension.__new__(WorkerExtension)
    torch.manual_seed(0)

    prefixes, cand_list, by_prefix = [], [], {}
    for i in range(B):
        T = 3 + i  # vary number of steps per prompt
        prefix = list(range(10 + i)) + list(range(50, 50 + T))  # unique per prompt
        prefixes.append(prefix)
        logits_steps = [torch.randn(1 + n, vocab) for _ in range(T)]
        cand_list.append([we._topk_store(s, store_k) for s in logits_steps])

        # Build prompt_logprobs: a list aligned to prompt positions; only the LAST
        # T positions (the response region) are read. Pad the head with junk dicts.
        plp = [None] * (len(prefix) - T)
        for s in logits_steps:
            t_ids = torch.randperm(vocab)[:16].tolist()
            t_lp = torch.log_softmax(torch.randn(16), -1).tolist()
            plp.append({tid: _FakeLogprob(lp) for tid, lp in zip(t_ids, t_lp)})
        by_prefix[tuple(prefix)] = _FakeOut(plp)

    engine = _FakeEngine(by_prefix)
    scorer = TeacherScorer(teacher_engine=engine, top_k=top_k,
                           top_k_strategy=strategy,
                           teacher_temperature=1.0, weight_mode=weight_mode)
    return scorer, prefixes, cand_list


def _assert_wave_eq_serial(scorer, prefixes, cand_list):
    wave = scorer.score_wave(prefixes, cand_list)
    assert len(wave) == len(prefixes)
    for i, prefix in enumerate(prefixes):
        Lq_ref, Lc_ref = scorer.score_rollout(prefix, cand_list[i])
        Lq_w, Lc_w = wave[i]
        assert len(Lq_w) == len(Lq_ref) and len(Lc_w) == len(Lc_ref)
        for t in range(len(Lq_ref)):
            assert torch.allclose(Lq_w[t], Lq_ref[t], atol=1e-6), \
                f"prompt {i} step {t} L_q mismatch"
            assert abs(Lc_w[t] - Lc_ref[t]) < 1e-6, \
                f"prompt {i} step {t} L_clean mismatch"


@pytest.mark.parametrize("weight_mode", ["student_p", "teacher_p", "none"])
@pytest.mark.parametrize("strategy", ["only_stu", "only_tch", "intersection", "union"])
def test_score_wave_matches_serial_score_rollout(monkeypatch, strategy, weight_mode):
    # ONE batched teacher generate over B prompts must yield per-prompt results
    # element-equal to B serial score_rollout calls. The fake engine is keyed by
    # prefix so the serial (dict) and batched (list) paths see identical teacher
    # prompt_logprobs for each prompt.
    scorer, prefixes, cand_list = _build_wave_fixture(
        monkeypatch, strategy=strategy, weight_mode=weight_mode, B=3)
    _assert_wave_eq_serial(scorer, prefixes, cand_list)


def test_score_wave_chunking_transparent(monkeypatch):
    # With teacher_batch_size=2 and 3 prompts the wave is processed in chunks
    # (2 + 1) but the result still matches serial score_rollout per prompt.
    scorer, prefixes, cand_list = _build_wave_fixture(monkeypatch, B=3)
    scorer.teacher_batch_size = 2
    _assert_wave_eq_serial(scorer, prefixes, cand_list)


def test_score_wave_serial_fallback(monkeypatch):
    # teacher_batch_size <= 1 -> serial fallback (loop score_rollout per prompt),
    # still matches serial score_rollout per prompt.
    scorer, prefixes, cand_list = _build_wave_fixture(monkeypatch, B=3)
    scorer.teacher_batch_size = 1
    _assert_wave_eq_serial(scorer, prefixes, cand_list)


def test_teacher_batch_size_default_is_16():
    sc = TeacherScorer(teacher_engine=None, top_k=8, top_k_strategy="union",
                       teacher_temperature=1.0, weight_mode="none")
    assert sc.teacher_batch_size == 16
