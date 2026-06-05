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
