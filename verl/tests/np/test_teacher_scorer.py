import torch
from verl.trainer.np.teacher_scorer import reverse_kl_topk


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
