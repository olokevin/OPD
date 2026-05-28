import torch
from verl.trainer.np.seeding import noise_seed, draw_noise


def test_noise_seed_is_deterministic_and_64bit():
    s1 = noise_seed(global_seed=42, step=3, layer="model.layers.0.mlp.down_proj", rollout=1, q=2)
    s2 = noise_seed(global_seed=42, step=3, layer="model.layers.0.mlp.down_proj", rollout=1, q=2)
    assert s1 == s2                      # deterministic
    assert 0 <= s1 < 2**63               # fits a signed 64-bit generator seed


def test_noise_seed_varies_with_each_field():
    base = dict(global_seed=42, step=3, layer="L", rollout=1, q=2)
    s = noise_seed(**base)
    assert noise_seed(**{**base, "step": 4}) != s
    assert noise_seed(**{**base, "layer": "M"}) != s
    assert noise_seed(**{**base, "rollout": 2}) != s
    assert noise_seed(**{**base, "q": 3}) != s


def test_draw_noise_reproducible_from_seed():
    seed = 123456789
    a = draw_noise(seed, (4, 8), torch.device("cpu"), torch.float32, method="gaussian")
    b = draw_noise(seed, (4, 8), torch.device("cpu"), torch.float32, method="gaussian")
    assert torch.equal(a, b)             # same seed -> identical noise
    assert a.shape == (4, 8)


def test_draw_noise_bernoulli_is_pm1():
    n = draw_noise(7, (1000,), torch.device("cpu"), torch.float32, method="bernoulli")
    uniq = set(n.unique().tolist())
    assert uniq.issubset({-1.0, 1.0})    # Rademacher: only +/-1


def test_draw_noise_uniform_in_unit_range():
    n = draw_noise(7, (1000,), torch.device("cpu"), torch.float32, method="uniform")
    assert n.min() >= -1.0 and n.max() <= 1.0


def test_draw_noise_methods_differ():
    g = draw_noise(7, (1000,), torch.device("cpu"), torch.float32, method="gaussian")
    b = draw_noise(7, (1000,), torch.device("cpu"), torch.float32, method="bernoulli")
    assert not torch.equal(g, b)
