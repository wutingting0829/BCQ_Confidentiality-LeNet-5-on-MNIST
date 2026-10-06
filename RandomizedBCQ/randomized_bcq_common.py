import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "RandomizedBCQ"
WEIGHT_DIR = ROOT / "weight_data"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Attacker"))

from bcq_format import b_mtx_to_flat  # noqa: E402


WEIGHT_NAMES = ["conv1.weight", "conv2.weight", "fc1.weight", "fc2.weight", "fc3.weight"]
BIAS_NAMES = ["conv1.bias", "conv2.bias", "fc1.bias", "fc2.bias", "fc3.bias"]
NUM_BITS = 3
RANDOM_SEED = 20260810

FULL_SECRET_PATH = OUT_DIR / "01_randomized_BCQ_full_secret_result.pth"
ATTACKER_VISIBLE_PATH = OUT_DIR / "01_attacker_visible_randomized_B_only.pth"
LEAKY_CONTROL_PATH = OUT_DIR / "01_leaky_true_B_control.pth"
ATTACK_WEIGHTS_PATH = OUT_DIR / "02_weighted_sum_attack_weights.pth"

ATTACKS = {
    "equal_sum_1_1_1": torch.tensor([1.0, 1.0, 1.0]),
    "heuristic_half_1_0.5_0.25": torch.tensor([1.0, 0.5, 0.25]),
    "heuristic_binary_4_2_1": torch.tensor([4.0, 2.0, 1.0]),
}


def stable_layer_seed(layer_name, suffix):
    stable = sum((idx + 1) * ord(ch) for idx, ch in enumerate(f"{layer_name}:{suffix}"))
    return RANDOM_SEED + stable


def bcq_quantize(weight, num_bits=NUM_BITS):
    original_shape = weight.shape
    m = original_shape[0]
    weight_flat = weight.reshape(m, -1).clone()
    z = weight_flat.mean(dim=1, keepdim=True)
    residual = weight_flat - z

    b_list = []
    alpha_list = []
    w_bcq_centered_flat = torch.zeros_like(residual)

    for _ in range(num_bits):
        b = torch.sign(residual)
        b[b == 0] = 1
        alpha = residual.abs().mean(dim=1, keepdim=True)

        b_list.append(b)
        alpha_list.append(alpha)

        w_bcq_centered_flat += alpha * b
        residual = residual - alpha * b

    b_true = torch.stack(b_list, dim=0)
    alpha = torch.stack(alpha_list, dim=0)
    w_bcq = (w_bcq_centered_flat + z).reshape(original_shape)
    return b_true, alpha, z, w_bcq


def make_secret_sign_mask(shape, layer_name):
    generator = torch.Generator().manual_seed(stable_layer_seed(layer_name, "sign_mask"))
    mask = torch.randint(0, 2, shape, generator=generator, dtype=torch.int64)
    return mask.float().mul(2).sub(1)


def make_secret_plane_perm(layer_name, num_bits=NUM_BITS):
    generator = torch.Generator().manual_seed(stable_layer_seed(layer_name, "plane_perm"))
    return torch.randperm(num_bits, generator=generator)


def protect_b_mtx(b_true, layer_name):
    secret_sign_mask = make_secret_sign_mask(b_true.shape, layer_name)
    secret_plane_perm = make_secret_plane_perm(layer_name, b_true.shape[0])
    b_masked = b_true * secret_sign_mask
    b_public = b_masked[secret_plane_perm]
    return b_public, secret_sign_mask, secret_plane_perm


def recover_true_b_from_public(b_public, secret_sign_mask, secret_plane_perm):
    b_masked = torch.empty_like(b_public)
    b_masked[secret_plane_perm] = b_public
    return b_masked * secret_sign_mask


def reconstruct_bcq_from_public(b_public, alpha, z, secret_sign_mask, secret_plane_perm, target_shape):
    b_true = recover_true_b_from_public(b_public, secret_sign_mask, secret_plane_perm)
    centered = (alpha * b_true).sum(dim=0)
    return (centered + z).reshape(target_shape)


def zscore(x, eps=1e-12):
    return (x - x.mean()) / (x.std(unbiased=False) + eps)


def rescale_to_true(w_attack, w_true, eps=1e-12):
    return zscore(w_attack, eps=eps) * w_true.std(unbiased=False) + w_true.mean()


def weighted_sum_from_b(b_mtx, bit_weights, target_shape):
    b_flat = b_mtx_to_flat(b_mtx)
    bit_weights = bit_weights.to(dtype=b_flat.dtype)
    return (b_flat @ bit_weights).reshape(target_shape)


def pearson_corr(a, b, eps=1e-12):
    a_centered = a - a.mean()
    b_centered = b - b.mean()
    numerator = torch.sum(a_centered * b_centered)
    denominator = torch.sqrt(torch.sum(a_centered**2)) * torch.sqrt(torch.sum(b_centered**2))
    return (numerator / (denominator + eps)).item()


def rankdata(x):
    order = torch.argsort(x)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(x.numel(), dtype=torch.float32)
    return ranks


def spearman_corr(a, b):
    return pearson_corr(rankdata(a), rankdata(b))


def sign_accuracy(w_true, w_attack):
    true_sign = torch.sign(w_true)
    attack_sign = torch.sign(w_attack)
    true_sign[true_sign == 0] = 1
    attack_sign[attack_sign == 0] = 1
    return (true_sign == attack_sign).float().mean().item()


def make_random_bias_overrides(base_state):
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    overrides = {}
    for name in BIAS_NAMES:
        bias = base_state[name]
        random_bias = torch.randn(bias.shape, generator=generator, dtype=bias.dtype)
        random_bias = random_bias * bias.std(unbiased=False) + bias.mean()
        overrides[name] = random_bias
    return overrides


def with_overrides(base_state, overrides):
    state = dict(base_state)
    state.update(overrides)
    return state
