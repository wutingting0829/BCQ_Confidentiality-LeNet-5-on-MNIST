"""
Randomized residual-sign BCQ experiment for LeNet-5 MNIST.

Goal
----
The original residual-sign BCQ stores B_mtx as plaintext sign planes. That makes
B_mtx highly correlated with W_true and allows simple weighted-sum attacks.

This experiment keeps the same defender-side BCQ reconstruction quality, but the
attacker-visible plaintext B_mtx is randomized with two secret transforms:

1. element-wise sign mask: B_masked = B_true * secret_mask
2. bit-plane assignment: B_public[k] = B_masked[secret_plane_perm[k]]

The defender can invert both transforms before applying alpha and z. The
attacker only sees B_public, so B_public no longer directly exposes residual
signs or the original bit-plane importance order.

1. 建立 randomized BCQ model weights
2. 建立 attacker-visible randomized B_public
3. 跑 Weighted-sum Attack metrics
4. 把 attacker 重建 weights 放回 LeNet-5 測 MNIST accuracy
5. 產生 distribution plot

"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "RandomizedBCQ"
OUT_DIR.mkdir(exist_ok=True)

WEIGHT_DIR = ROOT / "weight_data"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Attacker"))

from model import Model  # noqa: E402
from bcq_format import b_mtx_to_flat  # noqa: E402


WEIGHT_NAMES = ["conv1.weight", "conv2.weight", "fc1.weight", "fc2.weight", "fc3.weight"]
BIAS_NAMES = ["conv1.bias", "conv2.bias", "fc1.bias", "fc2.bias", "fc3.bias"]
NUM_BITS = 3
RANDOM_SEED = 20260810

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


def weighted_sum_from_b(b_mtx, bit_weights, target_shape):
    b_flat = b_mtx_to_flat(b_mtx)
    bit_weights = bit_weights.to(dtype=b_flat.dtype)
    return (b_flat @ bit_weights).reshape(target_shape)


def flatten_pair(w_true, b_mtx):
    return w_true.detach().cpu().float().flatten(), b_mtx_to_flat(b_mtx)


def compute_attack_metrics(layer_name, attacker_view, w_true, b_mtx, attack_name, bit_weights):
    w_flat, b_flat = flatten_pair(w_true, b_mtx)
    bit_weights = bit_weights.to(dtype=b_flat.dtype)
    w_attack = b_flat @ bit_weights
    w_attack_rescaled = rescale_to_true(w_attack, w_flat)

    return {
        "layer": layer_name,
        "attacker_view": attacker_view,
        "attack": attack_name,
        "bit_weights": ",".join(f"{x:g}" for x in bit_weights.tolist()),
        "n_weights": w_flat.numel(),
        "sign_acc": sign_accuracy(w_flat, w_attack),
        "pearson": pearson_corr(w_flat, w_attack),
        "spearman": spearman_corr(w_flat, w_attack),
        "rescaled_mae": (w_attack_rescaled - w_flat).abs().mean().item(),
        "rescaled_rmse": torch.sqrt(((w_attack_rescaled - w_flat) ** 2).mean()).item(),
    }


def load_mnist_test_loader(batch_size=256):
    images, labels = torch.load(ROOT / "test" / "MNIST" / "processed" / "test.pt", map_location="cpu")
    images = images.float().div(255.0).unsqueeze(1)
    return DataLoader(TensorDataset(images, labels), batch_size=batch_size)


def evaluate_accuracy(state_dict, test_loader, device):
    model = Model().to(device)
    model.load_state_dict(state_dict)
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.shape[0]
    return correct / total


def with_overrides(base_state, overrides):
    state = dict(base_state)
    state.update(overrides)
    return state


def make_random_bias_overrides(base_state):
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    overrides = {}
    for name in BIAS_NAMES:
        bias = base_state[name]
        random_bias = torch.randn(bias.shape, generator=generator, dtype=bias.dtype)
        random_bias = random_bias * bias.std(unbiased=False) + bias.mean()
        overrides[name] = random_bias
    return overrides


def save_distribution_plot(w_true_all, views):
    fig, axes = plt.subplots(len(views), 1, figsize=(9, 3.2 * len(views)), constrained_layout=True)
    if len(views) == 1:
        axes = [axes]

    true_z = zscore(w_true_all).numpy()
    bit_weights = ATTACKS["heuristic_half_1_0.5_0.25"]

    for ax, (view_name, b_all) in zip(axes, views.items()):
        attack_z = zscore(b_all @ bit_weights.to(dtype=b_all.dtype)).numpy()
        ax.hist(true_z, bins=80, density=True, alpha=0.55, label="W_true z-score")
        ax.hist(attack_z, bins=80, density=True, alpha=0.55, label="W_attack z-score")
        ax.set_title(f"{view_name}: heuristic_half_1_0.5_0.25")
        ax.set_xlabel("standardized value")
        ax.set_ylabel("density")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()

    output_path = OUT_DIR / "randomized_weighted_sum_attack_distributions.png"
    fig.suptitle("Randomized Residual-sign BCQ: Distribution Comparison")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    w_true_state = torch.load(WEIGHT_DIR / "W_true.pth", map_location="cpu")

    full_result = {}
    attacker_visible = {}
    leaky_true_b = {}

    max_reconstruction_diff = 0.0
    for name in WEIGHT_NAMES:
        b_true, alpha, z, w_bcq_original = bcq_quantize(w_true_state[name], num_bits=NUM_BITS)
        b_public, secret_sign_mask, secret_plane_perm = protect_b_mtx(b_true, name)
        w_bcq_recovered = reconstruct_bcq_from_public(
            b_public,
            alpha,
            z,
            secret_sign_mask,
            secret_plane_perm,
            w_true_state[name].shape,
        )
        max_reconstruction_diff = max(
            max_reconstruction_diff,
            (w_bcq_recovered - w_bcq_original).abs().max().item(),
        )

        full_result[name] = {
            "W_true": w_true_state[name],
            "B_mtx_true_secret": b_true,
            "B_mtx_public": b_public,
            "secret_sign_mask": secret_sign_mask,
            "secret_plane_perm": secret_plane_perm,
            "alpha": alpha,
            "z": z,
            "W_BCQ": w_bcq_recovered,
        }
        attacker_visible[name] = {"B_mtx": b_public}
        leaky_true_b[name] = {"B_mtx": b_true}

    torch.save(full_result, OUT_DIR / "randomized_BCQ_full_secret_result.pth")
    torch.save(attacker_visible, OUT_DIR / "attacker_visible_randomized_B_only.pth")

    metric_rows = []
    all_true = []
    all_randomized_b = []
    all_leaky_b = []
    for name in WEIGHT_NAMES:
        all_true.append(w_true_state[name].detach().cpu().float().flatten())
        all_randomized_b.append(b_mtx_to_flat(attacker_visible[name]["B_mtx"]))
        all_leaky_b.append(b_mtx_to_flat(leaky_true_b[name]["B_mtx"]))

        for attack_name, bit_weights in ATTACKS.items():
            metric_rows.append(
                compute_attack_metrics(
                    name,
                    "leaky_true_B_control",
                    w_true_state[name],
                    leaky_true_b[name]["B_mtx"],
                    attack_name,
                    bit_weights,
                )
            )
            metric_rows.append(
                compute_attack_metrics(
                    name,
                    "randomized_public_B",
                    w_true_state[name],
                    attacker_visible[name]["B_mtx"],
                    attack_name,
                    bit_weights,
                )
            )

    overall_true = torch.cat(all_true)
    overall_randomized_b = torch.cat(all_randomized_b)
    overall_leaky_b = torch.cat(all_leaky_b)
    for attack_name, bit_weights in ATTACKS.items():
        metric_rows.append(
            compute_attack_metrics(
                "OVERALL",
                "leaky_true_B_control",
                overall_true,
                overall_leaky_b,
                attack_name,
                bit_weights,
            )
        )
        metric_rows.append(
            compute_attack_metrics(
                "OVERALL",
                "randomized_public_B",
                overall_true,
                overall_randomized_b,
                attack_name,
                bit_weights,
            )
        )

    metrics_df = pd.DataFrame(metric_rows)
    metrics_path = OUT_DIR / "randomized_weighted_sum_attack_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    test_loader = load_mnist_test_loader()
    random_bias_overrides = make_random_bias_overrides(w_true_state)

    accuracy_rows = [
        {
            "attacker_view": "baseline",
            "attack": "W_true",
            "scale": "true",
            "bias": "trained",
            "accuracy": evaluate_accuracy(w_true_state, test_loader, device),
        },
        {
            "attacker_view": "defender_secret_reconstruction",
            "attack": "W_BCQ_from_randomized_public_B_plus_secrets",
            "scale": "alpha_z",
            "bias": "trained",
            "accuracy": evaluate_accuracy(
                with_overrides(w_true_state, {name: full_result[name]["W_BCQ"] for name in WEIGHT_NAMES}),
                test_loader,
                device,
            ),
        },
    ]

    for view_name, b_source in [
        ("leaky_true_B_control", leaky_true_b),
        ("randomized_public_B", attacker_visible),
    ]:
        for attack_name, bit_weights in ATTACKS.items():
            raw_overrides = {}
            rescaled_overrides = {}
            for name in WEIGHT_NAMES:
                w_attack = weighted_sum_from_b(b_source[name]["B_mtx"], bit_weights, w_true_state[name].shape)
                raw_overrides[name] = w_attack
                rescaled_overrides[name] = rescale_to_true(w_attack, w_true_state[name])

            for scale_name, overrides in [
                ("raw", raw_overrides),
                ("oracle_rescaled_to_W_true_mean_std", rescaled_overrides),
            ]:
                for bias_name, bias_overrides in [("trained", {}), ("random", random_bias_overrides)]:
                    accuracy_rows.append(
                        {
                            "attacker_view": view_name,
                            "attack": attack_name,
                            "scale": scale_name,
                            "bias": bias_name,
                            "accuracy": evaluate_accuracy(
                                with_overrides(w_true_state, {**overrides, **bias_overrides}),
                                test_loader,
                                device,
                            ),
                        }
                    )

    accuracy_df = pd.DataFrame(accuracy_rows)
    accuracy_path = OUT_DIR / "randomized_attack_reconstruction_accuracy.csv"
    accuracy_df.to_csv(accuracy_path, index=False)

    plot_path = save_distribution_plot(
        overall_true,
        {
            "leaky_true_B_control": overall_leaky_b,
            "randomized_public_B": overall_randomized_b,
        },
    )

    readme_path = OUT_DIR / "README_randomized_bcq.md"
    readme_path.write_text(
        "\n".join(
            [
                "# Randomized Residual-sign BCQ",
                "",
                "This folder contains a randomized residual-sign BCQ experiment.",
                "",
                "Attacker-visible plaintext data is `B_mtx_public`, not the true aligned residual-sign `B_mtx_true_secret`.",
                "The defender reconstructs by using the secret sign mask and secret bit-plane permutation before applying alpha and z.",
                "",
                "Generated files:",
                "",
                "- `randomized_BCQ_full_secret_result.pth`: full defender-side result, including secrets.",
                "- `attacker_visible_randomized_B_only.pth`: attacker-visible plaintext randomized B only.",
                "- `randomized_weighted_sum_attack_metrics.csv`: correlation/sign/rank attack metrics.",
                "- `randomized_attack_reconstruction_accuracy.csv`: MNIST accuracy after loading attack weights.",
                "- `randomized_weighted_sum_attack_distributions.png`: distribution comparison plot.",
                "",
                f"Max defender reconstruction diff after inverting secrets: {max_reconstruction_diff:.8g}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print("Randomized residual-sign BCQ experiment")
    print("Output directory:", OUT_DIR)
    print("Max defender reconstruction diff after inverting secrets:", max_reconstruction_diff)
    print()
    print("OVERALL weighted-sum attack metrics:")
    print(metrics_df[metrics_df["layer"] == "OVERALL"].to_string(index=False))
    print()
    print("Attack reconstruction accuracy:")
    print(accuracy_df.to_string(index=False))
    print()
    print("Saved:", metrics_path)
    print("Saved:", accuracy_path)
    print("Saved:", plot_path)
    print("Saved:", readme_path)


if __name__ == "__main__":
    main()
