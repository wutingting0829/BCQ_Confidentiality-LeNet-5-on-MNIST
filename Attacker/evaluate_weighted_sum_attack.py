from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from bcq_format import b_mtx_to_flat


ROOT = Path(__file__).resolve().parents[1]
WEIGHT_DIR = ROOT / "weight_data"
ATTACKER_DIR = ROOT / "Attacker"

ATTACKS = {
    "equal_sum_1_1_1": torch.tensor([1.0, 1.0, 1.0]), #Attack 1 baseline: Equal-sum
    "heuristic_half_1_0.5_0.25": torch.tensor([1.0, 0.5, 0.25]), # Heuristic Weighted-sum
    "heuristic_binary_4_2_1": torch.tensor([4.0, 2.0, 1.0]), # Heuristic Weighted-sum
}


def flatten_pair(w_true, b_mtx):
    w = w_true.detach().cpu().float().flatten()
    b = b_mtx_to_flat(b_mtx)
    return w, b


def pearson_corr(a, b, eps=1e-12):
    a_centered = a - a.mean()
    b_centered = b - b.mean()
    numerator = torch.sum(a_centered * b_centered)
    denominator = torch.sqrt(torch.sum(a_centered ** 2)) * torch.sqrt(torch.sum(b_centered ** 2))
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


def zscore(x, eps=1e-12):
    return (x - x.mean()) / (x.std(unbiased=False) + eps)


def rescale_to_true(w_attack, w_true, eps=1e-12):
    attack_z = zscore(w_attack, eps=eps)
    return attack_z * w_true.std(unbiased=False) + w_true.mean()


def histogram_probs(x, bins):
    counts = torch.histc(x, bins=bins.numel() - 1, min=bins[0].item(), max=bins[-1].item())
    return counts / counts.sum().clamp_min(1)


def kl_divergence(p, q, eps=1e-12):
    p = p + eps
    q = q + eps
    p = p / p.sum()
    q = q / q.sum()
    return torch.sum(p * torch.log(p / q)).item()


def distribution_kl(w_true, w_attack, n_bins=80):
    # Compare shape after z-score normalization, because B-only attack scale is arbitrary.
    true_z = zscore(w_true)
    attack_z = zscore(w_attack)

    combined = torch.cat([true_z, attack_z])
    low = torch.quantile(combined, 0.001).item()
    high = torch.quantile(combined, 0.999).item()
    if low == high:
        low, high = combined.min().item(), combined.max().item()
    bins = torch.linspace(low, high, n_bins + 1)

    p_true = histogram_probs(true_z.clamp(low, high), bins)
    p_attack = histogram_probs(attack_z.clamp(low, high), bins)

    return {
        "kl_true_to_attack_z": kl_divergence(p_true, p_attack),
        "kl_attack_to_true_z": kl_divergence(p_attack, p_true),
    }


def compute_attack(layer_name, w_true, b_flat, attack_name, bit_weights, seed=0):
    bit_weights = bit_weights.to(dtype=b_flat.dtype)
    w_attack = b_flat @ bit_weights
    w_attack_rescaled = rescale_to_true(w_attack, w_true)

    generator = torch.Generator().manual_seed(seed)
    shuffled = w_attack[torch.randperm(w_attack.numel(), generator=generator)]

    kl = distribution_kl(w_true, w_attack)

    return {
        "layer": layer_name,
        "attack": attack_name,
        "bit_weights": ",".join(f"{x:g}" for x in bit_weights.tolist()),
        "n_weights": w_true.numel(),
        "sign_acc": sign_accuracy(w_true, w_attack),
        "pearson": pearson_corr(w_true, w_attack),
        "spearman": spearman_corr(w_true, w_attack),
        "rescaled_mae": (w_attack_rescaled - w_true).abs().mean().item(),
        "rescaled_rmse": torch.sqrt(((w_attack_rescaled - w_true) ** 2).mean()).item(),
        "shuffled_pearson": pearson_corr(w_true, shuffled),
        "shuffled_spearman": spearman_corr(w_true, shuffled),
        **kl,
    }


def save_distribution_plot(w_true, b_flat):
    output_path = ATTACKER_DIR / "weighted_sum_attack_distributions.png"
    fig, axes = plt.subplots(len(ATTACKS), 1, figsize=(9, 3.2 * len(ATTACKS)), constrained_layout=True)
    if len(ATTACKS) == 1:
        axes = [axes]

    true_z = zscore(w_true).numpy()
    for ax, (attack_name, bit_weights) in zip(axes, ATTACKS.items()):
        w_attack = b_flat @ bit_weights.to(dtype=b_flat.dtype)
        attack_z = zscore(w_attack).numpy()
        ax.hist(true_z, bins=80, density=True, alpha=0.55, label="W_true z-score")
        ax.hist(attack_z, bins=80, density=True, alpha=0.55, label="W_attack z-score")
        ax.set_title(attack_name)
        ax.set_xlabel("standardized value")
        ax.set_ylabel("density")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()

    fig.suptitle("Heuristic Weighted-sum Attack: Distribution Comparison")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def main():
    w_true_path = WEIGHT_DIR / "W_true.pth"
    b_only_path = ATTACKER_DIR / "attacker_visible_B_only.pth"

    w_true_state = torch.load(w_true_path, map_location="cpu")
    attacker_visible = torch.load(b_only_path, map_location="cpu")

    rows = []
    all_true = []
    all_b = []

    for layer_name, data in attacker_visible.items():
        w_true, b_flat = flatten_pair(w_true_state[layer_name], data["B_mtx"])
        all_true.append(w_true)
        all_b.append(b_flat)

        for attack_name, bit_weights in ATTACKS.items():
            rows.append(compute_attack(layer_name, w_true, b_flat, attack_name, bit_weights))

    overall_true = torch.cat(all_true)
    overall_b = torch.cat(all_b)
    for attack_name, bit_weights in ATTACKS.items():
        rows.append(compute_attack("OVERALL", overall_true, overall_b, attack_name, bit_weights))

    output_path = ATTACKER_DIR / "weighted_sum_attack_metrics.csv"
    plot_path = save_distribution_plot(overall_true, overall_b)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    print("Heuristic Weighted-sum Attack")
    print("Formula: W_attack = sum_q w_q B_q")
    print()
    print(df.to_string(index=False))
    print()
    print("Saved metrics CSV to:", output_path)
    print("Saved distribution plot to:", plot_path)


if __name__ == "__main__":
    main()
