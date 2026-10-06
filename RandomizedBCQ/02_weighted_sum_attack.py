"""
Step 2: run Weighted-sum Attack on randomized public B_mtx.

Outputs:
- 02_weighted_sum_attack_weights.pth
- 02_weighted_sum_attack_metrics.csv
- 02_weighted_sum_attack_distributions.png
"""

import matplotlib.pyplot as plt
import pandas as pd
import torch

from randomized_bcq_common import (
    ATTACKER_VISIBLE_PATH,
    ATTACKS,
    ATTACK_WEIGHTS_PATH,
    LEAKY_CONTROL_PATH,
    OUT_DIR,
    WEIGHT_DIR,
    WEIGHT_NAMES,
    b_mtx_to_flat,
    pearson_corr,
    rescale_to_true,
    sign_accuracy,
    spearman_corr,
    weighted_sum_from_b,
    zscore,
)


def compute_attack_metrics(layer_name, attacker_view, w_true, b_mtx, attack_name, bit_weights):
    w_flat = w_true.detach().cpu().float().flatten()
    b_flat = b_mtx_to_flat(b_mtx)
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
def save_distribution_plot1(w_true_all, leaky_b_all, randomized_b_all):
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.4), constrained_layout=True)
    true_z = zscore(w_true_all).numpy()
    attack_name = "equal_sum_1_1_1"
    bit_weights = ATTACKS[attack_name]

    for ax, (view_name, b_all) in zip(
        axes,
        [
            ("leaky_true_B_control", leaky_b_all),
            ("randomized_public_B", randomized_b_all),
        ],
    ):
        attack_z = zscore(b_all @ bit_weights.to(dtype=b_all.dtype)).numpy()
        ax.hist(true_z, bins=80, density=True, alpha=0.55, label="W_true z-score")
        ax.hist(attack_z, bins=80, density=True, alpha=0.55, label="W_attack z-score")
        ax.set_title(f"{view_name}: {attack_name}")
        ax.set_xlabel("standardized value")
        ax.set_ylabel("density")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()

    output_path = OUT_DIR / "02_weighted_sum_attack_distribution1.png"
    fig.suptitle("Randomized Residual-sign BCQ: Weighted-sum Attack")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path

def save_distribution_plot2(w_true_all, leaky_b_all, randomized_b_all):
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.4), constrained_layout=True)
    true_z = zscore(w_true_all).numpy()
    attack_name = "heuristic_binary_4_2_1"
    bit_weights = ATTACKS[attack_name]

    for ax, (view_name, b_all) in zip(
        axes,
        [
            ("leaky_true_B_control", leaky_b_all),
            ("randomized_public_B", randomized_b_all),
        ],
    ):
        attack_z = zscore(b_all @ bit_weights.to(dtype=b_all.dtype)).numpy()
        ax.hist(true_z, bins=80, density=True, alpha=0.55, label="W_true z-score")
        ax.hist(attack_z, bins=80, density=True, alpha=0.55, label="W_attack z-score")
        ax.set_title(f"{view_name}: {attack_name}")
        ax.set_xlabel("standardized value")
        ax.set_ylabel("density")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()

    output_path = OUT_DIR / "02_weighted_sum_attack_distributions2.png"
    fig.suptitle("Randomized Residual-sign BCQ: Weighted-sum Attack")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path

def save_distribution_plot(w_true_all, leaky_b_all, randomized_b_all):
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.4), constrained_layout=True)
    true_z = zscore(w_true_all).numpy()
    bit_weights = ATTACKS["heuristic_half_1_0.5_0.25"]

    for ax, (view_name, b_all) in zip(
        axes,
        [
            ("leaky_true_B_control", leaky_b_all),
            ("randomized_public_B", randomized_b_all),
        ],
    ):
        attack_z = zscore(b_all @ bit_weights.to(dtype=b_all.dtype)).numpy()
        ax.hist(true_z, bins=80, density=True, alpha=0.55, label="W_true z-score")
        ax.hist(attack_z, bins=80, density=True, alpha=0.55, label="W_attack z-score")
        ax.set_title(f"{view_name}: heuristic_half_1_0.5_0.25")
        ax.set_xlabel("standardized value")
        ax.set_ylabel("density")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()

    output_path = OUT_DIR / "02_weighted_sum_attack_distributions.png"
    fig.suptitle("Randomized Residual-sign BCQ: Weighted-sum Attack")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def main():
    w_true_state = torch.load(WEIGHT_DIR / "W_true.pth", map_location="cpu")
    attacker_visible = torch.load(ATTACKER_VISIBLE_PATH, map_location="cpu")
    leaky_true_b = torch.load(LEAKY_CONTROL_PATH, map_location="cpu")

    attack_weights = {"leaky_true_B_control": {}, "randomized_public_B": {}}
    metric_rows = []
    all_true = []
    all_randomized_b = []
    all_leaky_b = []

    for name in WEIGHT_NAMES:
        all_true.append(w_true_state[name].detach().cpu().float().flatten())
        all_randomized_b.append(b_mtx_to_flat(attacker_visible[name]["B_mtx"]))
        all_leaky_b.append(b_mtx_to_flat(leaky_true_b[name]["B_mtx"]))

        for view_name, b_source in [
            ("leaky_true_B_control", leaky_true_b),
            ("randomized_public_B", attacker_visible),
        ]:
            attack_weights[view_name][name] = {}
            for attack_name, bit_weights in ATTACKS.items():
                w_attack = weighted_sum_from_b(b_source[name]["B_mtx"], bit_weights, w_true_state[name].shape)
                attack_weights[view_name][name][attack_name] = {
                    "raw": w_attack,
                    "oracle_rescaled_to_W_true_mean_std": rescale_to_true(w_attack, w_true_state[name]),
                }
                metric_rows.append(
                    compute_attack_metrics(
                        name,
                        view_name,
                        w_true_state[name],
                        b_source[name]["B_mtx"],
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

    torch.save(attack_weights, ATTACK_WEIGHTS_PATH)

    metrics_df = pd.DataFrame(metric_rows)
    metrics_path = OUT_DIR / "02_weighted_sum_attack_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    plot_path = save_distribution_plot(overall_true, overall_leaky_b, overall_randomized_b)
    plot_path1 = save_distribution_plot1(overall_true, overall_leaky_b, overall_randomized_b)
    plot_path2 = save_distribution_plot2(overall_true, overall_leaky_b, overall_randomized_b)

    print("Step 2 complete: weighted-sum attack weights created")
    print("Saved:", ATTACK_WEIGHTS_PATH)
    print("Saved:", metrics_path)
    print("Saved:", plot_path1)
    print("Saved:", plot_path)
    print("Saved:", plot_path2)

    print()
    print(metrics_df[metrics_df["layer"] == "OVERALL"].to_string(index=False))


if __name__ == "__main__":
    main()
