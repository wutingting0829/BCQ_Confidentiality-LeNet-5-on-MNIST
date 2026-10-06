# 這個 attack 並不是要精準還原原始 weight，
# 而是測試在不知道 alpha 的情況下，
# 單純把 binary matrices 相加，是否仍然能得到和 W_true 有規律關係的 coarse value level。
# Pearson：看數值線性像不像
# Spearman：看大小排序像不像

# 流程是：
# 1. 讀入 weight_data/W_true.pth，這是 ground truth，只用來評分。
# 2. 讀入 Attacker/attacker_visible_B_only.pth，這是攻擊者真正可見的資料。
# 3. 把 B_mtx 攤平成每個 weight 對應一組 [B1, B2, B3]。
# 4. 計算 W_attack = sum(Bq)。
# 5. 比較 W_attack 和 W_true 的關係：sign_acc：正負號猜對多少
#    pearson：線性相關
#    spearman：排序相關
#    value_bin_acc：把真實權重分成四個 quantile，看攻擊 level 是否對到
#    monotonic_mean：看 -3 -> -1 -> +1 -> +3 時，真實權重平均值是否也單調上升

# 所以它測的是：即使攻擊者不知道 alpha/z，只看 binary patterns，加總後是否仍然洩漏權重大小與符號資訊

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from bcq_format import b_mtx_to_flat


ROOT = Path(__file__).resolve().parents[1]
WEIGHT_DIR = ROOT / "weight_data"
ATTACKER_DIR = ROOT / "Attacker"
LEVELS = [-3, -1, 1, 3]


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


def make_true_bins(w_true):
    q25, q50, q75 = torch.quantile(w_true, torch.tensor([0.25, 0.50, 0.75]))
    bins = torch.empty_like(w_true, dtype=torch.long)
    bins[w_true <= q25] = 0
    bins[(w_true > q25) & (w_true <= q50)] = 1
    bins[(w_true > q50) & (w_true <= q75)] = 2
    bins[w_true > q75] = 3
    return bins


def attack_level_to_bin(w_attack):
    mapping = {-3: 0, -1: 1, 1: 2, 3: 3}
    bins = torch.empty_like(w_attack, dtype=torch.long)
    for level, bin_id in mapping.items():
        bins[w_attack == level] = bin_id
    return bins


def confusion_matrix(true_bins, attack_bins, n=4):
    matrix = torch.zeros((n, n), dtype=torch.long)
    for t, a in zip(true_bins, attack_bins):
        matrix[t, a] += 1
    return matrix


def compute_layer(layer_name, w_true, b_mtx, seed=0):
    w_true, b_flat = flatten_pair(w_true, b_mtx)
    w_attack = b_flat.sum(dim=1)

    generator = torch.Generator().manual_seed(seed)
    shuffled = w_attack[torch.randperm(w_attack.numel(), generator=generator)]

    true_bins = make_true_bins(w_true)
    attack_bins = attack_level_to_bin(w_attack)
    bin_acc = (true_bins == attack_bins).float().mean().item()

    metrics = {
        "layer": layer_name,
        "n_weights": w_true.numel(),
        "sign_acc": sign_accuracy(w_true, w_attack),
        "pearson": pearson_corr(w_true, w_attack),
        "spearman": spearman_corr(w_true, w_attack),
        "shuffled_pearson": pearson_corr(w_true, shuffled),
        "shuffled_spearman": spearman_corr(w_true, shuffled),
        "value_bin_acc": bin_acc,
        "monotonic_mean": True,
    }

    group_rows = []
    means = []
    for level in LEVELS:
        values = w_true[w_attack == level]
        if values.numel() == 0:
            mean = float("nan")
            std = float("nan")
            q25 = float("nan")
            median = float("nan")
            q75 = float("nan")
        else:
            mean = values.mean().item()
            std = values.std(unbiased=False).item()
            q25, median, q75 = torch.quantile(values, torch.tensor([0.25, 0.50, 0.75])).tolist()

        means.append(mean)
        group_rows.append(
            {
                "layer": layer_name,
                "attack_level": level,
                "count": values.numel(),
                "w_true_mean": mean,
                "w_true_std": std,
                "w_true_q25": q25,
                "w_true_median": median,
                "w_true_q75": q75,
            }
        )

    metrics["monotonic_mean"] = all(a < b for a, b in zip(means, means[1:]))

    conf = confusion_matrix(true_bins, attack_bins)
    conf_rows = []
    true_labels = ["true_strong_neg", "true_weak_neg", "true_weak_pos", "true_strong_pos"]
    attack_labels = ["attack_-3", "attack_-1", "attack_+1", "attack_+3"]
    for i, true_label in enumerate(true_labels):
        row_total = conf[i].sum().item()
        for j, attack_label in enumerate(attack_labels):
            count = conf[i, j].item()
            conf_rows.append(
                {
                    "layer": layer_name,
                    "true_bin": true_label,
                    "attack_bin": attack_label,
                    "count": count,
                    "row_percent": count / row_total if row_total else 0.0,
                }
            )

    plot_values = {level: w_true[w_attack == level].numpy() for level in LEVELS}
    return metrics, group_rows, conf_rows, plot_values


def save_boxplot(all_plot_values):
    output_path = ATTACKER_DIR / "value_leakage_attack_boxplot.png"
    n_layers = len(all_plot_values)
    fig, axes = plt.subplots(n_layers, 1, figsize=(9, 2.7 * n_layers), constrained_layout=True)
    if n_layers == 1:
        axes = [axes]

    for ax, (layer, values_by_level) in zip(axes, all_plot_values.items()):
        data = [values_by_level[level] for level in LEVELS]
        ax.boxplot(data, tick_labels=[str(level) for level in LEVELS], showfliers=False)
        ax.set_title(layer)
        ax.set_xlabel("W_attack = sum(Bq)")
        ax.set_ylabel("W_true")
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Equal-sum Value Attack: W_true Distribution by Attack Level")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def main():
    w_true_path = WEIGHT_DIR / "W_true.pth"
    b_only_path = ATTACKER_DIR / "attacker_visible_B_only.pth"

    w_true = torch.load(w_true_path, map_location="cpu")
    attacker_visible = torch.load(b_only_path, map_location="cpu")

    metric_rows = []
    group_rows = []
    conf_rows = []
    all_plot_values = {}

    all_true = []
    all_b = []

    for name, data in attacker_visible.items():
        metrics, groups, conf, plot_values = compute_layer(name, w_true[name], data["B_mtx"])
        metric_rows.append(metrics)
        group_rows.extend(groups)
        conf_rows.extend(conf)
        all_plot_values[name] = plot_values

        layer_true, layer_b = flatten_pair(w_true[name], data["B_mtx"])
        all_true.append(layer_true)
        all_b.append(layer_b)

    overall_true = torch.cat(all_true)
    overall_b = torch.cat(all_b).reshape(-1, 3)
    overall_metrics, overall_groups, overall_conf, overall_plot_values = compute_layer(
        "OVERALL", overall_true, overall_b
    )
    metric_rows.append(overall_metrics)
    group_rows.extend(overall_groups)
    conf_rows.extend(overall_conf)
    all_plot_values["OVERALL"] = overall_plot_values

    metrics_path = ATTACKER_DIR / "value_leakage_attack_metrics.csv"
    groups_path = ATTACKER_DIR / "value_leakage_attack_groups.csv"
    conf_path = ATTACKER_DIR / "value_leakage_attack_confusion.csv"
    plot_path = save_boxplot(all_plot_values)

    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)
    pd.DataFrame(group_rows).to_csv(groups_path, index=False)
    pd.DataFrame(conf_rows).to_csv(conf_path, index=False)

    print("Equal-sum Value Attack")
    print("Formula: W_attack = sum_q B_q")
    print()
    print(pd.DataFrame(metric_rows).to_string(index=False))
    print()
    print("Saved metrics CSV to:", metrics_path)
    print("Saved grouped W_true stats CSV to:", groups_path)
    print("Saved value-bin confusion CSV to:", conf_path)
    print("Saved boxplot to:", plot_path)


if __name__ == "__main__":
    main()
