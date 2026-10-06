from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from bcq_format import b_mtx_to_flat


ROOT = Path(__file__).resolve().parents[1]
WEIGHT_DIR = ROOT / "weight_data"
ATTACKER_DIR = ROOT / "Attacker"

PATTERNS = ["---", "--+", "-+-", "-++", "+--", "+-+", "++-", "+++"]
PATTERN_TO_BITS = {
    pattern: torch.tensor([1.0 if char == "+" else -1.0 for char in pattern])
    for pattern in PATTERNS
}
PATTERN_TO_SCORE = {
    pattern: int((PATTERN_TO_BITS[pattern] * torch.tensor([4.0, 2.0, 1.0])).sum().item())
    for pattern in PATTERNS
}


def flatten_pair(w_true, b_mtx):
    w = w_true.detach().cpu().float().flatten()
    b = b_mtx_to_flat(b_mtx)
    return w, b


def bits_to_pattern(bits):
    return "".join("+" if value > 0 else "-" for value in bits.tolist())


def pattern_masks(b_flat):
    masks = {}
    for pattern, bits in PATTERN_TO_BITS.items():
        masks[pattern] = torch.all(b_flat == bits.to(dtype=b_flat.dtype), dim=1)
    return masks


def pattern_code(b_flat):
    codes = torch.empty(b_flat.shape[0], dtype=torch.long)
    for idx, pattern in enumerate(PATTERNS):
        bits = PATTERN_TO_BITS[pattern].to(dtype=b_flat.dtype)
        codes[torch.all(b_flat == bits, dim=1)] = idx
    return codes


def eta_squared(w_true, codes):
    overall_mean = w_true.mean()
    total_ss = torch.sum((w_true - overall_mean) ** 2)
    between_ss = torch.tensor(0.0)
    for idx in range(len(PATTERNS)):
        values = w_true[codes == idx]
        if values.numel() > 0:
            between_ss += values.numel() * (values.mean() - overall_mean) ** 2
    return (between_ss / total_ss.clamp_min(1e-12)).item()


def compute_layer(layer_name, w_true, b_mtx):
    w_true, b_flat = flatten_pair(w_true, b_mtx)
    masks = pattern_masks(b_flat)
    codes = pattern_code(b_flat)

    rows = []
    means_by_pattern = []
    for pattern in PATTERNS:
        values = w_true[masks[pattern]]
        if values.numel() == 0:
            stats = {
                "count": 0,
                "w_true_mean": float("nan"),
                "w_true_median": float("nan"),
                "w_true_std": float("nan"),
                "w_true_q25": float("nan"),
                "w_true_q75": float("nan"),
                "positive_rate": float("nan"),
            }
        else:
            q25, median, q75 = torch.quantile(values, torch.tensor([0.25, 0.50, 0.75])).tolist()
            stats = {
                "count": values.numel(),
                "w_true_mean": values.mean().item(),
                "w_true_median": median,
                "w_true_std": values.std(unbiased=False).item(),
                "w_true_q25": q25,
                "w_true_q75": q75,
                "positive_rate": (values > 0).float().mean().item(),
            }

        means_by_pattern.append(stats["w_true_mean"])
        rows.append(
            {
                "layer": layer_name,
                "pattern": pattern,
                "weighted_score_4_2_1": PATTERN_TO_SCORE[pattern],
                **stats,
            }
        )

    metrics = {
        "layer": layer_name,
        "n_weights": w_true.numel(),
        "eta_squared_pattern_to_w_true": eta_squared(w_true, codes),
        "monotonic_mean_by_4_2_1_score": all(
            a < b for a, b in zip(means_by_pattern, means_by_pattern[1:])
        ),
    }

    plot_values = {pattern: w_true[masks[pattern]].numpy() for pattern in PATTERNS}
    return rows, metrics, plot_values


def save_boxplot(all_plot_values):
    output_path = ATTACKER_DIR / "pattern_value_level_boxplot.png"
    n_layers = len(all_plot_values)
    fig, axes = plt.subplots(n_layers, 1, figsize=(10, 2.8 * n_layers), constrained_layout=True)
    if n_layers == 1:
        axes = [axes]

    for ax, (layer, values_by_pattern) in zip(axes, all_plot_values.items()):
        data = [values_by_pattern[pattern] for pattern in PATTERNS]
        ax.boxplot(data, tick_labels=PATTERNS, showfliers=False)
        ax.set_title(layer)
        ax.set_xlabel("binary pattern (B1 B2 B3), ordered by 4B1 + 2B2 + B3")
        ax.set_ylabel("W_true")
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Pattern-to-Value-Level Attack: W_true Distribution by Binary Pattern")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def main():
    w_true_path = WEIGHT_DIR / "W_true.pth"
    b_only_path = ATTACKER_DIR / "attacker_visible_B_only.pth"

    w_true_state = torch.load(w_true_path, map_location="cpu")
    attacker_visible = torch.load(b_only_path, map_location="cpu")

    rows = []
    metric_rows = []
    all_plot_values = {}

    all_true = []
    all_b = []

    for layer_name, data in attacker_visible.items():
        layer_rows, layer_metrics, plot_values = compute_layer(
            layer_name, w_true_state[layer_name], data["B_mtx"]
        )
        rows.extend(layer_rows)
        metric_rows.append(layer_metrics)
        all_plot_values[layer_name] = plot_values

        w_true, b_flat = flatten_pair(w_true_state[layer_name], data["B_mtx"])
        all_true.append(w_true)
        all_b.append(b_flat)

    overall_rows, overall_metrics, overall_plot_values = compute_layer(
        "OVERALL", torch.cat(all_true), torch.cat(all_b)
    )
    rows.extend(overall_rows)
    metric_rows.append(overall_metrics)
    all_plot_values["OVERALL"] = overall_plot_values

    stats_path = ATTACKER_DIR / "pattern_value_level_stats.csv"
    metrics_path = ATTACKER_DIR / "pattern_value_level_metrics.csv"
    plot_path = save_boxplot(all_plot_values)

    stats_df = pd.DataFrame(rows)
    metrics_df = pd.DataFrame(metric_rows)
    stats_df.to_csv(stats_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)

    print("Pattern-to-Value-Level Attack")
    print("Each row groups W_true by the exact visible binary pattern B1B2B3.")
    print()
    print(metrics_df.to_string(index=False))
    print()
    print(stats_df[stats_df["layer"] == "OVERALL"].to_string(index=False))
    print()
    print("Saved pattern stats CSV to:", stats_path)
    print("Saved pattern metrics CSV to:", metrics_path)
    print("Saved boxplot to:", plot_path)


if __name__ == "__main__":
    main()
