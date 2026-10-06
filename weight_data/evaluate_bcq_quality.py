from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
WEIGHT_DIR = ROOT / "weight_data"


def flatten_pair(a, b):
    return a.detach().cpu().float().flatten(), b.detach().cpu().float().flatten()


def sign_accuracy(a, b):
    a_sign = torch.sign(a)
    b_sign = torch.sign(b)
    a_sign[a_sign == 0] = 1
    b_sign[b_sign == 0] = 1
    return (a_sign == b_sign).float().mean().item()


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


def compute_metrics(w_true, w_bcq):
    w_true, w_bcq = flatten_pair(w_true, w_bcq)
    diff = w_bcq - w_true

    return {
        "mae": diff.abs().mean().item(), # 平均每個 weight 差多少。
        "rmse": torch.sqrt((diff ** 2).mean()).item(), # 平均每個 weight 差多少，平方後再開根號。 看誤差，但會放大「特別大的錯誤」
        "cosine": torch.nn.functional.cosine_similarity(w_true, w_bcq, dim=0).item(),
        "sign_acc": sign_accuracy(w_true, w_bcq), # 看正負號猜對多少。
        "pearson": pearson_corr(w_true, w_bcq), # 看兩組數值是不是線性相關
        "spearman": spearman_corr(w_true, w_bcq), # 看不直接看數值，而是看排序像不像，整體排名有沒有保住
    }


def print_metrics(name, metrics):
    print(name)
    print(f"  MAE:                 {metrics['mae']:.8f}")
    print(f"  RMSE:                {metrics['rmse']:.8f}")
    print(f"  Cosine similarity:   {metrics['cosine']:.8f}")
    print(f"  Sign accuracy:       {metrics['sign_acc']:.8f}")
    print(f"  Pearson correlation: {metrics['pearson']:.8f}")
    print(f"  Spearman correlation:{metrics['spearman']:.8f}")
    print()


def save_metrics_csv(rows):
    output_path = WEIGHT_DIR / "bcq_quality_metrics.csv"
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return output_path


def save_metrics_plot(rows):
    output_path = WEIGHT_DIR / "bcq_quality_metrics.png"
    df = pd.DataFrame(rows)

    layer_df = df[df["layer"] != "OVERALL"].copy()
    layers = layer_df["layer"].tolist()

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)

    similarity_metrics = ["cosine", "sign_acc", "pearson", "spearman"]

    x = range(len(layers))
    width = 0.35

    axes[0].bar([i - width / 2 for i in x], layer_df["mae"], width=width, label="MAE")
    axes[0].bar([i + width / 2 for i in x], layer_df["rmse"], width=width, label="RMSE")
    axes[0].set_title("BCQ Reconstruction Error: W_true vs W_BCQ")
    axes[0].set_ylabel("Lower is better")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(layers, rotation=25, ha="right")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    for metric in similarity_metrics:
        axes[1].plot(layers, layer_df[metric], marker="o", label=metric)

    axes[1].set_title("BCQ Reconstruction Similarity: W_true vs W_BCQ")
    axes[1].set_ylabel("Higher is better")
    axes[1].set_ylim(0, 1.05)
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)

    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def main():
    bcq_path = WEIGHT_DIR / "W_BCQ_result.pth"
    bcq_result = torch.load(bcq_path, map_location="cpu")

    all_true = []
    all_bcq = []
    rows = []

    print("Compare W_true vs W_BCQ")
    print("Source:", bcq_path)
    print()

    for name, data in bcq_result.items():
        w_true = data["W_true"]
        w_bcq = data["W_BCQ"]

        metrics = compute_metrics(w_true, w_bcq)
        print_metrics(name, metrics)
        rows.append({"layer": name, **metrics})

        all_true.append(w_true.flatten())
        all_bcq.append(w_bcq.flatten())

    overall_true = torch.cat(all_true)
    overall_bcq = torch.cat(all_bcq)
    overall_metrics = compute_metrics(overall_true, overall_bcq)

    print_metrics("OVERALL", overall_metrics)
    rows.append({"layer": "OVERALL", **overall_metrics})

    csv_path = save_metrics_csv(rows)
    plot_path = save_metrics_plot(rows)

    print("Saved metrics CSV to:", csv_path)
    print("Saved metrics plot to:", plot_path)


if __name__ == "__main__":
    main()
