"""
Step 3: load attacker-created weights into LeNet-5 and evaluate MNIST accuracy.

Outputs:
- 03_attack_reconstruction_accuracy.csv
"""

import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from randomized_bcq_common import (
    ATTACKS,
    ATTACK_WEIGHTS_PATH,
    OUT_DIR,
    ROOT,
    WEIGHT_DIR,
    WEIGHT_NAMES,
    make_random_bias_overrides,
    with_overrides,
)

sys.path.insert(0, str(ROOT))
from model import Model  # noqa: E402


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


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_loader = load_mnist_test_loader()

    w_true_state = torch.load(WEIGHT_DIR / "W_true.pth", map_location="cpu")
    attack_weights = torch.load(ATTACK_WEIGHTS_PATH, map_location="cpu")
    random_bias_overrides = make_random_bias_overrides(w_true_state)

    rows = [
        {
            "attacker_view": "baseline",
            "attack": "W_true",
            "scale": "true",
            "bias": "trained",
            "accuracy": evaluate_accuracy(w_true_state, test_loader, device),
        },
    ]

    view_name = "randomized_public_B"
    for attack_name in ATTACKS:
        for scale_name in ["raw", "oracle_rescaled_to_W_true_mean_std"]:
            overrides = {
                name: attack_weights[view_name][name][attack_name][scale_name]
                for name in WEIGHT_NAMES
            }
            for bias_name, bias_overrides in [("trained", {}), ("random", random_bias_overrides)]:
                rows.append(
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

    accuracy_df = pd.DataFrame(rows)
    accuracy_path = OUT_DIR / "03_attack_reconstruction_accuracy.csv"
    accuracy_df.to_csv(accuracy_path, index=False)

    print("Step 3 complete: attacker weights evaluated in LeNet-5")
    print("Saved:", accuracy_path)
    print()
    print(accuracy_df.to_string(index=False))


if __name__ == "__main__":
    main()
