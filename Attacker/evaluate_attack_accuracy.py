"""
Evaluate whether reconstructed weights still behave like a useful MNIST model.

This script is a functional test, not another weight-similarity metric:
it loads candidate weight tensors into the real LeNet-5 architecture and runs
the MNIST test set. If an attacker-created W_attack gets high accuracy here,
then the public B_mtx leaks enough structure to recover some model behavior.

Important threat-model distinction:
- W_true is loaded here only by the experiment evaluator, not by the attacker.
- B-only attacks construct W_attack from plaintext B_mtx only.
- The "rescaled" rows are oracle/best-case rows because they use W_true mean/std.

For B-only attacks, this script now reports two bias settings:
- trained bias: the old behavior, useful as a lenient comparison.
- random bias: stricter setting for the threat model where bias is protected too.

Rows reported:
  W_true (baseline)                          -- original trained checkpoint
  W_BCQ (full reconstruction, has alpha + z)  -- defender-side reconstruction using B_mtx + alpha + z
  <attack> (raw scale, trained bias)          -- B-only W_attack, but original trained bias remains
  <attack> (raw scale, random bias)           -- B-only W_attack with randomized bias tensors
  <attack> (rescaled, trained/random bias)    -- oracle upper bound; not available to a real B-only attacker
"""

import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import mnist
from torchvision.transforms import ToTensor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model import Model  # noqa: E402
from bcq_format import weighted_sum_to_shape  # noqa: E402

WEIGHT_DIR = ROOT / "weight_data"
ATTACKER_DIR = ROOT / "Attacker"

WEIGHT_NAMES = ["conv1.weight", "conv2.weight", "fc1.weight", "fc2.weight", "fc3.weight"]
BIAS_NAMES = ["conv1.bias", "conv2.bias", "fc1.bias", "fc2.bias", "fc3.bias"]
RANDOM_BIAS_SEED = 0

ATTACKS = {
    "equal_sum_1_1_1": torch.tensor([1.0, 1.0, 1.0]),
    "heuristic_half_1_0.5_0.25": torch.tensor([1.0, 0.5, 0.25]),
    "heuristic_binary_4_2_1": torch.tensor([4.0, 2.0, 1.0]),
}


def zscore(x, eps=1e-12):
    # Standardize a tensor so it has roughly zero mean and unit variance.
    return (x - x.mean()) / (x.std(unbiased=False) + eps)


def rescale_to_true(w_attack, w_true, eps=1e-12):
    # Oracle rescale for analysis only: a real attacker should not know W_true's
    # mean/std if W_true is protected.
    return zscore(w_attack, eps=eps) * w_true.std(unbiased=False) + w_true.mean()


def with_weight_overrides(base_state, overrides):
    # Start from the original state dict so untouched parameters, especially
    # biases, remain the trained values. Then replace selected weight tensors.
    state = dict(base_state)
    state.update(overrides)
    return state


def make_random_bias_overrides(base_state, seed=RANDOM_BIAS_SEED):
    # A stricter attacker setting: bias is also protected, so the attack model
    # receives random bias values instead of the trained bias. Matching each
    # tensor's mean/std keeps the scale plausible while removing learned values.
    generator = torch.Generator().manual_seed(seed)
    overrides = {}
    for name in BIAS_NAMES:
        bias = base_state[name]
        random_bias = torch.randn(bias.shape, generator=generator, dtype=bias.dtype)
        random_bias = random_bias * bias.std(unbiased=False) + bias.mean()
        overrides[name] = random_bias
    return overrides


def evaluate_accuracy(state_dict, test_loader, device):
    # Load a candidate state dict into a fresh model and measure test accuracy.
    model = Model().to(device)
    model.load_state_dict(state_dict)
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x.float()).argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.shape[0]
    return correct / total


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # W_true.pth comes from weight_data/load_w_true.py, which extracts the state
    # dict from the best local models/mnist_*.pkl checkpoint.
    w_true_state = torch.load(WEIGHT_DIR / "W_true.pth", map_location="cpu")

    # Defender-only full BCQ data. This contains protected magnitude parameters
    # alpha and z, so it is not attacker-visible.
    bcq_result = torch.load(WEIGHT_DIR / "W_BCQ_result.pth", map_location="cpu")

    # Attacker-visible file: only plaintext B_mtx, no alpha, no z, no W_true.
    attacker_visible = torch.load(ATTACKER_DIR / "attacker_visible_B_only.pth", map_location="cpu")

    test_dataset = mnist.MNIST(root=str(ROOT / "test"), train=False, transform=ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=256)

    rows = []
    random_bias_overrides = make_random_bias_overrides(w_true_state)

    # Baseline: reload the original trained model weights and re-evaluate them.
    # The result can differ slightly from the rounded checkpoint filename.
    acc = evaluate_accuracy(w_true_state, test_loader, device)
    rows.append({"attack": "W_true (baseline)", "accuracy": acc})

    # Defender-side BCQ reconstruction: uses alpha and z, so this is a quantized
    # model quality check, not a B-only attacker result.
    w_bcq_overrides = {name: bcq_result[name]["W_BCQ"] for name in WEIGHT_NAMES}
    acc = evaluate_accuracy(with_weight_overrides(w_true_state, w_bcq_overrides), test_loader, device)
    rows.append({"attack": "W_BCQ (full reconstruction, has alpha + z)", "accuracy": acc})

    for attack_name, bit_weights in ATTACKS.items():
        overrides_raw = {}
        overrides_rescaled = {}
        for name in WEIGHT_NAMES:
            b_mtx = attacker_visible[name]["B_mtx"]  # FIGLUT format: (num_bits, m, n)
            w_true = w_true_state[name]

            # Realistic B-only reconstruction: no alpha, no z, no W_true values.
            w_attack = weighted_sum_to_shape(b_mtx, bit_weights, w_true.shape)

            overrides_raw[name] = w_attack

            # Oracle-only comparison: this uses W_true statistics, so it should
            # be interpreted as an upper bound on leakage, not a real attack.
            overrides_rescaled[name] = rescale_to_true(w_attack, w_true)

        acc_raw = evaluate_accuracy(with_weight_overrides(w_true_state, overrides_raw), test_loader, device)
        rows.append({"attack": f"{attack_name} (raw scale, trained bias)", "accuracy": acc_raw})

        acc_raw_random_bias = evaluate_accuracy(
            with_weight_overrides(w_true_state, {**overrides_raw, **random_bias_overrides}),
            test_loader,
            device,
        )
        rows.append({"attack": f"{attack_name} (raw scale, random bias)", "accuracy": acc_raw_random_bias})

        acc_rescaled = evaluate_accuracy(with_weight_overrides(w_true_state, overrides_rescaled), test_loader, device)
        rows.append({"attack": f"{attack_name} (rescaled to W_true mean/std, trained bias)", "accuracy": acc_rescaled})

        acc_rescaled_random_bias = evaluate_accuracy(
            with_weight_overrides(w_true_state, {**overrides_rescaled, **random_bias_overrides}),
            test_loader,
            device,
        )
        rows.append(
            {
                "attack": f"{attack_name} (rescaled to W_true mean/std, random bias)",
                "accuracy": acc_rescaled_random_bias,
            }
        )

    df = pd.DataFrame(rows)
    out_path = ATTACKER_DIR / "attack_reconstruction_accuracy.csv"
    df.to_csv(out_path, index=False)

    print("Reconstructed-weight MNIST test accuracy")
    print()
    print(df.to_string(index=False))
    print()
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
