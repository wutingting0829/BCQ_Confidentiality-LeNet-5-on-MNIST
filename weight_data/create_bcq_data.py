from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
WEIGHT_DIR = ROOT / "weight_data"
ATTACKER_DIR = ROOT / "Attacker"
ATTACKER_DIR.mkdir(exist_ok=True)

W_true = torch.load(WEIGHT_DIR / "W_true.pth", map_location="cpu")

weight_names = [
    "conv1.weight",
    "conv2.weight",
    "fc1.weight",
    "fc2.weight",
    "fc3.weight",
]


# FIGLUT Section II-B style BCQ:
# W_BCQ = sum_k(alpha_k * B_k) + z
#
# z 是 per-row offset（每個 output unit 一個平均值）。先對 W_true 做中心化，
# 再對中心化後的 residual 執行 K-bit binary residual decomposition。


def bcq_quantize(weight, num_bits=3):
    original_shape = weight.shape
    m = original_shape[0]
    weight_flat = weight.reshape(m, -1).clone()  # (m, n): row i = one output unit's weights
    z = weight_flat.mean(dim=1, keepdim=True)    # (m, 1)
    residual = weight_flat - z

    B_list = []
    alpha_list = []
    W_BCQ_centered_flat = torch.zeros_like(residual)

    for _ in range(num_bits):
        B = torch.sign(residual)

        # 避免 sign(0) 變成 0，Binary Matrix 應該只有 +1 / -1
        B[B == 0] = 1

        alpha = residual.abs().mean(dim=1, keepdim=True)  # per-row alpha, shape (m, 1)

        B_list.append(B)
        alpha_list.append(alpha)

        W_BCQ_centered_flat += alpha * B
        residual = residual - alpha * B

    B_mtx = torch.stack(B_list, dim=0)      # (num_bits, m, n)
    alpha = torch.stack(alpha_list, dim=0)  # (num_bits, m, 1)

    W_BCQ = (W_BCQ_centered_flat + z).reshape(original_shape)

    return B_mtx, alpha, z, W_BCQ


# 每一個 output unit（row）、每一個 bit-plane 各自一個 alpha, Q-bits先給3
bcq_result = {}
attacker_visible = {}

num_bits = 3

for name in weight_names:
    weight = W_true[name]

    B_mtx, alpha, z, W_BCQ = bcq_quantize(weight, num_bits=num_bits)

    bcq_result[name] = {
        "W_true": weight,
        "B_mtx": B_mtx,
        "alpha": alpha,
        "z": z,
        "W_BCQ": W_BCQ,
    }

    attacker_visible[name] = {
        "B_mtx": B_mtx,
    }

    print(name)
    print("W_true shape:", weight.shape)
    print("B_mtx shape:", B_mtx.shape)
    print("alpha shape:", alpha.shape)
    print("z shape:", z.shape)
    print("W_BCQ shape:", W_BCQ.shape)
    print()

torch.save(bcq_result, WEIGHT_DIR / "W_BCQ_result.pth")
torch.save(attacker_visible, ATTACKER_DIR / "attacker_visible_B_only.pth")

print("Saved full BCQ result to:", WEIGHT_DIR / "W_BCQ_result.pth")
print("Saved attacker-visible B only to:", ATTACKER_DIR / "attacker_visible_B_only.pth")
