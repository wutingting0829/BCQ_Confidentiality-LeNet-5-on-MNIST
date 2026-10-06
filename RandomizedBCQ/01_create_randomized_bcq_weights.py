"""
Step 1: create randomized residual-sign BCQ model weights.

Outputs:
- 01_randomized_BCQ_full_secret_result.pth
- 01_attacker_visible_randomized_B_only.pth
- 01_leaky_true_B_control.pth
- 01_randomized_bcq_creation_summary.csv
"""

import pandas as pd
import torch

from randomized_bcq_common import (
    ATTACKER_VISIBLE_PATH,
    FULL_SECRET_PATH,
    LEAKY_CONTROL_PATH,
    OUT_DIR,
    WEIGHT_DIR,
    WEIGHT_NAMES,
    bcq_quantize,
    protect_b_mtx,
    reconstruct_bcq_from_public,
)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    w_true_state = torch.load(WEIGHT_DIR / "W_true.pth", map_location="cpu")

    full_result = {}
    attacker_visible = {}
    leaky_true_b = {}
    rows = []

    for name in WEIGHT_NAMES:
        b_true, alpha, z, w_bcq_original = bcq_quantize(w_true_state[name])
        b_public, secret_sign_mask, secret_plane_perm = protect_b_mtx(b_true, name)
        w_bcq_recovered = reconstruct_bcq_from_public(
            b_public,
            alpha,
            z,
            secret_sign_mask,
            secret_plane_perm,
            w_true_state[name].shape,
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

        rows.append(
            {
                "layer": name,
                "W_true_shape": tuple(w_true_state[name].shape),
                "B_mtx_public_shape": tuple(b_public.shape),
                "alpha_shape": tuple(alpha.shape),
                "z_shape": tuple(z.shape),
                "secret_plane_perm": ",".join(str(int(x)) for x in secret_plane_perm.tolist()),
                "max_reconstruction_diff": (w_bcq_recovered - w_bcq_original).abs().max().item(),
            }
        )

    torch.save(full_result, FULL_SECRET_PATH)
    torch.save(attacker_visible, ATTACKER_VISIBLE_PATH)
    torch.save(leaky_true_b, LEAKY_CONTROL_PATH)

    summary_path = OUT_DIR / "01_randomized_bcq_creation_summary.csv"
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(summary_path, index=False)

    print("Step 1 complete: randomized residual-sign BCQ weights created")
    print("Saved:", FULL_SECRET_PATH)
    print("Saved:", ATTACKER_VISIBLE_PATH)
    print("Saved:", LEAKY_CONTROL_PATH)
    print("Saved:", summary_path)
    print()
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
