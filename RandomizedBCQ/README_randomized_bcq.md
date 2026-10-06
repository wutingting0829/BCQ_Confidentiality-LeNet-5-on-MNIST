# Randomized Residual-sign BCQ

This folder separates the randomized residual-sign BCQ experiment into three clear steps.

Attacker-visible plaintext data is `B_mtx_public`, not the true aligned residual-sign `B_mtx_true_secret`.
The defender reconstructs by using the secret sign mask and secret bit-plane permutation before applying alpha and z.

## Step 1: Create Randomized BCQ Model Weights

Run:

```bash
python RandomizedBCQ/01_create_randomized_bcq_weights.py
```

Outputs:

- `01_randomized_BCQ_full_secret_result.pth`: defender-side BCQ result, including secrets.
- `01_attacker_visible_randomized_B_only.pth`: attacker-visible plaintext randomized B only.
- `01_leaky_true_B_control.pth`: insecure control with true aligned B.
- `01_randomized_bcq_creation_summary.csv`: layer shapes and reconstruction check.

## Step 2: Weighted-sum Attack

Run:

```bash
python RandomizedBCQ/02_weighted_sum_attack.py
```

Outputs:

- `02_weighted_sum_attack_weights.pth`: attacker-created reconstructed weights.
- `02_weighted_sum_attack_metrics.csv`: sign/correlation/rank leakage metrics.
- `02_weighted_sum_attack_distributions.png`: distribution comparison plot.

## Step 3: Accuracy Comparison

Run:

```bash
python RandomizedBCQ/03_evaluate_attack_accuracy.py
```

Output:

- `03_attack_reconstruction_accuracy.csv`: MNIST accuracy after loading attacker-created weights into LeNet-5.

## One-command Legacy Script

The original all-in-one script is still available:

```bash
python RandomizedBCQ/randomized_residual_sign_bcq_experiment.py
```
