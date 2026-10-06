# What this is

A minimal PyTorch LeNet-5 implementation trained on MNIST (upstream: ChawDoe/LeNet5-MNIST-PyTorch), extended locally with a two-folder pipeline that studies information leakage from BCQ (Binary-Coded Quantization) weight representations — i.e., given only the binary sign matrices of a quantized model (not the scale factors), how well can an attacker reconstruct the original weight values?
- `weight_data/` — defender-side: loads the trained model, produces the BCQ quantization, evaluates quantization fidelity.
- `Attacker/` — attacker-side: everything that only uses the attacker-visible binary matrices (no alpha) to try to reconstruct `W_true`.

## Commands

Train the model (also produces checkpoints used by the weight-leakage pipeline):
```
python3 train.py
```
Trains for up to 100 epochs, evaluates on the MNIST test set each epoch, and saves every epoch's checkpoint to `models/mnist_<test_acc>.pkl` (e.g. `models/mnist_0.890.pkl`). Training stops early once test accuracy changes by less than `1e-4` between epochs. Uses CUDA automatically if available, otherwise CPU. `models/` is gitignored — checkpoints are local artifacts, not committed.

No test suite, linter, or requirements.txt exists in this repo. There is no build step (pure Python/PyTorch scripts run directly).

### Weight-leakage attack pipeline (`weight_data/` + `Attacker/`)

Run in this order — each script consumes the previous script's output:

```
python3 weight_data/load_w_true.py                    # 1. extract W_true from the best checkpoint in models/
python3 weight_data/create_bcq_data.py                 # 2. BCQ-quantize W_true; writes W_BCQ_result.pth to weight_data/, attacker_visible_B_only.pth to Attacker/
python3 weight_data/evaluate_bcq_quality.py             # 3. compare W_true vs full BCQ reconstruction (W_BCQ) — defender-side sanity check, not an attack
python3 Attacker/evaluate_value_leakage_attack.py       # 4. equal-sum attack: W_attack = sum_q(B_q), only B_mtx, no alpha
python3 Attacker/evaluate_weighted_sum_attack.py        # 5. heuristic weighted-sum attacks: W_attack = sum_q(w_q * B_q) for several guessed bit-plane weightings
python3 Attacker/evaluate_pattern_value_level_attack.py  # 6. pattern-to-value-level attack: group W_true by exact 3-bit pattern B1B2B3
python3 Attacker/evaluate_attack_accuracy.py             # 7. loads W_BCQ / W_attack back into Model, measures real MNIST test accuracy
```

Every script resolves `ROOT = Path(__file__).resolve().parents[1]` and then references `ROOT / "weight_data"` and/or `ROOT / "Attacker"` explicitly — both folders sit one level under repo root, so this works regardless of which folder the script lives in. No CLI args; must be run with `ROOT` resolving to the repo root.

`load_w_true.py` picks the checkpoint with the highest test accuracy by parsing the accuracy out of the filename (`models/mnist_*.pkl`, sorted by the trailing float), so `train.py` must have been run first.

## Architecture

### Model (`model.py`)
`Model` (LeNet-5 variant): conv1(1→6,5x5) → ReLU → maxpool → conv2(6→16,5x5) → ReLU → maxpool → fc1(256→120) → ReLU → fc2(120→84) → ReLU → fc3(84→10) → ReLU. Note this repo uses MaxPool+ReLU instead of the original paper's AvgPool+Sigmoid. State dict keys used throughout `weight_data/` scripts: `conv1.weight`, `conv2.weight`, `fc1.weight`, `fc2.weight`, `fc3.weight` (biases are not part of the BCQ pipeline).

### BCQ quantization (`weight_data/create_bcq_data.py`)
Implements the FIGLUT Section II-B style BCQ form with `num_bits=3` bit-planes, **per-row** offset `z`, and **per-row** alpha (one offset/alpha per output unit — per output neuron for `fc*.weight`, per output channel for `conv*.weight`, i.e. per `weight.shape[0]` row — not a single global per-layer scalar):
```
W_flat = W.reshape(m, -1)         # m = weight.shape[0], one row per output unit
z = mean(W_flat, dim=1, keepdim=True)
residual = W_flat - z
for k in range(num_bits):
    B_k = sign(residual)          # ±1 only, sign(0) forced to +1
    alpha_k = mean(|residual|, dim=1, keepdim=True)   # per-row, shape (m, 1)
    residual -= alpha_k * B_k
W_BCQ = sum_k(alpha_k * B_k) + z
```
`B_mtx` is saved in FIGLUT-style bit-first flattened form `(num_bits, m, n)`, where `n` is the flattened row width. `alpha` is saved as `(num_bits, m, 1)`, and `z` is saved as `(m, 1)`.
Produces two saved artifacts with different visibility, split across the two folders:
- `weight_data/W_BCQ_result.pth` — full data per layer (`W_true`, `B_mtx`, `alpha`, `z`, `W_BCQ`), defender-only, used for internal quality comparison.
- `Attacker/attacker_visible_B_only.pth` — **only** `B_mtx` (the sign bit-planes), simulating an attacker who can observe the quantized binary representation but not protected magnitude parameters (`alpha` and `z`). This is the shared input for every script under `Attacker/`.

### Quality evaluation (`weight_data/evaluate_bcq_quality.py`)
Sanity-checks the quantization itself: compares `W_true` vs. `W_BCQ` (the *full* reconstruction, alpha and z included) per layer and overall. Metrics: MAE, RMSE, cosine similarity, sign accuracy, Pearson, Spearman. This is not an attack — it measures quantization fidelity, not what an attacker without alpha/z could infer. Outputs `bcq_quality_metrics.{csv,png}` to `weight_data/`.

### `Attacker/` — leakage attacks against `attacker_visible_B_only.pth`

Both scripts here load `weight_data/W_true.pth` (ground truth, for scoring only) and `Attacker/attacker_visible_B_only.pth` (the attacker's actual input), and write their own outputs into `Attacker/`.

**`evaluate_value_leakage_attack.py`** — equal-sum attack: assumes each bit-plane contributes equally since alpha is hidden, `W_attack = sum_q(B_q)`, which only takes values in `{-3, -1, 1, 3}`. Compares `W_attack` against `W_true` per layer and overall via:
- sign accuracy, Pearson/Spearman correlation (with a shuffled-baseline control for both)
- quantile-binned classification accuracy + confusion matrix (true weight quartile vs. attack-level bucket)
- `monotonic_mean` check — whether mean `W_true` increases monotonically across attack levels `[-3, -1, 1, 3]`, i.e. whether the leak is at least *directionally* informative even without exact values
- boxplot (`value_leakage_attack_boxplot.png`) of `W_true` distribution grouped by attack level

Outputs: `value_leakage_attack_metrics.csv`, `value_leakage_attack_groups.csv`, `value_leakage_attack_confusion.csv`, `value_leakage_attack_boxplot.png`.

**`evaluate_weighted_sum_attack.py`** — generalizes the equal-sum attack to several heuristic bit-plane weightings (`ATTACKS` dict: `equal_sum_1_1_1`, `heuristic_half_1_0.5_0.25`, `heuristic_binary_4_2_1`), i.e. `W_attack = sum_q(w_q * B_q)` for guessed weight vectors `w_q` instead of assuming equal weight. Also rescales `W_attack` to `W_true`'s mean/std (`rescaled_mae`/`rescaled_rmse`) and compares distribution shape via KL divergence on z-scored histograms, since the raw attack scale is arbitrary without the true alpha. Outputs `weighted_sum_attack_metrics.csv` and `weighted_sum_attack_distributions.png`.

**`evaluate_pattern_value_level_attack.py`** — keeps the full 3-bit visible pattern instead of collapsing it to a summed value. Groups true weights by the exact pattern (`---`, `--+`, `-+-`, `-++`, `+--`, `+-+`, `++-`, `+++`) and reports count, mean, median, std, quartiles, positive rate, monotonicity by `4B1 + 2B2 + B3`, and eta-squared explained variance. Outputs `pattern_value_level_stats.csv`, `pattern_value_level_metrics.csv`, and `pattern_value_level_boxplot.png`.

**`evaluate_attack_accuracy.py`** — closes the loop from "statistically similar to `W_true`" to "actually usable as a classifier": loads `weight_data/W_true.pth`, `weight_data/W_BCQ_result.pth`, and `Attacker/attacker_visible_B_only.pth`, swaps each candidate weight tensor (`W_true` baseline, `W_BCQ` full reconstruction, and each `evaluate_weighted_sum_attack.py` heuristic's `W_attack` — both raw scale and rescaled to `W_true`'s mean/std) into a fresh `Model`'s state dict (biases always left as the real trained values — biases aren't part of the BCQ pipeline), then runs the real MNIST test set through it and reports actual classification accuracy per variant. Outputs `attack_reconstruction_accuracy.csv`.

### Known gap
None currently tracked — `evaluate_attack_accuracy.py` closes the previous gap (reconstructed weights are now scored on real MNIST test accuracy, not just statistical similarity to `W_true`).

### Environment note
`.venv/` has `torch`/`torchvision`/`numpy` but is missing `matplotlib` and `pandas`, which every `weight_data/`/`Attacker/` evaluation script requires (no `requirements.txt` exists to pin this).

## Re-run after changing BCQ format

When `weight_data/create_bcq_data.py` changes the BCQ decomposition or saved tensor format, regenerate all derived artifacts before comparing results. Run from repo root:

```
rm -f weight_data/W_BCQ_result.pth \
      Attacker/attacker_visible_B_only.pth \
      weight_data/bcq_quality_metrics.csv \
      weight_data/bcq_quality_metrics.png \
      Attacker/value_leakage_attack_metrics.csv \
      Attacker/value_leakage_attack_groups.csv \
      Attacker/value_leakage_attack_confusion.csv \
      Attacker/value_leakage_attack_boxplot.png \
      Attacker/weighted_sum_attack_metrics.csv \
      Attacker/weighted_sum_attack_distributions.png

python3 weight_data/load_w_true.py
python3 weight_data/create_bcq_data.py
python3 weight_data/evaluate_bcq_quality.py
python3 Attacker/evaluate_value_leakage_attack.py
python3 Attacker/evaluate_weighted_sum_attack.py
python3 Attacker/evaluate_pattern_value_level_attack.py
```

Expected sanity check from `create_bcq_data.py`: each layer should print `B_mtx shape: torch.Size([3, m, n])`, `alpha shape: torch.Size([3, m, 1])`, and `z shape: torch.Size([m, 1])`, where `m = weight.shape[0]`:

```
conv1.weight -> B [3, 6, 25], alpha [3, 6, 1], z [6, 1]
conv2.weight -> B [3, 16, 150], alpha [3, 16, 1], z [16, 1]
fc1.weight   -> B [3, 120, 256], alpha [3, 120, 1], z [120, 1]
fc2.weight   -> B [3, 84, 120], alpha [3, 84, 1], z [84, 1]
fc3.weight   -> B [3, 10, 84], alpha [3, 10, 1], z [10, 1]
```

The attacker scripts only load `B_mtx`; `alpha` and `z` remain defender-only protected magnitude data.
