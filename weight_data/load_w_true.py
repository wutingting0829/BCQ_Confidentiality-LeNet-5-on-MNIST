import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def get_accuracy_from_checkpoint(path):
    return float(path.stem.split("_")[-1])


checkpoint_paths = sorted(
    (ROOT / "models").glob("mnist_*.pkl"),
    key=get_accuracy_from_checkpoint,
)

if not checkpoint_paths:
    raise FileNotFoundError("No checkpoint found in models/mnist_*.pkl")

checkpoint_path = checkpoint_paths[-1]

model = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
model.eval()

W_true = model.state_dict()

output_path = ROOT / "weight_data" / "W_true.pth"
torch.save(W_true, output_path)

print("Loaded original model from:", checkpoint_path)
print("Saved W_true to:", output_path)
print("Weights:")
for name, tensor in W_true.items():
    print(name, tensor.shape)
