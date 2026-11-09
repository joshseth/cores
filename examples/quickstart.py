"""
Apply ACE to a small network: train a two-hidden-layer MLP on a toy task, extract the core at its first
hidden layer, and check that the core is sufficient (keeping only the core preserves accuracy) and
necessary (removing it drops accuracy to chance). The task's answer depends on two directions of the
input, and ACE finds a two-dimensional core. Runs on a CPU in a few seconds:

    python examples/quickstart.py
"""
import torch
import torch.nn as nn

from cores.extraction import ace
from cores.ablations import SubspaceAblationHook

torch.manual_seed(0)

# toy task: the class is the largest of three fixed linear readouts of a 20-dimensional input
A = torch.randn(3, 20)
X, X_test = torch.randn(4000, 20), torch.randn(2000, 20)
y, y_test = (X @ A.T).argmax(-1), (X_test @ A.T).argmax(-1)

model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 3))
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
for _ in range(500):
    loss = nn.functional.cross_entropy(model(X), y)
    opt.zero_grad(); loss.backward(); opt.step()

# ACE: the activations at a layer, and the rest of the network from that layer on
H = model[:2](X).detach()             # activations after the first hidden layer, shape (N, 64)
Q, r = ace(H, lambda: model[2:])      # the core: an orthonormal basis, shape (64, r)
print(f"core: {r} of {H.shape[1]} dimensions")

def accuracy(mode=None):
    """Test accuracy, optionally keeping only the core ('keep') or removing it ('remove') at that layer."""
    hook = model[1].register_forward_hook(SubspaceAblationHook(Q, mode)) if mode else None
    with torch.no_grad():
        acc = (model(X_test).argmax(-1) == y_test).float().mean().item()
    if hook:
        hook.remove()
    return acc

print(f"accuracy: model {accuracy():.3f}, core only {accuracy('keep'):.3f}, "
      f"core removed {accuracy('remove'):.3f} (chance 0.333)")
