import torch

class SubspaceAblationHook:
    def __init__(self, U: torch.Tensor, mode: str):
        assert mode in ("keep", "remove")

        self.U = U
        self.mode = mode

    def __call__(self, module, inp, out):
        U = self.U.to(out.device)
        proj = (out @ U) @ U.T
        return proj if self.mode == "keep" else out - proj

@torch.no_grad()
def ablate(model, layer, U, mode, loader, device, *, last_token=False):
    hook = SubspaceAblationHook(U, mode)
    handle = layer.register_forward_hook(hook)
    try:
        model.eval()
        correct = total = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            if last_token:
                pred, tgt = logits[:, -1].argmax(-1), y[:, -1]
            else:
                pred, tgt = logits.argmax(-1), y
            correct += (pred == tgt).sum().item()
            total += tgt.numel()
        return correct / max(1, total)
    finally:
        handle.remove()