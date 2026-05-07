import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class OneLayerTransformer(nn.Module):
    def __init__(self, vocab_size: int = 4, d_model: int = 64, n_head: int = 4, max_len: int = 64):
        super().__init__()
        self.vocab_size = vocab_size

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        self.layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=4*d_model,
            batch_first=True, 
        )

        self.lm_head = nn.Linear(d_model, vocab_size)

    @staticmethod
    def causal_mask(T: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
    
    def forward(self, x: torch.Tensor, return_states: bool = False):
        B, T = x.shape
        pos = torch.arange(T, device = x.device).unsqueeze(0)
        h0 = self.token_emb(x) + self.pos_emb(pos)

        attn_mask = self.causal_mask(T, x.device)

        h1 = self.layer(h0, src_mask = attn_mask)
        logits = self.lm_head(h1)

        if return_states:
            return logits, h1
        return logits
    
class TwoLayerTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 64, n_head: int = 4, max_len: int = 64, d_ff: int = 512):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        self.layer1 = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=d_ff, batch_first=True)
        self.layer2 = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=d_ff, batch_first=True)

        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor, return_states: bool = False):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h0 = self.token_emb(x) + self.pos_emb(pos)  # (B,T,D)
 
        causal = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
        h1 = self.layer1(h0, src_mask=causal)
        h2 = self.layer2(h1, src_mask=causal)

        logits = self.lm_head(h2)
        if return_states:
            return logits, h1, h2
        return logits 

    
def train_markov_lm(model, loader, device, *, epochs=40, lr=1e-4):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    for ep in range(1, epochs+1):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(-1, V), y.view(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
        yield ep

@torch.no_grad()
def eval_accuracy(model, loader, device, *, last_token=False):
    was_training = model.training
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
    if was_training:
        model.train()
    return correct / max(1, total)


# Data
class MarkovDataset(Dataset):
    def __init__(self, T: np.ndarray, 
                 seed: int = 0,
                 num_samples: int = 3000,
                 seq_len: int = 32):
        self.T = T
        V = T.shape[0]
        rng = np.random.RandomState(seed)
        seqs = np.zeros((num_samples, seq_len), dtype=np.int64)
        seqs[:,0] = rng.choice(V, size=num_samples)
        for t in range(1, seq_len):
            probs = T[seqs[:, t-1]]
            seqs[:, t] = np.array([rng.choice(V, p=p) for p in probs])

        self.x = torch.from_numpy(seqs[:,:-1])
        self.y = torch.from_numpy(seqs[:,1:])

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, i):
        return self.x[i], self.y[i]
    

def collect_markov_activations(model, data_loader, device):
    X, H, Y = [], [], []
    model.eval()

    for x, y in data_loader:
        x = x.to(device)
        _, h = model(x, return_states = True)
        X.append(x.detach().cpu())
        H.append(h.detach().cpu())
        Y.append(y.detach().cpu())

    Xout = torch.cat(X, dim=0)
    Hout = torch.cat(H, dim=0)
    Yout = torch.cat(Y, dim=0)

    return Xout, Hout, Yout

class ModAddDataset(Dataset):
    """
    a + b = c mod P. We use triples [a, b, c] and train on next-token prediction.

    Input:  [a, b]
    Target: [b, c]
    """
    def __init__(self, p=53, train=True):
        data = []
        for a in range(p):
            for b in range(p):
                c = (a + b) % p
                data.append(torch.tensor([a, b, c], dtype=torch.long))
        data = torch.stack(data)

        rng = np.random.RandomState(42)
        perm = rng.permutation(len(data))
        split = len(data) // 2
        if train:
            self.data = data[perm[:split]]
        else:
            self.data = data[perm[split:]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        return seq[:-1], seq[1:]  # ([a,b], [b,c])


def build_modadd_dataloader(train=True, seed=0, batch_size=512):
    torch.manual_seed(seed)
    ds = ModAddDataset(train=train)
    return DataLoader(ds, batch_size=batch_size, shuffle=train)
        

@torch.no_grad()
def collect_last_token_h2(model, loader, device, max_seqs=5000):
    """Collect last-token layer-2 hidden states. Returns (N, D) on CPU."""
    model.eval()
    chunks = []
    n = 0
    for x, _ in loader:
        _, _, h2 = model(x.to(device), return_states=True)
        chunks.append(h2[:, -1, :].detach().cpu())
        n += x.size(0)
        if n >= max_seqs:
            break
    return torch.cat(chunks, dim=0)