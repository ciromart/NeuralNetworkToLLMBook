"""Cap. 8 / Appendice A: LoRA da zero. Si congela un layer lineare e si allena solo la correzione a basso rango B·A."""
import torch, torch.nn as nn
torch.manual_seed(0)
D, R = 256, 4

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=R, alpha=8):
        super().__init__()
        self.base = base
        for p in self.base.parameters(): p.requires_grad = False       # pesi originali CONGELATI
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))        # B=0 → all'inizio W' = W
        self.scale = alpha / r
    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale

base = nn.Linear(D, D)
# "Compito nuovo": una trasformazione target = W originale + una correzione di rango 4
with torch.no_grad():
    U, V_ = torch.randn(D, R), torch.randn(R, D)
    target_W = base.weight + 0.5 * (U @ V_) / D ** 0.5

layer = LoRALinear(base)
trainable = sum(p.numel() for p in layer.parameters() if p.requires_grad)
total = sum(p.numel() for p in layer.parameters())
print(f"parametri totali: {total}  allenabili (LoRA): {trainable}  = {100*trainable/total:.2f}%")

W_before = base.weight.detach().clone()
opt = torch.optim.Adam([layer.A, layer.B], lr=1e-2)
X = torch.randn(512, D)
Y = X @ target_W.T + base.bias
for step in range(601):
    loss = ((layer(X) - Y) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 150 == 0: print(f"step {step:3d}  loss {loss.item():.5f}")
print("pesi base identici a prima del training?", torch.equal(W_before, base.weight))
merged = base.weight + (layer.B @ layer.A) * layer.scale      # merge: si può eliminare l'adapter
print("errore dopo il merge:", ((X @ merged.T + base.bias - Y) ** 2).mean().item())
