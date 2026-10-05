"""Video 4: perché l'inizializzazione conta. Statistiche delle attivazioni in una rete tanh a 6 strati."""
import torch
torch.manual_seed(0)
N, D, L = 1000, 100, 6
x0 = torch.randn(N, D)

def run(scale_fn, batchnorm=False):
    x, rows = x0, []
    for i in range(L):
        W = torch.randn(D, D) * scale_fn(D)
        h = x @ W
        if batchnorm:
            h = (h - h.mean(0)) / (h.std(0) + 1e-5)
        x = torch.tanh(h)
        rows.append((i + 1, x.std().item(), (x.abs() > 0.97).float().mean().item() * 100))
    return rows

cases = {
    "troppo piccola (0.01)":      (lambda d: 0.01, False),
    "troppo grande (1.0)":        (lambda d: 1.0, False),
    "Kaiming (5/3)/sqrt(fan_in)": (lambda d: (5 / 3) / d ** 0.5, False),
    "pesi grandi + BatchNorm":    (lambda d: 1.0, True),
}
for name, (fn, bn) in cases.items():
    print(f"\n{name}")
    print("  strato   std attivazioni   % saturi (|tanh|>0.97)")
    for i, s, sat in run(fn, bn):
        print(f"  {i:^6}   {s:^15.3f}   {sat:^10.1f}")
