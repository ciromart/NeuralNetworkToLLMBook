"""Video 1/5: verifica dei gradienti. Confronta micrograd_mini, differenze finite e PyTorch autograd."""
import math, torch
from micrograd_mini import Value

def f_micro(a, b, c):
    return ((a * b + c).tanh() * 2.0 + a ** 2)

def f_torch(a, b, c):
    return torch.tanh(a * b + c) * 2.0 + a ** 2

vals = dict(a=0.7, b=-1.3, c=0.4)

# 1) micrograd
a, b, c = (Value(v) for v in vals.values())
out = f_micro(a, b, c); out.backward()
g_micro = [a.grad, b.grad, c.grad]

# 2) PyTorch
ta, tb, tc = (torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in vals.values())
f_torch(ta, tb, tc).backward()
g_torch = [ta.grad.item(), tb.grad.item(), tc.grad.item()]

# 3) differenze finite (derivata numerica)
def numeric(i, h=1e-6):
    p = list(vals.values()); q = list(vals.values()); p[i] += h; q[i] -= h
    fp = f_micro(*(Value(x) for x in p)).data
    fm = f_micro(*(Value(x) for x in q)).data
    return (fp - fm) / (2 * h)
g_num = [numeric(i) for i in range(3)]

print(f"{'':8}{'micrograd':>12}{'torch':>12}{'numerico':>12}")
for n, x, y, z in zip("abc", g_micro, g_torch, g_num):
    print(f"d/d{n:<5}{x:12.6f}{y:12.6f}{z:12.6f}")
assert all(abs(x - y) < 1e-9 and abs(x - z) < 1e-6 for x, y, z in zip(g_micro, g_torch, g_num))
print("OK: i tre metodi coincidono")

# Video 5: derivata di softmax + cross-entropy = p - onehot
logits = torch.randn(1, 5, dtype=torch.float64, requires_grad=True)
target = torch.tensor([2])
torch.nn.functional.cross_entropy(logits, target).backward()
p = torch.softmax(logits.detach(), dim=1); p[0, 2] -= 1
print("dlogits autograd == p - onehot ?", torch.allclose(logits.grad, p))
