"""Video 2-3: language model a caratteri con MLP + embedding (stile makemore) su nomi italiani. PyTorch."""
import random, torch, torch.nn.functional as F

NAMES = """giulia sofia aurora alice ginevra emma giorgia beatrice greta chiara martina anna vittoria noemi sara alessia
matilde bianca camilla elisa francesca ludovica gaia arianna rebecca federica angelica lucia eleonora nicole irene
marco luca francesco alessandro andrea matteo lorenzo gabriele riccardo davide leonardo tommaso mattia federico simone
giuseppe antonio giovanni salvatore filippo edoardo samuele nicolo diego pietro enrico stefano claudio fabio paolo
roberto massimo daniele michele valerio emanuele cristian manuel gianluca raffaele vincenzo ciro carmine gennaro
rosa maria concetta teresa antonella paola laura silvia monica elena valentina serena veronica roberta daniela
cinzia lorella patrizia donatella giovanna annamaria carla lidia flavia ilaria claudia manuela marta olga rita""".split()

random.seed(42); torch.manual_seed(42)
random.shuffle(NAMES)
chars = ["."] + sorted(set("".join(NAMES)))
stoi = {c: i for i, c in enumerate(chars)}; itos = {i: c for c, i in stoi.items()}
V, BLOCK, EMB, HID = len(chars), 3, 4, 30

def build(words):
    X, Y = [], []
    for w in words:
        ctx = [0] * BLOCK
        for ch in w + ".":
            X.append(ctx); Y.append(stoi[ch]); ctx = ctx[1:] + [stoi[ch]]
    return torch.tensor(X), torch.tensor(Y)

n1, n2 = int(.8 * len(NAMES)), int(.9 * len(NAMES))
Xtr, Ytr = build(NAMES[:n1]); Xdv, Ydv = build(NAMES[n1:n2]); Xte, Yte = build(NAMES[n2:])
print(f"nomi: {len(NAMES)}  vocabolario: {V}  esempi train/dev/test: {len(Xtr)}/{len(Xdv)}/{len(Xte)}")
print(f"loss attesa di un modello casuale: ln({V}) = {torch.log(torch.tensor(float(V))).item():.3f}")

C  = torch.randn(V, EMB)
W1 = torch.randn(BLOCK * EMB, HID) * (5 / 3) / (BLOCK * EMB) ** 0.5
b1 = torch.zeros(HID)
W2 = torch.randn(HID, V) * 0.01
b2 = torch.zeros(V)
params = [C, W1, b1, W2, b2]
for p in params: p.requires_grad = True
print("parametri:", sum(p.numel() for p in params))

def loss_on(X, Y):
    h = torch.tanh(C[X].view(len(X), -1) @ W1 + b1)
    return F.cross_entropy(h @ W2 + b2, Y)

STEPS, best_dev, best = 6000, 9e9, None
for step in range(STEPS):
    ix = torch.randint(0, len(Xtr), (32,))
    loss = loss_on(Xtr[ix], Ytr[ix])
    for p in params: p.grad = None
    loss.backward()
    lr = 0.1 if step < STEPS * 0.6 else 0.01      # learning-rate decay
    for p in params: p.data -= lr * p.grad
    if step % 250 == 0 or step == STEPS - 1:
        with torch.no_grad():
            tr, dv = loss_on(Xtr, Ytr).item(), loss_on(Xdv, Ydv).item()
        if dv < best_dev:                                   # early stopping: ricorda i pesi migliori sul DEV
            best_dev, best = dv, [p.detach().clone() for p in params]
        if step % 1000 == 0 or step == STEPS - 1:
            print(f"step {step:5d}  train {tr:.3f}  dev {dv:.3f}" + ("   <- overfitting: dev risale" if dv > best_dev + 0.05 else ""))
for p, b in zip(params, best): p.data = b                    # ripristina il modello migliore
with torch.no_grad():
    print(f"miglior dev: {best_dev:.3f}  |  loss test finale: {loss_on(Xte, Yte).item():.3f}  (casuale: 3.045)")

g = torch.Generator().manual_seed(7)
out = []
for _ in range(10):
    ctx, s = [0] * BLOCK, ""
    while True:
        with torch.no_grad():
            h = torch.tanh(C[torch.tensor([ctx])].view(1, -1) @ W1 + b1)
            probs = F.softmax(h @ W2 + b2, dim=1)
        i = torch.multinomial(probs, 1, generator=g).item()
        if i == 0 or len(s) > 15: break
        s += itos[i]; ctx = ctx[1:] + [i]
    out.append(s)
print("nomi generati:", out)
