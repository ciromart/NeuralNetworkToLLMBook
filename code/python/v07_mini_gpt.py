"""Video 7: un mini-GPT a caratteri (decoder-only) in PyTorch, allenabile su CPU in ~1-2 minuti.
ATTENZIONE: il corpus è minuscolo e ripetuto 6 volte, quindi la loss di validazione NON misura la generalizzazione
(il val set contiene testo già visto): il modello memorizza. Serve a mostrare la meccanica, non la qualità.
Contiene: embedding token+posizione, self-attention causale multi-testa, MLP, residui, LayerNorm, generazione."""
import time, torch, torch.nn as nn, torch.nn.functional as F

TEXT = """un modello di linguaggio prevede il prossimo token dato il contesto. la rete neurale impara dai dati riducendo l errore.
il gradiente indica la direzione in cui cambiare i pesi per ridurre la loss. la backpropagation calcola i gradienti con la regola della catena.
il transformer usa l attenzione per far comunicare i token tra loro. ogni token guarda solo i token precedenti grazie alla maschera causale.
il tokenizer trasforma il testo in numeri. l embedding trasforma i numeri in vettori. il vettore contiene il significato del token.
spring ai permette di usare un modello di linguaggio da java. ollama esegue il modello in locale sul computer.
il rag cerca i documenti rilevanti e li aggiunge al prompt prima di chiedere la risposta al modello.
""" * 6

torch.manual_seed(1337)
chars = sorted(set(TEXT)); V = len(chars)
stoi = {c: i for i, c in enumerate(chars)}; itos = {i: c for c, i in stoi.items()}
data = torch.tensor([stoi[c] for c in TEXT])
n = int(0.9 * len(data)); train, val = data[:n], data[n:]
B, T, C, H, L = 32, 64, 64, 4, 3          # batch, contesto, embedding, teste, strati
print("NB: corpus ripetuto -> val non indipendente (il modello memorizza)")
print(f"caratteri: {len(TEXT)}  vocabolario: {V}  loss casuale attesa ln(V) = {torch.log(torch.tensor(float(V))).item():.3f}")

def batch(split):
    d = train if split == "train" else val
    ix = torch.randint(len(d) - T - 1, (B,))
    return torch.stack([d[i:i + T] for i in ix]), torch.stack([d[i + 1:i + T + 1] for i in ix])

class Head(nn.Module):
    def __init__(self, hs):
        super().__init__()
        self.k, self.q, self.v = (nn.Linear(C, hs, bias=False) for _ in range(3))
        self.register_buffer("mask", torch.tril(torch.ones(T, T)))
    def forward(self, x):
        t = x.shape[1]
        k, q, v = self.k(x), self.q(x), self.v(x)
        w = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5          # (B,t,t)
        w = w.masked_fill(self.mask[:t, :t] == 0, float("-inf"))   # maschera causale
        return F.softmax(w, dim=-1) @ v

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = nn.ModuleList(Head(C // H) for _ in range(H))
        self.proj = nn.Linear(C, C)
        self.ff = nn.Sequential(nn.Linear(C, 4 * C), nn.GELU(), nn.Linear(4 * C, C))
        self.ln1, self.ln2 = nn.LayerNorm(C), nn.LayerNorm(C)
    def forward(self, x):
        x = x + self.proj(torch.cat([h(self.ln1(x)) for h in self.heads], dim=-1))
        return x + self.ff(self.ln2(x))

class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok, self.pos = nn.Embedding(V, C), nn.Embedding(T, C)
        self.blocks = nn.Sequential(*[Block() for _ in range(L)])
        self.ln, self.head = nn.LayerNorm(C), nn.Linear(C, V)
    def forward(self, idx, targets=None):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1]))
        logits = self.head(self.ln(self.blocks(x)))
        loss = None if targets is None else F.cross_entropy(logits.view(-1, V), targets.view(-1))
        return logits, loss
    @torch.no_grad()
    def generate(self, idx, n_new, temperature=0.8, top_k=10):
        for _ in range(n_new):
            logits, _ = self(idx[:, -T:])
            logits = logits[:, -1] / temperature
            v, _ = torch.topk(logits, top_k); logits[logits < v[:, [-1]]] = -float("inf")
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, dim=-1), 1)], dim=1)
        return idx

model = MiniGPT()
print("parametri:", sum(p.numel() for p in model.parameters()))
opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

@torch.no_grad()
def eval_loss(split):
    model.eval(); l = sum(model(*batch(split))[1].item() for _ in range(10)) / 10; model.train(); return l

t0 = time.time()
for step in range(1500):
    xb, yb = batch("train")
    _, loss = model(xb, yb)
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if step % 300 == 0 or step == 1499:
        print(f"step {step:4d}  train {eval_loss('train'):.3f}  val {eval_loss('val'):.3f}  ({time.time()-t0:.0f}s)")

prompt = torch.tensor([[stoi[c] for c in "il transformer "]])
print("\n--- testo generato ---")
print("".join(itos[i] for i in model.generate(prompt, 200)[0].tolist()))
