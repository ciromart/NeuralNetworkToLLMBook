"""Genera tutte le figure del libro in book/figures/. Uso: python3 scripts/make_figures.py"""
import sys, pathlib, random, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "book" / "figures"; OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "code" / "python"))
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 130, "savefig.bbox": "tight", "savefig.facecolor": "white"})
BLU, VERDE, ARAN, ROSSO, VIO, GRI = "#2563eb", "#16a34a", "#f59e0b", "#dc2626", "#7c3aed", "#6b7280"

def save(name): plt.savefig(OUT / f"{name}.png"); plt.close()

def box(ax, x, y, w, h, text, color=BLU, fs=10, tc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=color, ec="none"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tc, fontsize=fs, fontweight="bold")

def arrow(ax, p, q, color=GRI, style="-|>"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=14, color=color, lw=1.6))

def canvas(w, h, xl, yl):
    fig, ax = plt.subplots(figsize=(w, h)); ax.set_xlim(0, xl); ax.set_ylim(0, yl); ax.axis("off"); return fig, ax

# 01 neurone
fig, ax = canvas(8, 3.6, 10, 4.5)
for i, y in enumerate([3.5, 2.25, 1.0]):
    ax.add_patch(Circle((0.8, y), 0.35, fc=ARAN)); ax.text(0.8, y, f"x{i+1}", ha="center", va="center", color="white", fontweight="bold")
    arrow(ax, (1.2, y), (4.1, 2.25)); ax.text(2.4, y*0.6+1.2+ (0.15 if i==0 else -0.2 if i==2 else 0.15), f"w{i+1}", color=GRI)
ax.add_patch(Circle((4.6, 2.25), 0.6, fc=BLU)); ax.text(4.6, 2.25, "Σ + b", ha="center", va="center", color="white", fontweight="bold")
arrow(ax, (5.25, 2.25), (6.4, 2.25)); box(ax, 6.4, 1.8, 1.6, 0.9, "tanh / ReLU", VERDE, 9)
arrow(ax, (8.0, 2.25), (9.2, 2.25)); ax.text(9.5, 2.25, "y", fontsize=14, fontweight="bold", va="center")
ax.text(5, 0.2, "y = f( w1·x1 + w2·x2 + w3·x3 + b )", ha="center", fontsize=11, style="italic")
save("01_neurone")

# 02 attivazioni
x = np.linspace(-4, 4, 400)
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(x, np.tanh(x), label="tanh", color=BLU, lw=2); ax.plot(x, 1/(1+np.exp(-x)), label="sigmoide", color=ROSSO, lw=2)
ax.plot(x, np.maximum(0, x), label="ReLU", color=VERDE, lw=2)
ax.plot(x, x/(1+np.exp(-1.702*x)), label="GELU (≈)", color=VIO, lw=2, ls="--")
ax.axhline(0, color="#ddd", lw=.8); ax.axvline(0, color="#ddd", lw=.8); ax.set_ylim(-1.5, 3); ax.legend(); ax.set_title("Funzioni di attivazione")
save("02_attivazioni")

# 03 discesa del gradiente + loss reale micrograd
f = lambda w: (w - 2) ** 2 + 0.5
w = np.linspace(-1, 5, 200); fig, ax = plt.subplots(figsize=(7, 4)); ax.plot(w, f(w), color=GRI, lw=2)
p = -0.5; pts = [p]
for _ in range(7): p -= 0.3 * 2 * (p - 2); pts.append(p)
ax.plot(pts, [f(q) for q in pts], "o-", color=ROSSO); ax.set_xlabel("parametro w"); ax.set_ylabel("loss")
ax.set_title("Discesa del gradiente: w ← w − η·∂L/∂w"); save("03_gradient_descent")

import micrograd_mini as mg
random.seed(1337)
xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]; ys = [1.0, -1.0, -1.0, 1.0]
net = mg.MLP(3, [4, 4, 1]); losses = []
for _ in range(60):
    loss = sum(((net(a) - b) ** 2 for a, b in zip(xs, ys)), mg.Value(0.0))
    for q in net.parameters(): q.grad = 0.0
    loss.backward()
    for q in net.parameters(): q.data -= 0.05 * q.grad
    losses.append(loss.data)
fig, ax = plt.subplots(figsize=(7, 4)); ax.plot(losses, color=BLU, lw=2); ax.set_yscale("log")
ax.set_xlabel("step"); ax.set_ylabel("loss (log)"); ax.set_title("Training reale di micrograd_mini.py"); save("04_loss_micrograd")

# 05 bigram heatmap (dati reali)
from collections import defaultdict
words = ["emma","olivia","ava","isabella","sophia","mia","amelia","harper","evelyn","abigail","emily","ella","elizabeth","camila","luna","sofia"]
chars = ["."] + sorted(set("".join(words))); idx = {c: i for i, c in enumerate(chars)}
N = np.zeros((len(chars), len(chars)))
for wd in words:
    s = ["."] + list(wd) + ["."]
    for a, b in zip(s, s[1:]): N[idx[a], idx[b]] += 1
fig, ax = plt.subplots(figsize=(7, 6)); ax.imshow(N, cmap="Blues")
ax.set_xticks(range(len(chars))); ax.set_xticklabels(chars); ax.set_yticks(range(len(chars))); ax.set_yticklabels(chars)
ax.set_xlabel("carattere successivo"); ax.set_ylabel("carattere corrente"); ax.set_title("Matrice dei conteggi bigramma (16 nomi)"); ax.spines[:].set_visible(False)
save("05_bigram_heatmap")

# 06 embedding (illustrativo)
rng = np.random.default_rng(3)
groups = {"vocali": (list("aeiou"), (-2, 1.5), BLU), "consonanti dolci": (list("lmnr"), (2, 1.5), VERDE), "occlusive": (list("bdgptk"), (0, -2), ROSSO)}
fig, ax = plt.subplots(figsize=(6.5, 4.5))
for name, (ls, c, col) in groups.items():
    for l in ls:
        px, py = rng.normal(c, 0.55); ax.scatter(px, py, color=col, s=60); ax.text(px + .08, py + .08, l, fontsize=12)
    ax.scatter([], [], color=col, label=name)
ax.legend(); ax.set_title("Embedding 2D (illustrativo): lettere simili → vicine"); ax.set_xticks([]); ax.set_yticks([]); save("06_embedding")

# 07 softmax temperatura
logits = np.array([2.0, 1.0, 0.5, 0.1, -1.0]); labs = ["il", "un", "lo", "la", "gli"]
fig, axs = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
for ax, T in zip(axs, [0.2, 1.0, 2.0]):
    e = np.exp(logits / T); pr = e / e.sum(); ax.bar(labs, pr, color=[BLU, VERDE, ARAN][[0.2, 1.0, 2.0].index(T)]); ax.set_title(f"temperature = {T}")
axs[0].set_ylabel("probabilità"); fig.suptitle("Softmax: la temperatura controlla la 'creatività'", y=1.04); save("07_softmax_temperatura")

# 08 attention heatmap (reale)
import attention as at
r = np.random.default_rng(0); T, C, H = 6, 8, 4
_, att = at.causal_self_attention(r.normal(size=(T, C)), *(r.normal(size=(C, H)) for _ in range(3)))
toks = ["Il", "gatto", "dorme", "sul", "divano", "."]
fig, ax = plt.subplots(figsize=(5.5, 4.8)); ax.imshow(att, cmap="Purples")
ax.set_xticks(range(T)); ax.set_xticklabels(toks, rotation=40); ax.set_yticks(range(T)); ax.set_yticklabels(toks)
ax.set_xlabel("token guardato (key)"); ax.set_ylabel("token che guarda (query)"); ax.set_title("Attenzione causale (triangolare)"); ax.spines[:].set_visible(False); save("08_attention_heatmap")

# 09 blocco transformer
fig, ax = canvas(5.2, 7.2, 8, 11)
steps = [("Token + posizione\n(embedding)", ARAN), ("LayerNorm", GRI), ("Masked Multi-Head\nSelf-Attention", BLU), ("+ residuo", VERDE), ("LayerNorm", GRI), ("Feed-Forward (MLP)\n4× → GELU → 1×", VIO), ("+ residuo", VERDE), ("LayerNorm finale", GRI), ("Linear → Softmax\n(prossimo token)", ROSSO)]
y = 0.3
for i, (t, c) in enumerate(steps):
    box(ax, 1.5, y, 5, 0.95, t, c, 9); 
    if i < len(steps) - 1: arrow(ax, (4, y + 0.95), (4, y + 1.2))
    y += 1.2
ax.add_patch(Rectangle((1.2, 1.5), 5.6, 7.0, fill=False, ec=BLU, ls="--")); ax.text(7.0, 5.0, "× N\nblocchi", color=BLU, fontsize=11, fontweight="bold")
save("09_transformer_block")

# 10 BPE compressione (reale su testo ripetuto)
import bpe
txt = ("il gatto dorme sul divano e il cane dorme sul tappeto mentre il gatto guarda il cane " * 8)
ids = list(txt.encode()); lens = [len(ids)]
merges = bpe.train(txt, 40)
cur = list(txt.encode())
for pair, nid in merges.items(): cur = bpe.merge(cur, pair, nid); lens.append(len(cur))
fig, ax = plt.subplots(figsize=(7, 4)); ax.plot(lens, color=VIO, lw=2); ax.set_xlabel("numero di merge BPE"); ax.set_ylabel("lunghezza sequenza (token)")
ax.set_title("BPE: più merge → sequenze più corte (dati reali)"); save("10_bpe_compressione")

# 11 pipeline training
fig, ax = canvas(11, 3.2, 14, 4)
for i, (t, sub, c) in enumerate([("Pretraining", "trilioni di token\nnext-token prediction", BLU), ("SFT", "esempi\nprompt→risposta", VERDE), ("Reward model", "confronti umani\nA > B", ARAN), ("RLHF / DPO", "ottimizza\nle preferenze", VIO)]):
    box(ax, i * 3.5 + .1, 1.6, 3.0, 1.1, t, c, 12); ax.text(i * 3.5 + 1.6, 1.0, sub, ha="center", va="center", fontsize=9, color=GRI)
    if i < 3: arrow(ax, (i * 3.5 + 3.15, 2.15), (i * 3.5 + 3.6, 2.15))
ax.text(1.6, 3.3, "Modello base", ha="center", color=GRI); ax.text(8.6, 3.3, "Modello assistente", ha="center", color=GRI); save("11_pipeline_training")

# 12 scaling
n = np.logspace(6, 11, 50); L = 1.7 + 400 / n ** 0.34
fig, ax = plt.subplots(figsize=(7, 4)); ax.loglog(n, L - 1.5, color=BLU, lw=2); ax.set_xlabel("parametri (scala log)"); ax.set_ylabel("loss riducibile (log)")
ax.set_title("Scaling law (curva illustrativa)"); save("12_scaling")

# 13 quantizzazione (calcolo reale: parametri × byte)
sizes = [("3B", 3), ("8B", 8), ("70B", 70)]; fmts = [("FP16", 2, ROSSO), ("INT8", 1, ARAN), ("Q4", 0.5, VERDE)]
fig, ax = plt.subplots(figsize=(7.5, 4)); wd = 0.25
for j, (fn, b, c) in enumerate(fmts):
    ax.bar(np.arange(3) + (j - 1) * wd, [s * b for _, s in sizes], wd, label=fn, color=c)
ax.set_xticks(range(3)); ax.set_xticklabels([s for s, _ in sizes]); ax.set_ylabel("RAM per i pesi (GB)"); ax.legend()
ax.set_title("Memoria dei pesi = parametri × byte (senza KV-cache)"); save("13_quantizzazione")

# 14 LoRA
fig, ax = canvas(8, 3.8, 10, 5)
box(ax, 0.5, 1.6, 2.6, 1.8, "W congelata\n(d × d)", GRI, 11); ax.text(3.45, 2.5, "+", fontsize=22, ha="center", va="center")
box(ax, 3.9, 2.9, 1.6, 0.8, "A (d × r)", VERDE, 10); box(ax, 3.9, 1.3, 1.6, 0.8, "B (r × d)", VERDE, 10); ax.text(4.7, 2.45, "r ≪ d", ha="center", color=VERDE, fontweight="bold")
ax.text(6.2, 2.5, "=", fontsize=22, ha="center", va="center"); box(ax, 6.7, 1.6, 2.6, 1.8, "W' = W + B·A", BLU, 11)
ax.text(5, 0.5, "Si allenano solo A e B: pochi milioni di parametri invece di miliardi", ha="center", style="italic"); save("14_lora")

# 15 RAG
fig, ax = canvas(12, 5, 16, 6.5)
box(ax, .2, 4.4, 2.6, 1.0, "Documenti\n(PDF, wiki, DB)", GRI, 9); box(ax, 3.6, 4.4, 2.2, 1.0, "Chunking", ARAN, 10); box(ax, 6.6, 4.4, 2.6, 1.0, "Embedding\nmodel", VIO, 10); box(ax, 10.0, 4.4, 2.8, 1.0, "Vector Store\n(PgVector)", VERDE, 10)
for a, b in [(2.8, 3.6), (5.8, 6.6), (9.2, 10.0)]: arrow(ax, (a, 4.9), (b, 4.9))
ax.text(1.5, 6.0, "① INGESTION (offline)", color=GRI, fontweight="bold")
box(ax, .2, 1.4, 2.4, 1.0, "Domanda\nutente", ARAN, 10); box(ax, 3.4, 1.4, 2.4, 1.0, "Embedding\ndomanda", VIO, 10); box(ax, 6.6, 1.4, 2.6, 1.0, "Top-K\nsimilarità", VERDE, 10); box(ax, 10.0, 1.4, 2.8, 1.0, "Prompt =\ndomanda + contesto", BLU, 9); box(ax, 13.2, 1.4, 2.4, 1.0, "LLM locale\n(Ollama)", ROSSO, 10)
for a, b in [(2.6, 3.4), (5.8, 6.6), (9.2, 10.0), (12.8, 13.2)]: arrow(ax, (a, 1.9), (b, 1.9))
arrow(ax, (11.4, 4.4), (7.9, 2.4), VERDE, "<|-|>"); ax.text(1.5, 3.0, "② QUERY (online)", color=GRI, fontweight="bold"); save("15_rag")

# 16 architettura Spring AI
fig, ax = canvas(11, 5.2, 15, 7)
box(ax, .2, 5.4, 3.2, 1.1, "Client\n(web / mobile)", GRI, 10); box(ax, 4.6, 5.4, 4.0, 1.1, "@RestController\n(validazione, SSE)", BLU, 10)
box(ax, 4.6, 3.6, 4.0, 1.1, "ChatClient + Advisors\n(memoria, RAG, log)", VIO, 10); box(ax, 4.6, 1.8, 4.0, 1.1, "ChatModel / EmbeddingModel\n(astrazione portabile)", VERDE, 9)
box(ax, 10.2, 1.8, 3.8, 1.1, "Ollama locale\nllama3.2 · mistral", ROSSO, 10); box(ax, 10.2, 3.6, 3.8, 1.1, "Tools → RestClient\nservizi REST esterni", ARAN, 9); box(ax, 10.2, 5.4, 3.8, 1.1, "VectorStore\nPgVector / Simple", VERDE, 10)
arrow(ax, (3.4, 5.9), (4.6, 5.9)); arrow(ax, (6.6, 5.4), (6.6, 4.7)); arrow(ax, (6.6, 3.6), (6.6, 2.9)); arrow(ax, (8.6, 2.35), (10.2, 2.35)); arrow(ax, (8.6, 4.15), (10.2, 4.15)); arrow(ax, (8.6, 4.6), (10.2, 5.7))
ax.text(7.5, 0.7, "Il codice applicativo dipende solo dalle interfacce Spring AI: cambi provider cambiando configurazione.", ha="center", style="italic", fontsize=9); save("16_spring_ai_architettura")

# 17 mockup chat UI
fig, ax = canvas(6, 7.4, 10, 12.4)
ax.add_patch(FancyBboxPatch((0.2, 0.2), 9.6, 12, boxstyle="round,pad=0.02,rounding_size=0.3", fc="#f8fafc", ec="#cbd5e1", lw=2))
ax.add_patch(Rectangle((0.2, 11.0), 9.6, 1.2, fc=BLU)); ax.text(0.8, 11.6, "Assistente Catalogo (llama3.2 · locale)", color="white", fontsize=11, va="center", fontweight="bold")
def bub(y, t, user): 
    x = 4.2 if user else 0.7; box(ax, x, y, 5.1, 1.5, t, BLU if user else "#e2e8f0", 9, "white" if user else "#111827")
bub(9.2, "Quanti pezzi ho in magazzino\ndello SKU ABC-123?", True); bub(7.2, "tool: productInfo(\"ABC-123\")", False); bub(5.2, "Tastiera meccanica: 49,90 €,\ngiacenza 12 pezzi.", False)
bub(3.2, "Fammene un riassunto in 1 frase", True)
ax.text(0.8, 2.3, "● ● ●  sta scrivendo (streaming SSE)…", color=GRI, fontsize=9)
box(ax, 0.6, 0.6, 7.2, 1.0, "Scrivi un messaggio…", "#e2e8f0", 10, GRI); box(ax, 8.0, 0.6, 1.5, 1.0, "Invia", VERDE, 10); save("17_mockup_chat")

# 18 sequenza tool calling
fig, ax = canvas(11, 5, 16, 7)
cols = [("Utente", 1.5), ("Controller", 4.8), ("ChatClient", 8.1), ("LLM (Ollama)", 11.4), ("API REST", 14.5)]
for n_, x_ in cols:
    box(ax, x_ - 1.2, 6.0, 2.4, 0.8, n_, BLU, 9); ax.plot([x_, x_], [0.3, 6.0], color="#cbd5e1", ls="--")
msgs = [(0, 1, 5.3, "domanda"), (1, 2, 4.7, "prompt()"), (2, 3, 4.1, "messaggi + schema tool"), (3, 2, 3.5, "tool_call productInfo(sku)"), (2, 4, 2.9, "GET /products/{sku}"), (4, 2, 2.3, "JSON prodotto"), (2, 3, 1.7, "risultato tool"), (3, 2, 1.1, "risposta finale"), (2, 0, 0.5, "risposta")]
for a, b, y_, t in msgs:
    arrow(ax, (cols[a][1], y_), (cols[b][1], y_), ROSSO if "tool" in t else GRI); ax.text((cols[a][1] + cols[b][1]) / 2, y_ + 0.12, t, ha="center", fontsize=8)
save("18_sequenza_tool_calling")
print("figure generate:", len(list(OUT.glob("*.png"))))
