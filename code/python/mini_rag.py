"""Mini-RAG senza dipendenze: embedding 'bag of words' + similarità coseno + prompt aumentato.
Serve solo a capire il meccanismo; in produzione l'embedding lo calcola un modello (nomic-embed-text)."""
import math, re
from collections import Counter

DOCS = {
    "ferie":   "Le ferie vanno richieste al responsabile con almeno 15 giorni di anticipo tramite il portale HR.",
    "trasferta": "Le spese di trasferta si rimborsano entro 30 giorni allegando le ricevute nel portale spese.",
    "vpn":     "Per lavorare da remoto occorre collegarsi alla VPN aziendale con autenticazione a due fattori.",
}
tok = lambda s: re.findall(r"[a-zàèéìòù]+", s.lower())
vocab = sorted({w for d in DOCS.values() for w in tok(d)})

def embed(text):
    c = Counter(tok(text)); v = [c[w] for w in vocab]
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]

cosine = lambda a, b: sum(x * y for x, y in zip(a, b))   # vettori già normalizzati
index = {k: embed(t) for k, t in DOCS.items()}

def retrieve(q, k=1):
    qv = embed(q)
    return sorted(((cosine(qv, v), key) for key, v in index.items()), reverse=True)[:k]

if __name__ == "__main__":
    q = "Entro quanti giorni devo chiedere le ferie?"
    hits = retrieve(q, 2)
    print("risultati:", [(round(s, 2), k) for s, k in hits])
    ctx = "\n".join(DOCS[k] for _, k in hits[:1])
    print("\n--- PROMPT AUMENTATO ---")
    print(f"Rispondi SOLO usando il contesto.\nContesto:\n{ctx}\n\nDomanda: {q}")
