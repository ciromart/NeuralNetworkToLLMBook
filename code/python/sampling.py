"""Temperature, top-k e top-p (nucleus) da zero con NumPy: ciò che fa il 'sampler' di un LLM."""
import numpy as np

def softmax(z):
    z = z - z.max(); e = np.exp(z); return e / e.sum()

def sample(logits, temperature=1.0, top_k=None, top_p=None, rng=None):
    rng = rng or np.random.default_rng()
    if temperature == 0:                       # greedy
        return int(np.argmax(logits))
    p = softmax(np.asarray(logits, float) / temperature)
    order = np.argsort(-p)                     # indici dal più probabile
    keep = np.ones_like(p, dtype=bool)
    if top_k:
        keep[order[top_k:]] = False
    if top_p:
        cum = np.cumsum(p[order])
        cutoff = np.searchsorted(cum, top_p) + 1
        keep[order[cutoff:]] = False
    p = np.where(keep, p, 0.0)
    return int(rng.choice(len(p), p=p / p.sum()))

if __name__ == "__main__":
    vocab = ["il", "un", "lo", "la", "gli"]
    logits = [2.0, 1.0, 0.5, 0.1, -1.0]
    rng = np.random.default_rng(0)
    for cfg in [dict(temperature=0), dict(temperature=1.0), dict(temperature=1.0, top_k=2),
                dict(temperature=1.0, top_p=0.8), dict(temperature=2.0)]:
        draws = [vocab[sample(logits, rng=rng, **cfg)] for _ in range(2000)]
        freq = {w: round(draws.count(w) / 2000, 2) for w in vocab}
        print(cfg, freq)
