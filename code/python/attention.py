"""Self-attention causale a una testa in NumPy (video 7, 'Let's build GPT')."""
import numpy as np

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def causal_self_attention(x, Wq, Wk, Wv):
    T, _ = x.shape
    q, k, v = x @ Wq, x @ Wk, x @ Wv
    scores = q @ k.T / np.sqrt(k.shape[-1])              # (T, T) similarità
    mask = np.tril(np.ones((T, T), dtype=bool))           # vietato guardare il futuro
    scores = np.where(mask, scores, -np.inf)
    att = softmax(scores)                                 # righe che sommano a 1
    return att @ v, att

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    T, C, H = 5, 8, 4
    x = rng.normal(size=(T, C))
    out, att = causal_self_attention(x, *(rng.normal(size=(C, H)) for _ in range(3)))
    np.set_printoptions(precision=2, suppress=True)
    print("pesi di attenzione (triangolare inferiore):\n", att)
    print("somma righe:", att.sum(axis=1), " output shape:", out.shape)
