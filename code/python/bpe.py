"""Byte Pair Encoding minimale (video 9, 'GPT Tokenizer')."""
from collections import Counter

def merge(ids, pair, new_id):
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            out.append(new_id); i += 2
        else:
            out.append(ids[i]); i += 1
    return out

def train(text, n_merges):
    ids, merges = list(text.encode("utf-8")), {}
    for k in range(n_merges):
        pair, _ = Counter(zip(ids, ids[1:])).most_common(1)[0]
        merges[pair] = 256 + k
        ids = merge(ids, pair, 256 + k)
    return merges

def encode(text, merges):
    ids = list(text.encode("utf-8"))
    for pair, new_id in merges.items():   # ordine di inserimento = ordine di training
        ids = merge(ids, pair, new_id)
    return ids

if __name__ == "__main__":
    text = "aaabdaaabac aaabdaaabac"
    m = train(text, 3)
    print("merge appresi:", m)
    print("byte originali:", len(text.encode()), "-> token:", len(encode(text, m)))
