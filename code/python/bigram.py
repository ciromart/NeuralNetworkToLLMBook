"""Language model bigramma a conteggi (video 2, makemore)."""
import random
from collections import defaultdict

words = ["emma", "olivia", "ava", "isabella", "sophia", "mia", "amelia", "harper",
         "evelyn", "abigail", "emily", "ella", "elizabeth", "camila", "luna", "sofia"]

counts = defaultdict(lambda: defaultdict(int))
for w in words:
    chars = ["."] + list(w) + ["."]          # "." = inizio/fine parola
    for a, b in zip(chars, chars[1:]):
        counts[a][b] += 1

def sample(rng):
    out, cur = [], "."
    while True:
        nxt = counts[cur]
        cur = rng.choices(list(nxt), weights=list(nxt.values()))[0]
        if cur == ".": return "".join(out)
        out.append(cur)

import math
nll, n = 0.0, 0
for w in words:
    chars = ["."] + list(w) + ["."]
    for a, b in zip(chars, chars[1:]):
        p = counts[a][b] / sum(counts[a].values())
        nll -= math.log(p); n += 1
print(f"NLL media (loss): {nll / n:.3f}")
rng = random.Random(42)
print("generati:", [sample(rng) for _ in range(6)])
