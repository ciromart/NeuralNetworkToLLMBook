# Video 7 — Let's build GPT: da zero, in codice (1:56:20)
🔗 <https://www.youtube.com/watch?v=kCc8FmEb1nY> · Codice: [`attention.py`](../code/python/attention.py) · Libro: [cap. 5](../book/05_transformer_gpt.md)

**Obiettivo:** costruire un Transformer *decoder-only* (stile GPT) su testo (Shakespeare, a livello di carattere), seguendo «Attention Is All You Need».

## Mappa dei capitoli (timestamp ufficiali)
| Tempo | Capitolo | Cosa si impara |
|---|---|---|
| 0:00 | intro: ChatGPT, Transformers, nanoGPT, Shakespeare | |
| 7:52 | reading and exploring the data | dataset testuale, vocabolario di caratteri |
| 9:28 | tokenization, train/val split | encode/decode a caratteri, split 90/10 |
| 14:27 | data loader: batches of chunks | campioni casuali di finestre `block_size`; target = input spostato di 1 |
| 22:11 | simplest baseline: bigram language model | riferimento minimo; loss iniziale attesa ≈ 4.17 (`ln 65`) |
| 34:53 | training the bigram model | |
| 38:00 | port our code to a script | |
| 42:13 | version 1: averaging past context with for loops | media dei token precedenti: forma più debole di aggregazione |
| 47:11 | the trick: matrix multiply as weighted aggregation | matrice triangolare × X = medie cumulative |
| 51:54 | version 2: using matrix multiply | |
| 54:42 | version 3: adding softmax | `masked_fill(-inf)` + softmax → pesi che sommano a 1 |
| 58:26 | minor code cleanup | |
| 1:00:18 | positional encoding | embedding di posizione appreso |
| 1:02:00 | **THE CRUX: version 4: self-attention** | Q, K, V; pesi dipendenti dai dati; testa singola |
| 1:11:38 | note 1: attention as communication | nodi di un grafo orientato che si scambiano messaggi |
| 1:12:46 | note 2: no notion of space, operates over sets | per questo servono le posizioni |
| 1:13:40 | note 3: no communication across batch dimension | gli esempi del batch sono indipendenti |
| 1:14:14 | note 4: encoder vs decoder blocks | la maschera triangolare distingue il decoder |
| 1:15:39 | note 5: attention vs self- vs cross-attention | dove nascono Q, K, V |
| 1:16:56 | note 6: "scaled" attention, why divide by √head_size | tiene la varianza ≈ 1, softmax non satura |
| 1:19:11 | inserting a single self-attention block | |
| 1:21:59 | multi-headed self-attention | più teste in parallelo, concatenate |
| 1:24:25 | feedforward layers of the transformer block | calcolo per-token dopo la comunicazione |
| 1:26:48 | residual connections | `x + f(x)` + proiezione |
| 1:32:51 | layernorm (and its relation to batchnorm) | normalizza per token; pre-norm |
| 1:37:49 | scaling up the model, adding dropout | iperparametri, regolarizzazione |
| 1:42:39 | encoder vs decoder vs both Transformers | |
| 1:46:22 | quick walkthrough of nanoGPT | teste "batched" in un unico tensore |
| 1:48:53 | back to ChatGPT, GPT-3, pretraining vs finetuning, RLHF | quello costruito è solo il **pretraining** |
| 1:54:32 | conclusions | |

## Concetti chiave
- `softmax(QKᵀ/√d)·V` con **maschera causale**.
- **Comunicazione** (attention) alternata a **calcolo** (MLP).
- Residui + LayerNorm = reti profonde allenabili.
- Il modello costruito è un *base model*: completa testo, non risponde a istruzioni.

## Formule
```
Attention(Q,K,V) = softmax( QKᵀ/√d_k + M ) V     M = 0 sotto/sulla diagonale, −∞ sopra
Block:  x = x + MHA(LN(x));  x = x + MLP(LN(x))
```

## Trappole
1. Dimenticare la maschera → il modello "bara" vedendo il futuro (loss irrealisticamente bassa).
2. Non dividere per `√d` → softmax satura.
3. Contesto più lungo del `block_size` a generazione: tagliare l'input.

## Esercizi
1. Disegna a mano la matrice dei pesi per 4 token e confrontala con [`attention.py`](../code/python/attention.py).
2. Rimuovi i residui e osserva quanto peggiora il training.
3. Aggiungi la generazione con temperatura e top-k ([`sampling.py`](../code/python/sampling.py)).

## Quiz
1. *Query, Key, Value in una frase?* — Q: cosa cerco; K: cosa offro; V: cosa trasmetto.
2. *Perché serve la codifica posizionale?* — l'attention è invariante all'ordine.
3. *Perché 4× nell'MLP?* — più capacità di calcolo per token; scelta storica del paper.
