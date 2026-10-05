# Video 3 — makemore Part 2: MLP (1:15:40)
🔗 <https://www.youtube.com/watch?v=TCH_1BHY58I> · Libro: [cap. 3](../book/03_mlp_embedding.md)

**Obiettivo:** passare da 1 carattere di contesto a più caratteri con un **MLP + embedding** (idea di Bengio et al., 2003) e imparare i fondamentali dell'allenamento.

## Mappa dei capitoli (timestamp ufficiali)
| Tempo | Capitolo | Cosa si impara |
|---|---|---|
| 0:00 | intro | |
| 1:48 | Bengio et al. 2003 paper walkthrough | embedding condivisi + MLP + softmax sul vocabolario |
| 9:03 | (re-)building our training dataset | finestre di `block_size` caratteri → carattere successivo |
| 12:19 | implementing the embedding lookup table | matrice `C`; indicizzare `C[X]` dà tensori (N, block, dim) |
| 18:35 | hidden layer + internals of torch.Tensor: storage, views | `view()` non copia: stesso storage, diverso shape; concatenare gli embedding con `view` |
| 29:15 | implementing the output layer | strato lineare → logits |
| 29:53 | implementing the negative log likelihood loss | |
| 32:17 | summary of the full network | |
| 32:49 | introducing `F.cross_entropy` and why | più efficiente e **numericamente stabile** (sottrae il massimo) |
| 37:56 | training loop, overfitting one batch | sanity check: la rete deve poter memorizzare un solo batch |
| 41:25 | training on the full dataset, minibatches | gradiente su sottoinsiemi: più rumoroso ma molto più rapido |
| 45:40 | finding a good initial learning rate | scansione esponenziale di lr e curva loss-vs-lr |
| 53:20 | train/val/test splits and why | train per i pesi, dev per gli iperparametri, test una sola volta |
| 1:00:49 | experiment: larger hidden layer | più capacità → rischio overfit |
| 1:05:27 | visualizing the character embeddings | vocali vicine tra loro, `.` isolato (vedi [fig. 6](../book/figures/06_embedding.png), illustrativa) |
| 1:07:16 | experiment: larger embedding size | |
| 1:11:46 | summary of our final code | |
| 1:13:24 | sampling from the model | |
| 1:14:55 | google colab notebook | |

## Concetti chiave
- **Embedding**: tabella appresa; indicizzare = selezionare una riga (equivale a one-hot × W).
- **Mini-batch**, **split train/dev/test**, **learning-rate finding**, **overfitting**.
- `cross_entropy` fonde softmax + NLL.

## Formule
```
emb = C[X]                      # (N, block, d)
h   = tanh(emb.view(N,-1) @ W1 + b1)
logits = h @ W2 + b2
loss = cross_entropy(logits, Y)
```

## Trappole
1. Valutare la loss sul train e credere che il modello sia bravo.
2. Cambiare iperparametri guardando il **test**: lo si "consuma".
3. `view` su tensori non contigui.

## Esercizi
1. Cerca il miglior `lr` con la scansione esponenziale e traccia il grafico.
2. Porta `block_size` da 3 a 5 e `emb_dim` da 2 a 10: cosa migliora?
3. Visualizza gli embedding a 2 dimensioni e commenta i cluster.

## Quiz
1. *Perché `cross_entropy` invece di softmax + log a mano?* — evita overflow (usa log-sum-exp) ed è più veloce.
2. *Perché provare a "overfittare" un batch?* — verifica che modello e loop di training funzionino.
3. *A cosa serve il dev set?* — scegliere gli iperparametri senza toccare il test.
