# Video 2 — makemore: il modello bigramma (1:57:45)
🔗 <https://www.youtube.com/watch?v=PaCmpygFfXo> · Codice: [`bigram.py`](../code/python/bigram.py) · Libro: [cap. 2](../book/02_language_modeling_bigram.md)

**Obiettivo:** primo language model a livello di carattere (generare nomi), prima con i **conteggi**, poi con una **rete neurale** — arrivando allo stesso risultato.

## Mappa dei capitoli (timestamp ufficiali)
| Tempo | Capitolo | Cosa si impara |
|---|---|---|
| 0:00 | intro | makemore: «fa più cose come i nomi» |
| 3:03 | reading and exploring the dataset | ~32k nomi, lunghezze, caratteri |
| 6:24 | exploring the bigrams | coppie di caratteri consecutivi con marcatori di inizio/fine |
| 9:24 | counting bigrams in a python dictionary | tabella dei conteggi |
| 12:45 | counting bigrams in a 2D torch tensor | la tabella diventa matrice 27×27 («training» = contare) |
| 18:19 | visualizing the bigram tensor | heatmap (vedi [fig. 5](../book/figures/05_bigram_heatmap.png)) |
| 20:54 | deleting spurious (S) and (E) tokens → single `.` | un solo simbolo per inizio e fine |
| 24:02 | sampling from the model | `torch.multinomial` dalle righe normalizzate |
| 36:17 | efficiency! vectorized normalization, broadcasting | regole di broadcasting e `keepdim=True` (bug classico) |
| 50:14 | loss function (negative log likelihood) | likelihood come prodotto, log → somma, media negativa |
| 1:00:50 | model smoothing with fake counts | +1 ai conteggi per evitare probabilità zero (loss infinita) |
| 1:02:57 | PART 2: the neural network approach | stesso obiettivo con pesi allenabili |
| 1:05:26 | creating the bigram dataset for the neural net | coppie (x, y) come interi |
| 1:10:01 | one-hot encodings | gli interi non si danno in pasto a una rete: si codificano |
| 1:13:53 | one linear layer of neurons via matrix multiplication | `logits = x_onehot @ W` |
| 1:18:46 | the softmax | logits → exp → normalizza = probabilità |
| 1:26:17 | summary, preview, reference to micrograd | |
| 1:35:49 | vectorized loss | selezionare la probabilità del carattere corretto |
| 1:38:36 | backward and update, in PyTorch | `loss.backward()`, update dei pesi |
| 1:42:55 | putting everything together | training completo |
| 1:47:49 | note 1: one-hot just selects a row of W | l'embedding è una **lookup** |
| 1:50:18 | note 2: smoothing as regularization | `W²` come penalità ≈ fake counts |
| 1:54:31 | sampling from the neural net | stessi risultati del metodo a conteggi |
| 1:56:16 | conclusion | |

## Concetti chiave
- **Language model** = distribuzione del prossimo simbolo dato il contesto.
- **NLL** come misura di qualità; la loss di un modello uniforme su 27 simboli è `ln 27 ≈ 3.30`.
- **Softmax** = trasformazione differenziabile logits→probabilità.
- Conteggi e rete **sono lo stesso modello**; la rete però scala a contesti più lunghi.
- **Regolarizzazione** ≈ smoothing.

## Trappole
1. Broadcasting sbagliato: dimenticare `keepdim=True` normalizza lungo l'asse errato *senza errori*.
2. Probabilità 0 → `log(0) = −∞` (serve smoothing).
3. Valutare sulla stessa loss di training.

## Esercizi
1. Calcola la NLL con e senza smoothing e osserva l'effetto.
2. Implementa il **trigramma** a conteggi: quanto cresce la tabella? Quanto migliora la loss?
3. Dimostra numericamente che one-hot × W seleziona una riga di W.

## Quiz
1. *Perché il log nella loss?* — trasforma il prodotto di probabilità in somma, stabile e differenziabile.
2. *Perché una rete anziché una tabella?* — la tabella cresce esponenzialmente col contesto; la rete generalizza con pochi parametri.
3. *Che loss ha un modello casuale su 27 caratteri?* — circa 3.30.
