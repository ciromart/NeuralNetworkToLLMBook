# Video 1 — micrograd: reti neurali e backpropagation (2:25:52)
🔗 <https://www.youtube.com/watch?v=VMj-3S1tku0> · Codice: [`micrograd_mini.py`](../code/python/micrograd_mini.py) · Libro: [cap. 1](../book/01_reti_neurali_backprop.md)

**Obiettivo:** capire la backpropagation costruendo da zero un motore di autograd scalare (~100 righe) e allenare una piccola rete.

## Mappa dei capitoli (timestamp ufficiali)
| Tempo | Capitolo (titolo originale) | Cosa si impara |
|---|---|---|
| 0:00 | intro | scopo: capire cosa c'è *dentro* `loss.backward()` |
| 0:25 | micrograd overview | una rete è un'espressione matematica; micrograd costruisce il grafo e calcola i gradienti |
| 8:08 | derivative of a simple function with one input | derivata come rapporto incrementale numerico: «se sposto x di h, quanto cambia f?» |
| 14:12 | derivative of a function with multiple inputs | derivate parziali: ogni ingresso ha la sua sensibilità |
| 19:09 | starting the core Value object and its visualization | classe `Value` che ricorda *chi l'ha prodotto* (figli e operazione) → grafo di calcolo |
| 32:10 | manual backpropagation example #1 | chain rule a mano su un'espressione semplice; `+` distribuisce il gradiente, `*` lo scambia con l'altro operando |
| 51:10 | preview of a single optimization step | spingere i dati nella direzione del gradiente fa salire/scendere l'output |
| 52:52 | manual backpropagation example #2: a neuron | backprop attraverso `tanh(Σ wx + b)` |
| 1:09:02 | implementing the backward function for each operation | ogni operazione definisce la sua `_backward` locale |
| 1:17:32 | … for a whole expression graph | ordinamento topologico, poi `_backward` in ordine inverso |
| 1:22:28 | fixing a backprop bug when one node is used multiple times | i gradienti vanno **accumulati** (`+=`), non assegnati |
| 1:27:05 | breaking up a tanh, exercising with more operations | tanh scomponibile in exp/divisioni; si aggiungono `**`, `exp`, `/`, `-` |
| 1:39:31 | doing the same thing but in PyTorch | stessi gradienti con tensori e `requires_grad` |
| 1:43:55 | building out a neural net library (MLP) | classi `Neuron`, `Layer`, `MLP` |
| 1:51:04 | creating a tiny dataset, writing the loss function | 4 esempi, loss = somma errori² |
| 1:57:56 | collecting all of the parameters | `parameters()` per poterli aggiornare |
| 2:01:12 | gradient descent manually, training the network | ciclo forward → azzera grad → backward → update; learning rate troppo alto = instabile |
| 2:14:03 | summary of what we learned | reti moderne = stessa idea, tensori e scala |
| 2:16:46 | walkthrough of the full code on GitHub | micrograd reale |
| 2:21:10 | diving into PyTorch, backward pass for tanh | in PyTorch la derivata di tanh è definita allo stesso modo |
| 2:24:39 / 2:25:20 | conclusion / outtakes | — |

## Concetti chiave
- **Derivata = sensibilità** dell'output a una variazione dell'input.
- **Chain rule:** `dL/dx = dL/dy · dy/dx`; la backprop la applica a ritroso nel grafo.
- Regole locali: somma → gradiente copiato; prodotto → gradiente × altro operando; `tanh` → `1 − t²`.
- **Accumulo** dei gradienti quando un nodo ha più usi.
- Training = ripetere forward/backward/update.

## Formule
```
f'(x) ≈ (f(x+h) − f(x)) / h
d(a·b)/da = b        d(tanh x)/dx = 1 − tanh²(x)
w ← w − η · ∂L/∂w
```

## Trappole
1. Dimenticare di **azzerare** i gradienti tra un passo e l'altro.
2. `=` invece di `+=` nell'accumulo (bug mostrato nel video).
3. `η` troppo grande → la loss oscilla/esplode.

## Esercizi
1. Aggiungi `exp` e `__truediv__` a `Value` e verifica con differenze finite.
2. Allena la rete su XOR; con quanti neuroni nascosti converge?
3. Confronta i gradienti con quelli di PyTorch su 3 espressioni diverse.

## Quiz
1. *Perché `+` copia il gradiente?* — `d(a+b)/da = 1`, quindi il gradiente in uscita passa invariato.
2. *Perché serve l'ordinamento topologico?* — il gradiente di un nodo si può propagare solo dopo che tutti i nodi che lo usano sono stati processati.
3. *Cosa succede se non azzeri i gradienti?* — si sommano a quelli dei passi precedenti: update sbagliati.
