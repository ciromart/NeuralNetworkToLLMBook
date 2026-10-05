# Video 4 — makemore Part 3: attivazioni, gradienti, BatchNorm (1:55:58)
🔗 <https://www.youtube.com/watch?v=P6sfmUTpUmc> · Libro: [cap. 4](../book/04_training_sano.md)

**Obiettivo:** «ispezionare» una rete profonda: statistiche delle attivazioni e dei gradienti, inizializzazione corretta, BatchNorm.

## Mappa dei capitoli (timestamp ufficiali)
| Tempo | Capitolo | Cosa si impara |
|---|---|---|
| 0:00 | intro | |
| 1:22 | starter code | MLP del video precedente |
| 4:19 | fixing the initial loss | loss iniziale attesa ≈ `−ln(1/27) ≈ 3.29`; se è 27 i logits iniziali sono troppo grandi → scala W2 (×0.01) e b2=0 |
| 12:59 | fixing the saturated tanh | pre-attivazioni troppo ampie → tanh a ±1 → gradiente ≈ 0 («neuroni morti») |
| 27:53 | calculating the init scale: Kaiming init | `std = gain/√fan_in` (gain 5/3 per tanh) mantiene la varianza stabile |
| 40:40 | batch normalization | normalizza per batch (media 0, std 1) + `γ`, `β` appresi; running mean/std per l'inferenza |
| 1:03:07 | batch normalization: summary | |
| 1:04:50 | real example: resnet50 walkthrough | dove stanno conv + BN + residui in una rete vera |
| 1:14:10 | summary of the lecture | |
| 1:18:35 | part 2: PyTorch-ifying the code | `Linear`, `BatchNorm1d`, `Tanh` come moduli |
| 1:26:51 | viz #1: forward pass activations statistics | istogrammi e % di saturazione per strato |
| 1:30:54 | viz #2: backward pass gradient statistics | i gradienti devono avere scala simile tra strati |
| 1:32:07 | the fully linear case of no non-linearities | perché serve un *gain* |
| 1:36:15 | viz #3: parameter activation and gradient statistics | |
| 1:39:55 | viz #4: update:data ratio over time | rapporto ≈ 10⁻³ è un buon segno |
| 1:46:04 | bringing back batchnorm, looking at the visualizations | BN rende la rete robusta a init e lr |
| 1:51:34 | summary of the lecture for real this time | |

## Concetti chiave
- Un'inizializzazione sbagliata spreca le prime migliaia di step.
- Stabilità = **varianza costante** tra strati (forward) e tra gradienti (backward).
- BatchNorm stabilizza ma introduce **accoppiamento tra esempi del batch** (fonte di bug sottili) → alternative: LayerNorm, GroupNorm.

## Formule
```
BN(x) = γ · (x − μ_B)/√(σ²_B + ε) + β
Kaiming: W ~ N(0, (gain/√fan_in)²)
```

## Trappole
1. Bias prima di una BN è inutile (la BN lo cancella).
2. In inferenza usare le **running statistics**, non quelle del batch.
3. Training/eval mode dimenticati.

## Esercizi
1. Riproduci gli istogrammi delle attivazioni con e senza gain corretto.
2. Misura il rapporto update/data e trova un `lr` fuori scala.
3. Sostituisci BN con LayerNorm: cosa cambia in training e inferenza?

## Quiz
1. *Perché la loss iniziale deve essere ~3.3?* — a pesi casuali il modello deve essere "agnostico"; una loss molto più alta significa fiducia ingiustificata.
2. *Cosa rende satura una tanh?* — ingressi di grande modulo: derivata `1−tanh²` ≈ 0.
3. *Perché LayerNorm nei Transformer e non BN?* — non dipende dal batch (sequenze di lunghezze diverse, batch piccoli, generazione token per token).
