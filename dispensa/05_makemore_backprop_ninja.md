# Video 5 — makemore Part 4: Becoming a Backprop Ninja (1:55:24)
🔗 <https://www.youtube.com/watch?v=q8SA3rM6ckI> · Libro: [cap. 4, §4.4](../book/04_training_sano.md)

**Obiettivo:** calcolare **a mano** i gradienti di tutta la rete (MLP + BatchNorm), a livello di tensori, e confrontarli con PyTorch.

## Mappa dei capitoli (timestamp ufficiali)
| Tempo | Capitolo | Cosa si impara |
|---|---|---|
| 0:00 | intro: why you should care & fun history | la backprop "a mano" insegna a riconoscere i bug; breve storia |
| 7:26 | starter code | |
| 13:01 | exercise 1: backproping the atomic compute graph | si scompone il forward in passi minimi e si propaga un gradiente alla volta; verifica con `cmp` |
| 1:05:17 | digression: Bessel's correction in batchnorm | varianza con `n−1` (stimatore non distorto) vs `n`: incoerenza storica fra training e inferenza |
| 1:26:31 | exercise 2: cross entropy loss backward pass | formula chiusa: `dlogits = softmax − onehot`, diviso per N |
| 1:36:37 | exercise 3: batch norm layer backward pass | derivata analitica compatta della BN |
| 1:50:02 | exercise 4: putting it all together | training senza `loss.backward()` |
| 1:54:24 | outro | |

## Concetti chiave
- **La forma del gradiente = forma del tensore**: ottimo controllo di sanità.
- Broadcasting in avanti ⇒ **somma** all'indietro (e viceversa).
- Il prodotto matriciale `C = A@B` ha gradienti `dA = dC @ Bᵀ`, `dB = Aᵀ @ dC`.
- Le derivate chiuse (softmax+CE, BN) sono molto più semplici dell'espansione nodo per nodo.

## Formule
```
C = A @ B       →  dA = dC @ Bᵀ ;  dB = Aᵀ @ dC
softmax+CE      →  dlogits = (p − y_onehot) / N
broadcast fwd   →  somma sulle dimensioni broadcastate nel bwd
```

## Trappole
1. Dimenticare di sommare lungo le dimensioni broadcastate.
2. Confrontare con tolleranze sbagliate (`allclose` vs uguaglianza esatta).
3. `n` vs `n−1` nella varianza.

## Esercizi
1. Deriva a mano `d/dW` di un layer lineare + tanh e verifica con PyTorch.
2. Implementa la backward della LayerNorm (formula chiusa).

## Quiz
1. *Perché `dlogits = p − y`?* — la derivata di CE∘softmax si semplifica elegantemente.
2. *Perché fare backprop manuale se esiste autograd?* — per capire, diagnosticare gradienti esplosi/nulli e scrivere kernel custom.
