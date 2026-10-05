# Video 6 — makemore Part 5: Building a WaveNet (56:22)
🔗 <https://www.youtube.com/watch?v=t3YJ5hKiMQ0> · Libro: [cap. 4, §4.5](../book/04_training_sano.md)

> ⚠️ YouTube **non espone capitoli** per questo video: la mappa sotto è **ricostruita da me** dal contenuto descritto («MLP reso più profondo con struttura ad albero, simile a un'architettura convoluzionale tipo WaveNet») e va verificata guardando il video.

**Obiettivo:** rendere il modello più profondo fondendo il contesto **in modo gerarchico** (a coppie), anticipando le reti convoluzionali.

## Mappa ricostruita
| Parte | Cosa si impara |
|---|---|
| Ripartenza dal codice dei video precedenti | contesto portato a 8 caratteri; ricomposizione in moduli |
| Moduli stile PyTorch (`Embedding`, `FlattenConsecutive`, `Sequential`) | si costruisce una mini-libreria di layer |
| Fusione gerarchica | invece di concatenare 8 caratteri in un colpo, si fondono 2 per volta in 3 livelli |
| Bug con BatchNorm su tensori a 3 dimensioni | le statistiche vanno calcolate su più dimensioni (batch e posizione) |
| Confronto loss | il modello gerarchico migliora a parità di parametri, ma l'allenamento richiede più cura |
| Cenni a convoluzioni causali dilatate | lo stesso schema, calcolato in modo efficiente |

## Concetti chiave
- **Gerarchia**: info locali fuse progressivamente (come i filtri dilatati di WaveNet).
- Costruire **blocchi riusabili** (`Sequential`) riduce i bug.
- Forma dei tensori e dimensioni di BN sono la fonte principale di errori.
- Il training resta lento senza un buon **set-up sperimentale** (dev loss, curve) — Karpathy lo evidenzia come tema aperto.

## Trappole
1. BN che calcola le statistiche solo sull'ultima dimensione dopo il reshape.
2. Perdere il collegamento tra la forma `(B, T, C)` e ciò che significa ogni asse.

## Esercizi
1. Cambia il contesto a 16 e aggiungi un livello di fusione.
2. Confronta una concatenazione "piatta" con quella gerarchica a parità di parametri.

## Quiz
1. *Qual è il vantaggio della fusione a coppie?* — il campo recettivo cresce esponenzialmente con la profondità, con pochi parametri.
2. *Cosa viene poi sostituito dall'attention?* — la fusione fissa e locale diventa una selezione appresa e dinamica di *quali* posizioni combinare.
