# Capitolo 7 — Dal pretraining all'assistente

*Video 8: «State of GPT» (42 min, Microsoft Build) · Video 10: «Let's reproduce GPT-2 (124M)» (4h01)*

## 7.1 La pipeline in quattro stadi

![Pipeline](figures/11_pipeline_training.png)

| Stadio | Dati | Obiettivo | Risultato |
|---|---|---|---|
| **Pretraining** | trilioni di token (web, libri, codice) | prevedere il token successivo | *modello base*: completa testo, non risponde |
| **SFT** (Supervised Fine-Tuning) | decine/centinaia di migliaia di dialoghi di qualità | imitare risposte ideali | modello che segue istruzioni |
| **Reward model** | confronti umani «A è meglio di B» | predire la preferenza | un "giudice" numerico |
| **RLHF / DPO** | prompt + giudice | massimizzare la preferenza | assistente allineato, utile e sicuro |

Il 99% del calcolo (e del costo) è nel **pretraining**. Le fasi successive sono più economiche e "modellano il comportamento".

## 7.2 Riprodurre GPT-2 (124M): cosa serve davvero
Nel video si ricostruisce GPT-2 small (12 layer, 12 teste, embedding 768, contesto 1024) e lo si allena su FineWeb-Edu. Le ottimizzazioni sono la lezione più preziosa:

| Ottimizzazione | Effetto |
|---|---|
| Precisione mista (**bf16**/TF32) | meno memoria e molto più veloce su GPU moderne |
| `torch.compile` | fonde operazioni, meno accessi alla memoria |
| **FlashAttention** | attention senza materializzare la matrice T×T |
| Vocabolario "numero bello" (50257→50304) | multiplo di 64: kernel GPU più efficienti |
| AdamW + weight decay 0.1 | ottimizzatore standard degli LLM |
| Warmup + cosine decay del `lr` | stabilità all'inizio, convergenza alla fine |
| **Gradient clipping** (norma ≤ 1) | evita esplosioni |
| Gradient accumulation | emula batch enormi (~0.5M token) con poca VRAM |
| DDP multi-GPU | scala su più schede |
| *Weight tying* embedding/uscita | meno parametri, miglior qualità |

![Scaling](figures/12_scaling.png)

Le **scaling law** (Kaplan 2020, Chinchilla 2022): la loss scende prevedibilmente come legge di potenza con parametri, dati e calcolo. Per questo i laboratori "scommettono" su modelli sempre più grandi.

## 7.3 Allucinazioni e limiti (dal "State of GPT")
- Un LLM è un **simulatore di testo plausibile**, non un database: può inventare.
- Il contesto è la sua **memoria di lavoro**: ciò che non è nel prompt, per lui spesso non esiste → **RAG**.
- Soggetto a *prompt injection*, bias, errori di ragionamento: servono verifiche e guardrail (cap. 14).
- Suggerimento di Karpathy: usare gli LLM come **assistenti con supervisione**, con prompt chiari, esempi (*few-shot*), strumenti esterni e controllo umano.
