# Video 8 — State of GPT (Microsoft Build 2023, 42:40)
🔗 <https://www.youtube.com/watch?v=bZQun8Y4L2A> · Libro: [cap. 7](../book/07_dal_pretraining_all_assistente.md)

> ⚠️ Nessun capitolo ufficiale su YouTube per questo video. La struttura è ricostruita dalla descrizione («pipeline di training degli assistenti GPT: tokenizzazione, pretraining, supervised finetuning, RLHF») e dalla mia conoscenza: **verifica i dettagli guardando il video**.

**Obiettivo:** panoramica da conferenza su come si addestra un assistente tipo ChatGPT e come usarlo bene.

## Struttura ricostruita
| Parte | Contenuto |
|---|---|
| 1. Pipeline di training | pretraining → SFT → reward modeling → RL (RLHF) |
| 2. Dati e tokenizzazione | mix di web, libri, codice; testo → token → sequenze |
| 3. Pretraining | next-token prediction su trilioni di token; costo enorme; produce il *base model* |
| 4. SFT | dataset di dialoghi di alta qualità; trasforma il base model in assistente |
| 5. Reward model & RLHF | modello che predice preferenze umane; ottimizzazione contro quel modello |
| 6. Perché RLHF funziona | confrontare risposte è più facile che scriverle; contro: minor diversità |
| 7. Usare gli LLM bene | prompt chiari, *few-shot*, «lascia pensare il modello» (chain of thought), self-consistency, riflessione |
| 8. Strumenti e contesto | retrieval/plugin, memoria del contesto, calcolatrici/codice |
| 9. Limiti | allucinazioni, bias, errori di ragionamento, prompt injection, knowledge cutoff |
| 10. Raccomandazioni | usare come assistente con supervisione umana, in domini a basso rischio |

## Concetti chiave
- Il modello base **non è un assistente**: va "indirizzato" (SFT/RLHF).
- Ogni token ha lo **stesso budget di calcolo** → i compiti difficili vanno spezzati in più token (ragionamento esplicito).
- Il contesto è la *memoria di lavoro*; ciò che non c'è nel prompt può non esistere per il modello → RAG.

## Esercizi
1. Prendi un compito (es. estrazione dati) e confronta *zero-shot* vs *few-shot*.
2. Aggiungi «ragiona passo passo» e misura se la correttezza migliora.

## Quiz
1. *Perché il pretraining costa quasi tutto?* — enormi quantità di dati e calcolo; le fasi successive usano molti meno dati.
2. *Perché RLHF?* — è più facile per gli umani giudicare che scrivere risposte ideali.
