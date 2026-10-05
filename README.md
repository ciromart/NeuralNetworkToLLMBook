# Dalla rete neurale al LLM locale — con Spring AI

Libro/guida in italiano: recap della playlist **Neural Networks: Zero to Hero** (Andrej Karpathy) + come usare un LLM locale da **Java / Spring AI** e servizi **REST**.

## Indice
| # | Capitolo | Video della playlist |
|---|---|---|
| 0 | [Prefazione, mappa e le 3 strade per un LLM locale](book/00_prefazione.md) | — |
| 1 | [Reti neurali e backpropagation](book/01_reti_neurali_backprop.md) | micrograd |
| 2 | [Language modeling: bigramma](book/02_language_modeling_bigram.md) | makemore 1 |
| 3 | [MLP ed embedding](book/03_mlp_embedding.md) | makemore 2 |
| 4 | [Far funzionare il training](book/04_training_sano.md) | makemore 3, 4, 5 |
| 5 | [Il Transformer: GPT da zero](book/05_transformer_gpt.md) | Let's build GPT |
| 6 | [Il tokenizer](book/06_tokenizer.md) | GPT Tokenizer |
| 7 | [Dal pretraining all'assistente](book/07_dal_pretraining_all_assistente.md) | State of GPT, GPT-2 (124M) |
| 8 | [LLM locale: inferenza, LoRA, RAG](book/08_llm_locale.md) | — |
| 9 | [**Installare Spring AI**](book/09_installare_spring_ai.md) | — |
| 10 | [**Fondamenti per gestire un LLM**](book/10_fondamenti_gestire_llm.md) | — |
| 11 | [Chat, streaming, memoria, structured output](book/11_spring_ai_chat.md) | — |
| 12 | [RAG con Spring AI](book/12_rag_spring_ai.md) | — |
| 13 | [Tool calling e API REST](book/13_tool_calling_rest.md) | — |
| 14 | [Best practice enterprise](book/14_best_practice.md) | — |
| A | [**Digressioni su Python**](book/A_digressioni_python.md) | — |
| B | [Glossario](book/B_glossario.md) | — |

## 📘 Dispensa unica
**[`DISPENSA_COMPLETA.md`](DISPENSA_COMPLETA.md)** (e **[`DISPENSA_COMPLETA.pdf`](DISPENSA_COMPLETA.pdf)**, 96 pagine con diagrammi e figure): un solo documento con schede dei 10 video, teoria, **codice vero incluso dai file eseguiti con il loro output reale**, e tutto il progetto Spring AI. Si rigenera con `python3 scripts/build_dispensa.py` (poi `scripts/build_html_pdf.py` per il PDF).

## Dispensa video per video
[`dispensa/`](dispensa/00_come_usare.md): una scheda per ognuno dei 10 video con **timestamp ufficiali dei capitoli**, spiegazioni in italiano, formule, trappole, esercizi e quiz. *Non è una trascrizione* (vedi la nota iniziale della dispensa).

## Codice
- `code/python/` — micrograd, bigramma, attention, BPE, sampling, mini-RAG, client Ollama + esempi PyTorch (`v01_gradcheck`, `v03_mlp_nomi`, `v04_init_batchnorm`, `v07_mini_gpt`, `v08_lora_da_zero`). Output reali in `code/python/outputs/`. Richiedono `pip install numpy torch`.
- `code/spring-ai-demo/` — Spring Boot 4.1.1 + Spring AI 2.0.1 + Java 21: `mvn test` · `mvn spring-boot:run` (serve Ollama: `docker compose up -d`).
- `scripts/make_figures.py` — rigenera le 18 figure in `book/figures/` (`pip install matplotlib numpy`).

## Stato della verifica
Eseguiti: script Python, compilazione e test (2/2) del progetto Spring. **Non** eseguito: dialogo end-to-end con un modello Ollama reale (nessun modello nel sandbox).

I diagrammi `mermaid` si visualizzano direttamente su GitHub.
