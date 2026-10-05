# Dalla rete neurale al LLM locale — con Spring AI

> Guida pratica in italiano: come nasce un LLM (percorso *Neural Networks: Zero to Hero* di Andrej Karpathy) e come usarlo da Java con **Spring AI** e servizi **REST**.

## 0.1 Ho capito la richiesta? Sì — ecco la mia lettura

| Tu hai chiesto | Come l'ho tradotto |
|---|---|
| Recap della playlist YouTube | La playlist è **"Neural Networks: Zero to Hero"** (Andrej Karpathy, 10 video). I capitoli 1–7 la seguono in ordine, con codice Python eseguibile e grafici. |
| "Parti dal principio e spiega come è costruito un LLM" | Neurone → backpropagation → language model → MLP/embedding → attivazioni → Transformer → tokenizer → pretraining → SFT/RLHF. |
| "Creazione di un LLM locale" | Capitolo 8: le **3 strade realistiche** (inferenza locale con Ollama, fine-tuning LoRA/QLoRA, RAG) e perché non si addestra da zero. |
| Java + Spring AI + REST | Capitoli 9–13 + progetto Maven funzionante in `code/spring-ai-demo` (chat, streaming, structured output, RAG, tool calling verso API REST). |
| Installare Spring AI e fondamenti per gestire un LLM | Capitolo 9 (installazione passo-passo) e 10 (fondamenti operativi: token, contesto, temperatura, memoria, streaming, risorse). |
| Immagini, grafici, mockup, flow model | 18 figure in `book/figures/` (generate da `scripts/make_figures.py`, i grafici dati usano dati reali dagli script) + diagrammi **Mermaid** per i flussi. |
| "Usa tutte le skill Java/Spring per le best practice" | In questa sessione non ho skill specifiche per Java/Spring: ho applicato le best practice che conosco (cap. 13) e **verificato** che il codice compili e i test passino. |

### Una precisazione importante
Il testo che hai incollato dice «il corso di Spring che hai visto con Ollama». Io non ho visibilità su quel corso: ho quindi basato il libro sulla playlist indicata e sulla documentazione Spring AI. Se vuoi che il libro segua anche quel corso, indicami titolo/link.

### Cosa ho (e non ho) verificato
- ✅ Gli script Python (`code/python/`) **sono stati eseguiti**; i grafici dei capitoli 1, 2, 5, 6 usano i loro output reali.
- ✅ Il progetto Spring (Spring Boot 4.1.1, Spring AI 2.0.1, Java 21) **compila** e i **2 test** passano.
- ⚠️ Non ho potuto far girare un modello Ollama reale nel sandbox: gli endpoint di chat/RAG sono compilati ma non provati end-to-end. Il capitolo 9 ha una checklist per provarli sulla tua macchina.
- ⚠️ I grafici "illustrativi" (embedding 2D, scaling law) sono dichiarati tali nella didascalia.

## 0.2 Mappa del libro

```mermaid
flowchart TB
    subgraph B["Basi (video 1-6)"]
        direction LR
        A[1 Neurone e backprop] --> B2[2 Bigramma] --> C[3 MLP + embedding] --> D[4 Training sano]
    end
    subgraph G["Dai Transformer ai LLM (video 7-10)"]
        direction LR
        E[5 Transformer / GPT] --> F[6 Tokenizer BPE] --> G2[7 Pretraining, SFT, RLHF]
    end
    subgraph S["LLM locale e Spring AI"]
        direction LR
        H[8 LLM locale] --> I[9 Installare Spring AI] --> J[10 Fondamenti] --> K[11 Chat e streaming]
        K --> L[12 RAG] --> M[13 Tool calling REST] --> N[14 Best practice]
    end
    B --> G --> S
```

## 0.3 Le tre strade per "avere un LLM locale"

Nel 99% dei casi un backend developer **non addestra un modello da zero** (milioni di dollari, cluster di GPU). Le strade reali:

```mermaid
flowchart TD
    Q{Che problema devo risolvere?}
    Q -->|Voglio solo usare un modello,<br/>dati che restano in casa| I[1. Inferenza locale<br/>Ollama + modello open-weight]
    Q -->|Il modello deve parlare<br/>con il MIO stile/dominio| F[2. Fine-tuning<br/>LoRA / QLoRA]
    Q -->|Deve rispondere su documenti<br/>che cambiano spesso| R[3. RAG<br/>vector store + Spring AI]
    I --> S[Spring AI ChatClient]
    F --> I
    R --> S
```

| Strada | Costo | Quando | Dati aggiornabili? |
|---|---|---|---|
| Inferenza (Ollama) | Basso (una GPU/CPU decente) | Sempre, è il punto di partenza | No (conoscenza congelata) |
| Fine-tuning LoRA | Medio (1 GPU + dataset curato) | Stile, formato, compiti ripetitivi | Richiede nuovo training |
| **RAG** | Basso-medio | **Il caso enterprise più comune** | **Sì, istantaneamente** |

**Regola pratica:** parti da RAG + prompt buono. Fai fine-tuning solo se misuri che RAG non basta.

## 0.4 Come leggere il libro
- Se non conosci il machine learning: leggi 1→8 in ordine.
- Se sei solo uno sviluppatore Java con fretta: leggi 0.3, 8, poi 9→14 e usa il progetto in `code/spring-ai-demo`.
- Il codice Python gira con `python3` senza dipendenze (solo `attention.py` richiede NumPy).
