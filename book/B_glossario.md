# Appendice B — Glossario

| Termine | Definizione breve |
|---|---|
| **Advisor** | In Spring AI, intercettore attorno alla chiamata al modello (memoria, RAG, log). |
| **Attention** | Meccanismo con cui ogni token pesa gli altri token del contesto. |
| **Backpropagation** | Calcolo dei gradienti applicando la chain rule al grafo di calcolo. |
| **BPE** | Byte Pair Encoding: algoritmo di tokenizzazione a sottoparole. |
| **Chunk** | Porzione di documento indicizzata nel RAG. |
| **Context window** | Numero massimo di token (input+output) gestibili in una richiesta. |
| **Embedding** | Vettore che rappresenta il significato di un token/testo. |
| **Fine-tuning** | Ulteriore addestramento di un modello su dati specifici. |
| **GGUF** | Formato di file per modelli quantizzati usato da llama.cpp/Ollama. |
| **Gradiente** | Derivata della loss rispetto a un parametro. |
| **KV-cache** | Cache di Key/Value dei token già elaborati, per velocizzare la generazione. |
| **LoRA / QLoRA** | Fine-tuning efficiente con matrici a basso rango (QLoRA su modello a 4 bit). |
| **Loss** | Numero che misura l'errore del modello da minimizzare. |
| **MCP** | Model Context Protocol: standard per collegare tool/risorse ai modelli. |
| **Open-weight** | Pesi scaricabili; non implica sempre codice/dati/licenza liberi. |
| **Perplexity** | `exp(loss)`; quanto il modello è "incerto". |
| **Prompt injection** | Istruzioni malevole nascoste in input/documenti per manipolare il modello. |
| **Quantizzazione** | Riduzione della precisione dei pesi (FP16→INT8/Q4) per risparmiare memoria. |
| **RAG** | Retrieval-Augmented Generation: recupero di contesto prima della risposta. |
| **RLHF / DPO** | Allineamento del modello con preferenze umane. |
| **SFT** | Supervised Fine-Tuning su coppie istruzione→risposta. |
| **Softmax** | Trasforma punteggi in probabilità. |
| **SSE** | Server-Sent Events: streaming HTTP unidirezionale server→client. |
| **Temperature** | Parametro che regola la casualità del campionamento. |
| **Token** | Unità minima di testo vista dal modello (sottoparola). |
| **Tool calling** | Il modello chiede all'app di eseguire una funzione e ne usa il risultato. |
| **Transformer** | Architettura basata su self-attention alla base degli LLM. |
| **Vector store** | Database per ricerca per similarità tra embedding. |
