# Video 9 — Let's build the GPT Tokenizer (2:13:35)
🔗 <https://www.youtube.com/watch?v=zduSFxRajkE> · Codice: [`bpe.py`](../code/python/bpe.py) · Libro: [cap. 6](../book/06_tokenizer.md)

**Obiettivo:** capire e implementare il tokenizer (BPE), e perché molte stranezze degli LLM nascono lì.

## Mappa dei capitoli (timestamp ufficiali)
| Tempo | Capitolo | Cosa si impara |
|---|---|---|
| 0:00 | intro: Tokenization, GPT-2 paper, issues | problemi tipici legati ai token |
| 5:50 | tokenization by example in a Web UI (tiktokenizer) | spazi, maiuscole, numeri e lingue diverse cambiano i token |
| 14:56 | strings in Python, Unicode code points | `ord()`/`chr()` |
| 18:15 | Unicode byte encodings: ASCII, UTF-8, UTF-16, UTF-32 | UTF-8 è la base dei tokenizer moderni |
| 22:47 | daydreaming: deleting tokenization | l'ideale sarebbe lavorare sui byte grezzi |
| 23:50 | Byte Pair Encoding algorithm walkthrough | fusione iterativa della coppia più frequente |
| 27:02 | starting the implementation | |
| 28:35 | counting consecutive pairs, most common pair | |
| 30:36 | merging the most common pair | |
| 34:58 | training the tokenizer: while loop, compression ratio | vocabolario = 256 + n_merge |
| 39:20 | tokenizer/LLM diagram: a completely separate stage | allenato a parte, con dati propri |
| 42:47 | decoding tokens to strings | attenzione ai byte UTF-8 non validi |
| 48:21 | encoding strings to tokens | applicare i merge nell'ordine di training |
| 57:36 | regex patterns to force splits across categories | impedisce di fondere lettere, numeri, punteggiatura |
| 1:11:38 | tiktoken library, GPT-2 vs GPT-4 regex | |
| 1:14:59 | GPT-2 encoder.py walkthrough | |
| 1:18:26 | special tokens | `<|endoftext|>` e simili, gestiti a parte |
| 1:25:28 | minbpe exercise | scrivi il tuo tokenizer GPT-4 |
| 1:28:42 | sentencepiece library, used for Llama 2 | BPE sui code point, fallback ai byte |
| 1:43:27 | how to set vocabulary size | compromesso fra lunghezza sequenza e embedding table |
| 1:48:11 | training new tokens, prompt compression | aggiungere token per comprimere i prompt |
| 1:49:58 | multimodal tokenization (vector quantization) | immagini/audio come token |
| 1:51:41 | revisiting the quirks of LLM tokenization | spiega errori su ortografia, aritmetica, lingue non inglesi, token "SolidGoldMagikarp" |
| 2:10:20 | final recommendations | |
| 2:12:50 | ??? :) | |

## Concetti chiave
- **BPE**: da 256 byte a un vocabolario di sottoparole massimizzando la compressione.
- Il tokenizer è **separato dal modello**: cambiarlo = riaddestrare.
- Lingue diverse dall'inglese, codice, numeri e spazi sono tokenizzati con efficienza diversa → **costi e contesto** diversi.
- Molti difetti attribuiti al «modello» sono difetti di **tokenizzazione**.

## Esercizi
1. Estendi [`bpe.py`](../code/python/bpe.py) con `decode()` e verifica `decode(encode(s)) == s` anche con emoji.
2. Confronta i token per la stessa frase in italiano e inglese (libreria `tiktoken` o `transformers`).
3. Allena un BPE su un testo italiano e osserva i primi 20 merge.

## Quiz
1. *Perché partire dai byte?* — nessun carattere "sconosciuto", qualsiasi stringa è codificabile.
2. *Perché la regex di pre-split?* — evita fusioni indesiderabili (`dog.` + `dog!`) e migliora la qualità del vocabolario.
3. *Perché l'italiano costa più token?* — il vocabolario è tipicamente addestrato su più inglese.
