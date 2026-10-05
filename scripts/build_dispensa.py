"""Assembla DISPENSA_COMPLETA.md: schede video + capitoli del libro + CODICE REALE incluso dai file
(e relativi output eseguiti in code/python/outputs). Uso: python3 scripts/build_dispensa.py"""
import pathlib, re

R = pathlib.Path(__file__).resolve().parent.parent
read = lambda p: (R / p).read_text(encoding="utf-8")
out = []

def shift(md, by):
    """Abbassa i titoli di `by` livelli (fuori dai blocchi di codice) e sistema i percorsi relativi."""
    res, fence = [], False
    for line in md.splitlines():
        if line.lstrip().startswith("```"): fence = not fence
        if not fence and re.match(r"#{1,6} ", line):
            line = "#" * min(6, len(line) - len(line.lstrip("#")) + by) + line.lstrip("#")
        res.append(line)
    t = "\n".join(res)
    t = t.replace("](figures/", "](book/figures/").replace("](../code/", "](code/").replace("](../book/", "](book/")
    return t

def drop_h1(md):
    lines = md.splitlines()
    return "\n".join(lines[1:]) if lines and lines[0].startswith("# ") else md

def card(path):
    return shift(drop_h1(read(path)), 2)

def chapter(path, title):
    return f"\n### 📖 Approfondimento: {title}\n\n" + shift(drop_h1(read(path)), 3) + "\n"

LANG = {".py": "python", ".java": "java", ".yml": "yaml", ".xml": "xml", ".sh": "bash"}
def code(path, output=None, title=None):
    p = pathlib.Path(path); lang = LANG.get(p.suffix, "")
    s = f"\n#### 💻 Codice: `{path}`" + (f" — {title}" if title else "") + f"\n\n```{lang}\n{read(path).rstrip()}\n```\n"
    if output:
        s += f"\n**Output reale** (eseguito durante la scrittura, `code/python/outputs/{output}`):\n\n```text\n{read('code/python/outputs/' + output).rstrip()}\n```\n"
    return s

def add(*parts): out.extend(parts)

add("""# 📘 Dispensa completa — Dalla rete neurale al LLM locale con Spring AI

> Documento unico del progetto. Unisce: **(1)** le schede dei 10 video di *Neural Networks: Zero to Hero* (A. Karpathy), **(2)** i capitoli teorici con grafici e diagrammi, **(3)** **codice vero**, incluso automaticamente dai file del repository insieme al suo **output reale**, **(4)** la parte Java / Spring AI / REST.
> Generato da `scripts/build_dispensa.py`: i listati sono sempre uguali ai file eseguiti.

## Come leggere questo documento
- Ogni video ha: **scheda con capitoli ufficiali** → **approfondimento** → **codice eseguito**.
- 💻 = codice completo e funzionante · 📖 = teoria · ⚠️ = limite o avvertenza.
- I diagrammi `mermaid` si vedono su GitHub; le figure sono in `book/figures/`.

## ⚠️ Cosa è verificato e cosa no
| Elemento | Stato |
|---|---|
| Script Python (`code/python/`) | ✅ **eseguiti**; gli output mostrati sono quelli reali (PyTorch 2.x, CPU) |
| Progetto Spring Boot 4.1.1 / Spring AI 2.0.1 / Java 21 | ✅ compila, 2 test passano (`code/python/outputs/spring_tests.txt`) |
| Dialogo con un modello **Ollama reale** (chat, RAG, tool calling) | ⚠️ **non eseguito** (nessun modello nel sandbox); il codice compila |
| Trascrizione parlata dei video | ❌ non disponibile: le schede usano i **capitoli ufficiali** dei video + spiegazioni mie |
| Video 6 e 8 | ⚠️ senza capitoli su YouTube: struttura ricostruita |

## Indice
""")
toc = [
 "Parte I — Panoramica e le tre strade per un LLM locale",
 "Parte II — Dai neuroni a GPT (i 10 video, con codice)",
 "Parte III — LLM locale: Ollama, quantizzazione, LoRA",
 "Parte IV — Spring AI: installazione, fondamenti, chat, RAG, tool calling REST (codice Java completo)",
 "Parte V — Python: digressioni",
 "Appendice — Glossario",
]
add("\n".join(f"{i}. {t}" for i, t in enumerate(toc, 1)), "\n\n---\n")

# ---------- PARTE I
add("\n# Parte I — Panoramica e le tre strade per un LLM locale\n")
add(shift(drop_h1(read("book/00_prefazione.md")), 1))
add("\n### Come è stata costruita la parte sui video\n", shift(drop_h1(read("dispensa/00_come_usare.md")), 3))

# ---------- PARTE II
add("\n\n---\n\n# Parte II — Dai neuroni a GPT (i 10 video, con codice)\n")
V = [
 ("dispensa/01_micrograd.md", [("book/01_reti_neurali_backprop.md", "Reti neurali e backpropagation")],
  [("code/python/micrograd_mini.py", "micrograd_mini.txt", "autograd + MLP, il cuore del video 1"),
   ("code/python/v01_gradcheck.py", "v01_gradcheck.txt", "verifica dei gradienti: micrograd vs PyTorch vs derivata numerica")]),
 ("dispensa/02_makemore_bigramma.md", [("book/02_language_modeling_bigram.md", "Language modeling e bigramma")],
  [("code/python/bigram.py", "bigram.txt", "bigramma a conteggi e NLL"),
   ("code/python/sampling.py", "sampling.txt", "temperatura, top-k, top-p da zero")]),
 ("dispensa/03_makemore_mlp.md", [("book/03_mlp_embedding.md", "MLP ed embedding")],
  [("code/python/v03_mlp_nomi.py", "v03_mlp_nomi.txt", "MLP + embedding su nomi italiani, con split train/dev/test, early stopping e overfitting REALE")]),
 ("dispensa/04_makemore_batchnorm.md", [("book/04_training_sano.md", "Far funzionare il training: init, BatchNorm, backprop manuale, WaveNet")],
  [("code/python/v04_init_batchnorm.py", "v04_init_batchnorm.txt", "perché l'inizializzazione conta: saturazione di tanh e BatchNorm")]),
 ("dispensa/05_makemore_backprop_ninja.md", [], []),
 ("dispensa/06_makemore_wavenet.md", [], []),
 ("dispensa/07_build_gpt.md", [("book/05_transformer_gpt.md", "Il Transformer")],
  [("code/python/attention.py", "attention.txt", "self-attention causale in NumPy"),
   ("code/python/v07_mini_gpt.py", "v07_mini_gpt.txt", "mini-GPT completo in PyTorch, allenato su CPU")]),
 ("dispensa/08_state_of_gpt.md", [("book/07_dal_pretraining_all_assistente.md", "Dal pretraining all'assistente (video 8 e 10)")], []),
 ("dispensa/09_tokenizer.md", [("book/06_tokenizer.md", "Il tokenizer")], [("code/python/bpe.py", "bpe.txt", "Byte Pair Encoding minimale")]),
 ("dispensa/10_gpt2_124m.md", [], []),
]
for i, (cardp, chs, codes) in enumerate(V, 1):
    first = read(cardp).splitlines()[0].lstrip("# ").strip()
    add(f"\n\n## {first}\n", "\n".join(card(cardp).splitlines()[1:]) if False else shift(drop_h1(read(cardp)), 2))
    for p, t in chs: add(chapter(p, t))
    for p, o, t in codes: add(code(p, o, t))
    add("\n---\n")

# ---------- PARTE III
add("\n# Parte III — LLM locale: Ollama, quantizzazione, LoRA\n", shift(drop_h1(read("book/08_llm_locale.md")), 1))
add(code("code/python/v08_lora_da_zero.py", "v08_lora_da_zero.txt", "LoRA da zero: congela W, allena solo A e B (≈3% dei parametri)"))
add(code("code/python/ollama_client.py", None, "client REST per Ollama con la sola libreria standard"))
add("\n---\n")

# ---------- PARTE IV
add("\n# Parte IV — Spring AI: installazione, fondamenti, chat, RAG, tool calling REST\n")
for p in ["book/09_installare_spring_ai.md", "book/10_fondamenti_gestire_llm.md", "book/11_spring_ai_chat.md", "book/12_rag_spring_ai.md", "book/13_tool_calling_rest.md", "book/14_best_practice.md"]:
    add("\n", shift(read(p), 1), "\n")
add("\n## 💻 Codice completo del progetto `code/spring-ai-demo`\n\nTutti i file sono quelli che compilano; i test `CatalogClientTest` passano (2/2).\n")
add("**Test eseguiti:**\n\n```text\n" + read("code/python/outputs/spring_tests.txt").strip() + "\n```\n")
base = "code/spring-ai-demo/"
for f in ["pom.xml", "src/main/resources/application.yml", "docker-compose.yml",
          "src/main/java/it/example/llmbook/LlmBookApplication.java",
          "src/main/java/it/example/llmbook/config/AiConfig.java",
          "src/main/java/it/example/llmbook/config/CatalogProperties.java",
          "src/main/java/it/example/llmbook/config/RestClientConfig.java",
          "src/main/java/it/example/llmbook/chat/ChatController.java",
          "src/main/java/it/example/llmbook/chat/StructuredService.java",
          "src/main/java/it/example/llmbook/chat/OptionsExample.java",
          "src/main/java/it/example/llmbook/rag/RagService.java",
          "src/main/java/it/example/llmbook/rag/RagController.java",
          "src/main/java/it/example/llmbook/tools/CatalogClient.java",
          "src/main/java/it/example/llmbook/tools/CatalogTools.java",
          "src/main/java/it/example/llmbook/web/AssistantController.java",
          "src/main/java/it/example/llmbook/web/ApiExceptionHandler.java",
          "src/test/java/it/example/llmbook/CatalogClientTest.java"]:
    add(code(base + f))
add("\n---\n")

# ---------- PARTE V
add("\n# Parte V — Python: digressioni\n", shift(drop_h1(read("book/A_digressioni_python.md")), 1))
add(code("code/python/mini_rag.py", "mini_rag.txt", "mini-RAG con embedding lessicale: mostra anche perché fallisce"))

# ---------- GLOSSARIO
add("\n\n---\n\n# Appendice — Glossario\n", shift(drop_h1(read("book/B_glossario.md")), 1))
add("\n\n---\n*Fine della dispensa.*\n")

(R / "DISPENSA_COMPLETA.md").write_text("\n".join(out), encoding="utf-8")
print("scritto DISPENSA_COMPLETA.md:", len("\n".join(out).splitlines()), "righe")
