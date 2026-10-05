"""Chiama Ollama (o il tuo servizio Spring) via REST usando solo la libreria standard."""
import json, urllib.request

def post(url, payload, timeout=120):
    req = urllib.request.Request(url, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)

def chat(prompt, model="llama3.2", base="http://localhost:11434"):
    out = post(f"{base}/api/chat", {"model": model, "stream": False,
               "messages": [{"role": "user", "content": prompt}],
               "options": {"temperature": 0.2, "num_ctx": 4096}})
    return out["message"]["content"]

def embed(texts, model="nomic-embed-text", base="http://localhost:11434"):
    return post(f"{base}/api/embed", {"model": model, "input": texts})["embeddings"]

def stream_chat(prompt, model="llama3.2", base="http://localhost:11434"):
    """Streaming: Ollama manda una riga JSON per chunk (NDJSON)."""
    req = urllib.request.Request(f"{base}/api/chat", json.dumps({"model": model, "stream": True,
          "messages": [{"role": "user", "content": prompt}]}).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        for line in r:
            chunk = json.loads(line)
            yield chunk["message"]["content"]
            if chunk.get("done"): break

if __name__ == "__main__":
    print(chat("Spiega cos'è un token in una frase."))
