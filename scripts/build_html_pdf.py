"""Converte DISPENSA_COMPLETA.md in HTML (immagini incorporate, diagrammi Mermaid renderizzati) e poi in PDF con Chromium.
Richiede: pip install markdown ; mermaid.min.js (npm pack mermaid@10) ; Chromium di Playwright.
Uso: python3 scripts/build_html_pdf.py /percorso/mermaid.min.js /percorso/chrome"""
import sys, re, base64, pathlib, subprocess, markdown, html

R = pathlib.Path(__file__).resolve().parent.parent
mermaid_js, chrome = pathlib.Path(sys.argv[1]), sys.argv[2]
tmp = pathlib.Path("/tmp/dispensa_build"); tmp.mkdir(exist_ok=True)

md = (R / "DISPENSA_COMPLETA.md").read_text(encoding="utf-8")
# blocchi mermaid -> placeholder (python-markdown non deve toccarli)
blocks = []
def keep(m):
    blocks.append(m.group(1)); return f"\n\n@@MERMAID{len(blocks)-1}@@\n\n"
md = re.sub(r"```mermaid\n(.*?)```", keep, md, flags=re.S)
body = markdown.markdown(md, extensions=["fenced_code", "tables", "toc", "sane_lists"])
for i, b in enumerate(blocks):
    body = body.replace(f"<p>@@MERMAID{i}@@</p>", f'<pre class="mermaid">{html.escape(b)}</pre>')
# immagini locali -> data URI
def img(m):
    p = R / m.group(1)
    if not p.exists(): return m.group(0)
    return 'src="data:image/png;base64,' + base64.b64encode(p.read_bytes()).decode() + '"'
body = re.sub(r'src="(book/figures/[^"]+)"', img, body)

css = """body{font:11pt/1.5 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;max-width:980px;margin:2rem auto;padding:0 1rem;color:#111}
h1{border-bottom:3px solid #2563eb;padding-bottom:.2em;page-break-before:always} h1:first-of-type{page-break-before:avoid}
h2{border-bottom:1px solid #ddd;margin-top:2em} h2,h3,h4{page-break-after:avoid}
pre{background:#f6f8fa;padding:.7em;border-radius:6px;overflow-wrap:anywhere;white-space:pre-wrap;font-size:8.6pt;page-break-inside:auto}
code{font-family:Menlo,Consolas,monospace;font-size:.92em;background:#f0f2f5;padding:0 .2em;border-radius:3px} pre code{background:none;padding:0}
table{border-collapse:collapse;margin:1em 0;font-size:9.5pt;width:100%} th,td{border:1px solid #ccc;padding:4px 7px;vertical-align:top} th{background:#eef2ff}
img{max-width:100%;height:auto;display:block;margin:1em auto} pre.mermaid{background:#fff;text-align:center;white-space:pre}
blockquote{border-left:4px solid #f59e0b;background:#fffbeb;margin:1em 0;padding:.4em 1em}"""
doc = f"""<!doctype html><html lang="it"><head><meta charset="utf-8"><title>Dispensa completa — Dalla rete neurale al LLM locale con Spring AI</title>
<style>{css}</style></head><body>{body}
<script>{mermaid_js.read_text(encoding="utf-8")}</script>
<script>mermaid.initialize({{startOnLoad:true,theme:'default',flowchart:{{useMaxWidth:true}},securityLevel:'loose'}});</script>
</body></html>"""
out_html = tmp / "dispensa.html"; out_html.write_text(doc, encoding="utf-8")
pdf = R / "DISPENSA_COMPLETA.pdf"
subprocess.run([chrome, "--headless=new", "--no-sandbox", "--disable-gpu", "--virtual-time-budget=60000",
                "--no-pdf-header-footer", f"--print-to-pdf={pdf}", out_html.as_uri()], check=True, timeout=300)
print("PDF:", pdf, pdf.stat().st_size // 1024, "KB")
