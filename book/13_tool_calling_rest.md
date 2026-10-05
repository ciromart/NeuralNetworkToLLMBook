# Capitolo 13 — Tool calling e integrazione con servizi REST

Due direzioni di integrazione REST:
1. **Il tuo servizio espone REST** verso i client (capitoli 11–12).
2. **Il LLM "usa" servizi REST** come strumenti (*tool calling*): questo capitolo.

## 13.1 Come funziona il tool calling
Il modello **non esegue nulla**. Riceve la descrizione dei tool; se serve, risponde *"chiama `productInfo` con sku=ABC-123"*; **è la tua applicazione a eseguire** la chiamata e a restituire il risultato.

![Sequenza tool calling](figures/18_sequenza_tool_calling.png)

> Serve un modello con supporto ai tool (con Ollama: `llama3.1+`, `llama3.2`, `qwen2.5`, `mistral`...).

## 13.2 Il client REST (`RestClient`) con timeout
```java
@ConfigurationProperties(prefix = "app.catalog")
public record CatalogProperties(String baseUrl, Duration timeout) {}

@Bean
RestClient catalogRestClient(CatalogProperties props, RestClient.Builder builder) {
    var settings = HttpClientSettings.defaults()
            .withConnectTimeout(props.timeout())
            .withReadTimeout(props.timeout());
    return builder.baseUrl(props.baseUrl())
            .requestFactory(ClientHttpRequestFactoryBuilder.detect().build(settings))
            .build();
}

@Component
public class CatalogClient {
    public record Product(String sku, String name, double price, int stock) {}
    private final RestClient rest;
    public CatalogClient(RestClient catalogRestClient) { this.rest = catalogRestClient; }

    public Product findBySku(String sku) {
        return rest.get().uri("/products/{sku}", sku).retrieve().body(Product.class);
    }
}
```
`RestClient` è il client sincrono moderno di Spring (sostituisce `RestTemplate`); per chiamate reattive usa `WebClient`; per client dichiarativi `@HttpExchange`.

## 13.3 Esporlo come tool
```java
@Component
public class CatalogTools {
    @Tool(description = "Restituisce nome, prezzo in euro e giacenza di un prodotto dato il suo SKU")
    public String productInfo(@ToolParam(description = "Codice SKU, es. ABC-123") String sku) {
        try {
            var p = catalog.findBySku(sku);
            return "%s (%s): %.2f EUR, giacenza %d".formatted(p.name(), p.sku(), p.price(), p.stock());
        } catch (RestClientException e) {
            return "Servizio catalogo non disponibile o SKU inesistente: " + sku;
        }
    }
}

// uso
String answer = chat.prompt().user("Quanti pezzi di ABC-123 ho?")
        .tools(catalogTools).call().content();
```
La **descrizione** del tool è il "prompt" che guida il modello: scrivila con cura. L'errore viene restituito come *testo*, così il modello può spiegarlo all'utente invece di far crollare la richiesta.

## 13.4 Regole d'oro per i tool
| Regola | Motivo |
|---|---|
| **Privilegio minimo**: tool *read-only* di default | il modello può sbagliare o essere manipolato (prompt injection) |
| Azioni con effetti (ordini, pagamenti) → **conferma umana** o endpoint con policy | irreversibilità |
| Validare gli argomenti come input non fidato | il modello genera i parametri |
| Timeout e limite di iterazioni | evitare loop di chiamate |
| Risultati brevi e strutturati | consumano contesto |
| Autorizzazione basata sull'**utente**, non sul modello | niente escalation |
| Loggare ogni chiamata (senza PII) | audit |

## 13.5 Testare senza Ollama né servizi reali
```java
var builder = RestClient.builder().baseUrl("http://catalog");
var server = MockRestServiceServer.bindTo(builder).build();
server.expect(requestTo("http://catalog/products/ABC-123"))
      .andRespond(withSuccess("""
          {"sku":"ABC-123","name":"Tastiera","price":49.9,"stock":12}""",
          MediaType.APPLICATION_JSON));

var tools = new CatalogTools(new CatalogClient(builder.build()));
assertThat(tools.productInfo("ABC-123")).contains("Tastiera", "giacenza 12");
```
Questi test (in `CatalogClientTest`) **sono eseguiti e passano** (2/2). Per i test d'integrazione col modello usa Testcontainers + Ollama e asserzioni "per proprietà".

## 13.6 Altre integrazioni REST
- **MCP (Model Context Protocol)**: standard per esporre tool/risorse a un LLM; Spring AI ha client e server MCP.
- **API OpenAPI** → genera client tipizzato e wrappalo con `@Tool`.
- **Provider compatibili OpenAI**: Ollama espone anche `/v1/chat/completions`, quindi puoi puntare lo starter OpenAI a `http://localhost:11434/v1`.
