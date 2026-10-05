package it.example.llmbook.tools;

import org.springframework.ai.tool.annotation.Tool;
import org.springframework.ai.tool.annotation.ToolParam;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestClientException;

/** Tool calling: il LLM decide *quando* chiamare questo metodo; lo eseguiamo noi. */
@Component
public class CatalogTools {

    private final CatalogClient catalog;

    public CatalogTools(CatalogClient catalog) {
        this.catalog = catalog;
    }

    @Tool(description = "Restituisce nome, prezzo in euro e giacenza di un prodotto dato il suo SKU")
    public String productInfo(@ToolParam(description = "Codice SKU, es. ABC-123") String sku) {
        try {
            var p = catalog.findBySku(sku);
            return "%s (%s): %.2f EUR, giacenza %d".formatted(p.name(), p.sku(), p.price(), p.stock());
        } catch (RestClientException e) {
            // L'errore torna al modello come testo: potrà spiegarlo all'utente.
            return "Servizio catalogo non disponibile o SKU inesistente: " + sku;
        }
    }
}
