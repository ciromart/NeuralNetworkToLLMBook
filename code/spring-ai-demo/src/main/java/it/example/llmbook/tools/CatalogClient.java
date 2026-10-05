package it.example.llmbook.tools;

import org.springframework.stereotype.Component;
import org.springframework.web.client.RestClient;

/** Client REST verso un servizio esterno (il "catalogo prodotti"). */
@Component
public class CatalogClient {

    public record Product(String sku, String name, double price, int stock) {}

    private final RestClient rest;

    public CatalogClient(RestClient catalogRestClient) {
        this.rest = catalogRestClient;
    }

    public Product findBySku(String sku) {
        return rest.get()
                .uri("/products/{sku}", sku)
                .retrieve()
                .body(Product.class);
    }
}
