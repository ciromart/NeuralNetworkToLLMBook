package it.example.llmbook;

import it.example.llmbook.tools.CatalogClient;
import it.example.llmbook.tools.CatalogTools;
import org.junit.jupiter.api.Test;
import org.springframework.http.MediaType;
import org.springframework.test.web.client.MockRestServiceServer;
import org.springframework.web.client.RestClient;

import static org.assertj.core.api.Assertions.assertThat;
import static org.springframework.test.web.client.match.MockRestRequestMatchers.requestTo;
import static org.springframework.test.web.client.response.MockRestResponseCreators.withServerError;
import static org.springframework.test.web.client.response.MockRestResponseCreators.withSuccess;

/** Test del layer REST senza avviare né Ollama né il servizio reale. */
class CatalogClientTest {

    private RestClient.Builder builder() {
        return RestClient.builder().baseUrl("http://catalog");
    }

    @Test
    void tool_returns_formatted_product() {
        var builder = builder();
        var server = MockRestServiceServer.bindTo(builder).build();
        server.expect(requestTo("http://catalog/products/ABC-123"))
              .andRespond(withSuccess("""
                      {"sku":"ABC-123","name":"Tastiera","price":49.9,"stock":12}""",
                      MediaType.APPLICATION_JSON));

        var tools = new CatalogTools(new CatalogClient(builder.build()));

        assertThat(tools.productInfo("ABC-123")).contains("Tastiera", "49", "giacenza 12");
        server.verify();
    }

    @Test
    void tool_degrades_gracefully_on_server_error() {
        var builder = builder();
        var server = MockRestServiceServer.bindTo(builder).build();
        server.expect(requestTo("http://catalog/products/X")).andRespond(withServerError());

        var tools = new CatalogTools(new CatalogClient(builder.build()));

        assertThat(tools.productInfo("X")).contains("non disponibile");
    }
}
