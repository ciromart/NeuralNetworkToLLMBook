package it.example.llmbook.config;

import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.boot.http.client.ClientHttpRequestFactoryBuilder;
import org.springframework.boot.http.client.HttpClientSettings;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.client.RestClient;

@Configuration
@EnableConfigurationProperties(CatalogProperties.class)
public class RestClientConfig {

    @Bean
    RestClient catalogRestClient(CatalogProperties props, RestClient.Builder builder) {
        var settings = HttpClientSettings.defaults()
                .withConnectTimeout(props.timeout())
                .withReadTimeout(props.timeout());
        return builder
                .baseUrl(props.baseUrl())
                .requestFactory(ClientHttpRequestFactoryBuilder.detect().build(settings))
                .build();
    }
}
