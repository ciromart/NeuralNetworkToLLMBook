package it.example.llmbook.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

import java.time.Duration;

@ConfigurationProperties(prefix = "app.catalog")
public record CatalogProperties(String baseUrl, Duration timeout) {}
