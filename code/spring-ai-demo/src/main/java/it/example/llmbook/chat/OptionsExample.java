package it.example.llmbook.chat;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.ollama.api.OllamaChatOptions;

/** Override delle opzioni per singola chiamata (cap. 10). */
class OptionsExample {
    String extract(ChatClient chatClient, String text) {
        return chatClient.prompt()
                .user("Estrai i dati dal testo: " + text)
                .options(OllamaChatOptions.builder()
                        .temperature(0.0)
                        .numPredict(300))
                .call()
                .content();
    }
}
