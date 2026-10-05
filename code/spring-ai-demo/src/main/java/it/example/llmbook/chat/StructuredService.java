package it.example.llmbook.chat;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.stereotype.Service;

import java.util.List;

/** Structured output: il modello risponde in JSON, Spring lo mappa su un record. */
@Service
public class StructuredService {

    public record TicketAnalysis(String category, int priority, List<String> keywords, String summary) {}

    private final ChatClient chat;

    public StructuredService(ChatClient.Builder builder) {
        // builder "pulito": niente memoria per un task one-shot
        this.chat = builder.build();
    }

    public TicketAnalysis analyze(String ticketText) {
        return chat.prompt()
                .system("Classifica il ticket. priority da 1 (bassa) a 5 (critica).")
                .user(u -> u.text("Ticket: {t}").param("t", ticketText))
                .call()
                .entity(TicketAnalysis.class);
    }
}
