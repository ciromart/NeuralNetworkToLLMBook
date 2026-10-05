package it.example.llmbook.web;

import it.example.llmbook.tools.CatalogTools;
import jakarta.validation.constraints.NotBlank;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/assistant")
public class AssistantController {

    public record Question(@NotBlank String question) {}

    private final ChatClient chat;
    private final CatalogTools tools;

    public AssistantController(ChatClient.Builder builder, CatalogTools tools) {
        this.chat = builder.build();
        this.tools = tools;
    }

    @PostMapping
    public java.util.Map<String, String> ask(@RequestBody Question q) {
        String answer = chat.prompt().user(q.question()).tools(tools).call().content();
        return java.util.Map.of("answer", answer);
    }
}
