package it.example.llmbook.chat;

import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;
import reactor.core.publisher.Flux;

import static org.springframework.ai.chat.memory.ChatMemory.CONVERSATION_ID;

@RestController
@RequestMapping("/api/chat")
public class ChatController {

    public record ChatRequest(@NotBlank @Size(max = 4000) String message) {}
    public record ChatResponse(String conversationId, String answer) {}

    private final ChatClient chat;

    public ChatController(ChatClient chat) {
        this.chat = chat;
    }

    /** Risposta completa (bloccante). */
    @PostMapping("/{conversationId}")
    public ChatResponse ask(@PathVariable String conversationId, @Valid @RequestBody ChatRequest req) {
        String answer = chat.prompt()
                .user(req.message())
                .advisors(a -> a.param(CONVERSATION_ID, conversationId))
                .call()
                .content();
        return new ChatResponse(conversationId, answer);
    }

    /** Streaming token-per-token via Server-Sent Events. */
    @PostMapping(value = "/{conversationId}/stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public Flux<String> stream(@PathVariable String conversationId, @Valid @RequestBody ChatRequest req) {
        return chat.prompt()
                .user(req.message())
                .advisors(a -> a.param(CONVERSATION_ID, conversationId))
                .stream()
                .content();
    }
}
