package it.example.llmbook.rag;

import jakarta.validation.constraints.NotBlank;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;

@Validated
@RestController
@RequestMapping("/api/rag")
public class RagController {

    public record IngestRequest(@NotBlank String source, @NotBlank String text) {}
    public record Question(@NotBlank String question) {}

    private final RagService rag;

    public RagController(RagService rag) {
        this.rag = rag;
    }

    @PostMapping("/documents")
    public java.util.Map<String, Object> ingest(@RequestBody IngestRequest r) {
        return java.util.Map.of("chunks", rag.ingest(r.source(), r.text()));
    }

    @PostMapping("/ask")
    public java.util.Map<String, String> ask(@RequestBody Question q) {
        return java.util.Map.of("answer", rag.ask(q.question()));
    }
}
