package it.example.llmbook.rag;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.document.Document;
import org.springframework.ai.rag.advisor.RetrievalAugmentationAdvisor;
import org.springframework.ai.rag.retrieval.search.VectorStoreDocumentRetriever;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Map;

@Service
public class RagService {

    private final VectorStore store;
    private final ChatClient ragClient;

    public RagService(VectorStore store, ChatClient.Builder builder) {
        this.store = store;
        var retriever = VectorStoreDocumentRetriever.builder()
                .vectorStore(store)
                .topK(4)
                .similarityThreshold(0.5)
                .build();
        this.ragClient = builder
                .defaultAdvisors(RetrievalAugmentationAdvisor.builder()
                        .documentRetriever(retriever)
                        .build())
                .build();
    }

    /** Ingestion: testo -> chunk -> embedding -> vector store. */
    public int ingest(String source, String text) {
        var chunks = TokenTextSplitter.builder()
                .withChunkSize(400).withMinChunkSizeChars(100)
                .withMinChunkLengthToEmbed(5).withMaxNumChunks(10_000)
                .withKeepSeparator(true).build()
                .apply(List.of(new Document(text, Map.of("source", source))));
        store.add(chunks);
        return chunks.size();
    }

    public String ask(String question) {
        return ragClient.prompt().user(question).call().content();
    }
}
