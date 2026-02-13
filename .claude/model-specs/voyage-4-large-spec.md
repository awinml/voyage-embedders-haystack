# voyage-4-large Model Specification

## Quick Reference

| Property | Value |
|----------|-------|
| Model ID | `voyage-4-large` |
| Model Type | Embedding |
| Default Dimensions | 1024 |
| Flexible Dimensions | 256, 512, 1024, 2048 |
| Max Input Tokens | 32,000 |
| Max Batch Tokens | 120,000 |
| Output Data Types | float, int8, uint8, binary, ubinary |

## Description

The best general-purpose and multilingual retrieval quality. Premium tier model offering highest accuracy for demanding retrieval tasks.

## Use Cases

- Highest quality retrieval
- Multilingual applications
- Complex semantic search
- RAG applications requiring best accuracy

## Special Features

- **Best retrieval quality** in voyage-4 series
- Flexible output dimensions (256, 512, 1024, 2048)
- Multiple output data types (float, int8, uint8, binary, ubinary)
- Compatible with all voyage-4 series embeddings

## API Test Results

| Method | Status | Version | Notes |
|--------|--------|---------|-------|
| curl | ✅ Pass | - | Successfully returned 1024-dimensional embedding |
| Python | ✅ Pass | 0.3.7 | Successfully returned 1024-dimensional embedding |
| TypeScript | ✅ Pass | 0.1.0 | Successfully returned 1024-dimensional embedding |

**Tested:** 2026-01-15

## Code Examples

### curl

```bash
curl -X POST "https://api.voyageai.com/v1/embeddings" \
    -H "Authorization: Bearer $VOYAGE_API_KEY" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "voyage-4-large",
        "input": "Your text to embed"
    }'
```

### Python

```python
import voyageai

client = voyageai.Client(api_key="your-api-key")

result = client.embed(
    texts=["Your text to embed"],
    model="voyage-4-large"
)

print(f"Embedding dimensions: {len(result.embeddings[0])}")
print(f"Total tokens: {result.total_tokens}")
```

### TypeScript

```typescript
import { VoyageAIClient } from "voyageai";

const client = new VoyageAIClient({
    apiKey: "your-api-key"
});

const result = await client.embed({
    model: "voyage-4-large",
    input: ["Your text to embed"]
});

console.log(`Embedding dimensions: ${result.data[0].embedding.length}`);
console.log(`Total tokens: ${result.usage.totalTokens}`);
```

## Documentation Links

- [Embeddings Documentation](https://docs.voyageai.com/docs/embeddings)
- [API Reference](https://docs.voyageai.com/reference/embeddings-api)
