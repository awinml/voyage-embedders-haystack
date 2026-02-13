# Model Spec: voyage-multimodal-3.5

**Generated:** 2025-12-17
**Status:** Preview
**API Tested:** ✅ All tests passed

---

## Quick Reference

| Property | Value |
|----------|-------|
| Model ID | `voyage-multimodal-3.5` |
| Type | Multimodal Embedding |
| Modalities | Text, Image, Video |
| Dimensions | 1024 (default) |
| Dimension Options | 256, 512, 1024, 2048 |
| Max Tokens | 32,000 |
| API Endpoint | `/v1/multimodalembeddings` |

---

## API Test Results

| SDK | Status | Version | Notes |
|-----|--------|---------|-------|
| curl | ✅ Pass | - | Text-only and text+image both work |
| Python | ✅ Pass | 0.3.5+ | Uses `multimodal_embed()` with list format. **Video requires v0.3.6+** |
| TypeScript | ✅ Pass | 0.1.0 | Uses `multimodalEmbed()` with content array |

---

## Use Cases

- Mixed-media document retrieval
- PDF and slide processing
- Image-text similarity search
- Video content retrieval
- Cross-modal semantic search

---

## Supported Input Types

| Type | Description |
|------|-------------|
| `text` | Plain text content |
| `image_url` | URL to an image file |
| `image_base64` | Base64-encoded image data |
| `video_url` | URL to a video file |
| `video_base64` | Base64-encoded video data |

---

## Special Features

- Supports interleaved text, images, and video in single inputs
- Variable output dimensions (256, 512, 1024, 2048)
- No text extraction workflow required for documents
- Image: max 20MB, 16M pixels
- Video: max 20MB
- Token counting: 560 image pixels = 1 token, 1120 video pixels = 1 token

---

## Code Examples

### cURL

```bash
curl -X POST "https://api.voyageai.com/v1/multimodalembeddings" \
  -H "Authorization: Bearer $VOYAGE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {
        "content": [
          {"type": "text", "text": "Your text here"},
          {"type": "image_url", "image_url": "https://example.com/image.jpg"}
        ]
      }
    ],
    "model": "voyage-multimodal-3.5"
  }'
```

### Python

```python
import voyageai
from PIL import Image

client = voyageai.Client(api_key="your-api-key")

# Text only
result = client.multimodal_embed(
    inputs=[["Your text here"]],
    model="voyage-multimodal-3.5"
)

# Text + Image
image = Image.open("image.jpg")
result = client.multimodal_embed(
    inputs=[["Your text here", image]],
    model="voyage-multimodal-3.5"
)

print(f"Dimensions: {len(result.embeddings[0])}")
print(f"Tokens: {result.total_tokens}")
```

### TypeScript

```typescript
import { VoyageAIClient } from "voyageai";

const client = new VoyageAIClient({ apiKey: "your-api-key" });

const result = await client.multimodalEmbed({
    model: "voyage-multimodal-3.5",
    inputs: [
        {
            content: [
                { type: "text", text: "Your text here" },
                { type: "image_url", image_url: "https://example.com/image.jpg" }
            ]
        }
    ]
});

console.log(`Dimensions: ${result.data[0].embedding.length}`);
console.log(`Usage: ${JSON.stringify(result.usage)}`);
```

---

## Documentation Links

- [Multimodal Embeddings](https://docs.voyageai.com/docs/multimodal-embeddings)
- [Embeddings Overview](https://docs.voyageai.com/docs/embeddings)
- [API Reference](https://docs.voyageai.com/reference/multimodalembeddings-api)

---

## Important Notes

⚠️ **This is a MULTIMODAL model** - use the `/v1/multimodalembeddings` endpoint, NOT `/v1/embeddings`

- Python SDK: Use `client.multimodal_embed()` with a list of strings/PIL images
  - **Video input requires voyageai >= 0.3.6**
- TypeScript SDK: Use `client.multimodalEmbed()` with content array format
- curl/REST: Use the content array with `type` field for each item
