# voyage-embedders-haystack - VoyageAI Model Configuration

## Project Overview
This is a Haystack integration for VoyageAI embedding and reranking models. It provides `VoyageTextEmbedder`, `VoyageDocumentEmbedder` for creating embeddings, and `VoyageRanker` for document reranking using VoyageAI's API.

**Type:** Python library (Haystack integration)
**Language:** Python
**Last analyzed:** 2025-12-21
**Supports:** Embeddings AND Rerankers

---

## How Models Are Handled

### Model Registry Location
**There is NO model registry in this codebase.** Models are passed as a `model` parameter string directly to the VoyageAI API client. The codebase does not validate or enumerate supported models - it relies on the VoyageAI API to handle model validation.

### Model Selection
Users specify the model name as a string when initializing components:
- `VoyageTextEmbedder(model="voyage-3")`
- `VoyageDocumentEmbedder(model="voyage-3-large")`
- `VoyageRanker(model="rerank-2")`

The model string is passed directly to the VoyageAI `Client.embed()` or `Client.rerank()` methods.

### Dimension Handling (Embeddings)
Dimensions are handled dynamically by the VoyageAI API. Some models support custom dimensions via the `output_dimension` parameter:
- `voyage-3-large` and `voyage-code-3` support: 2048, 1024 (default), 512, 256
- Other models use their fixed default dimension (typically 1024)

The codebase does NOT maintain a dimension mapping - dimensions are determined by the model and optional `output_dimension` parameter.

### Reranker Integration
Rerankers are handled by `VoyageRanker` in `src/haystack_integrations/components/rankers/voyage/ranker.py`:
- No dimension handling needed
- Model passed directly to `Client.rerank()` method
- Supports `top_k`, `truncate` parameters

---

## Adding a New Embedding Model - Locations to Update

### 1. Source Code (NO CHANGES REQUIRED)
The codebase is model-agnostic. New VoyageAI models work automatically as the model string is passed directly to the API.

### 2. Documentation (RECOMMENDED)

| File | What to Add |
|------|-------------|
| `README.md` | Update examples if demonstrating specific model |
| `src/.../voyage_text_embedder.py` | Update docstring examples if needed |
| `src/.../voyage_document_embedder.py` | Update docstring examples if needed |

### 3. Tests (RECOMMENDED)

| File | What to Add |
|------|-------------|
| `tests/test_voyage_text_embedder.py` | Add integration test with new model |
| `tests/test_voyage_document_embedder.py` | Add integration test with new model |

### 4. Examples (OPTIONAL)

| File | What to Add |
|------|-------------|
| `examples/text_embedder_example.py` | Update model if showcasing new model |
| `examples/document_embedder_example.py` | Update model if showcasing new model |

---

## Adding a New Reranker Model - Locations to Update

### 1. Source Code (NO CHANGES REQUIRED)
Same as embeddings - reranker models work automatically.

### 2. Documentation (RECOMMENDED)

| File | What to Add |
|------|-------------|
| `README.md` | Update reranker examples if needed |
| `src/.../ranker.py` | Update docstring examples if needed |

### 3. Tests (RECOMMENDED)

| File | What to Add |
|------|-------------|
| `tests/test_ranker.py` | Add integration test with new reranker model |

### 4. Examples (OPTIONAL)

| File | What to Add |
|------|-------------|
| `examples/reranker_example.py` | Update model if showcasing new model |

---

## Current Embedding Models (Used in Codebase)

| Model ID | Default Dimensions | Custom Dimensions Support |
|----------|-------------------|---------------------------|
| voyage-3 | 1024 | No |
| voyage-3-large | 1024 | Yes (2048, 1024, 512, 256) |
| voyage-code-3 | 1024 | Yes (2048, 1024, 512, 256) |
| voyage-4 | 1024 | Yes (2048, 1024, 512, 256) |
| voyage-4-large | 1024 | Yes (2048, 1024, 512, 256) |
| voyage-4-lite | 1024 | Yes (2048, 1024, 512, 256) |

*Note: These are models referenced in the codebase. VoyageAI may support additional models.*

## Current Reranker Models (Used in Codebase)

| Model ID | Status |
|----------|--------|
| rerank-2 | Active |

*Note: VoyageAI may support additional reranker models (rerank-2-lite, etc.).*

---

## Commands

### Run All Tests
```bash
hatch run test
```

### Run Unit Tests Only
```bash
pytest tests/ -m unit
```

### Run Integration Tests Only
```bash
VOYAGE_API_KEY=<your-key> pytest tests/ -m integration
```

### Run Tests with Coverage
```bash
hatch run test-cov
```

### Lint & Format
```bash
hatch run lint:fmt
```

### Type Check
```bash
hatch run lint:typing
```

### Run Examples
```bash
hatch run example-text-embedder
hatch run example-doc-embedder
hatch run example-semantic-search
hatch run example-reranker
```

---

## Special Considerations

- **No model registry**: New models work automatically without code changes
- **API key required**: Set `VOYAGE_API_KEY` environment variable
- **Integration tests**: Require valid API key and make real API calls
- **Dimensions**: `voyage-3-large`, `voyage-code-3`, `voyage-4`, `voyage-4-large`, and `voyage-4-lite` support custom dimensions via `output_dimension`
- **Output types**: `voyage-3-large`, `voyage-code-3`, `voyage-4`, `voyage-4-large`, and `voyage-4-lite` support quantized output (`int8`, `uint8`, `binary`, `ubinary`)

---

## Project Structure

```
src/haystack_integrations/
├── components/
│   ├── embedders/voyage_embedders/
│   │   ├── __init__.py                    # Exports VoyageDocumentEmbedder, VoyageTextEmbedder
│   │   ├── voyage_document_embedder.py    # Document embedding component
│   │   └── voyage_text_embedder.py        # Text embedding component
│   └── rankers/voyage/
│       ├── __init__.py                    # Exports VoyageRanker
│       └── ranker.py                      # Reranking component
tests/
├── test_voyage_document_embedder.py       # Document embedder tests
├── test_voyage_text_embedder.py           # Text embedder tests
└── test_ranker.py                         # Reranker tests
examples/
├── text_embedder_example.py               # Text embedding example
├── document_embedder_example.py           # Document embedding example
├── semantic_search_pipeline_example.py    # Full pipeline example
└── reranker_example.py                    # Reranking example
```

---

## New Embedding Model Checklist

Since this is a pass-through integration, adding a new model is minimal:

- [ ] Verify model works with existing code (just pass model name)
- [ ] Update documentation if showcasing new model features
- [ ] Add integration test for new model (optional but recommended)
- [ ] Update examples if relevant
- [ ] No code changes required for basic support

## New Reranker Model Checklist

- [ ] Verify reranker works with existing code (just pass model name)
- [ ] Update documentation if needed
- [ ] Add integration test for new reranker (optional but recommended)
- [ ] Update examples if relevant
- [ ] No code changes required for basic support

---

## Key Files Reference

| Purpose | File Path |
|---------|-----------|
| Text Embedder | `src/haystack_integrations/components/embedders/voyage_embedders/voyage_text_embedder.py` |
| Document Embedder | `src/haystack_integrations/components/embedders/voyage_embedders/voyage_document_embedder.py` |
| Ranker | `src/haystack_integrations/components/rankers/voyage/ranker.py` |
| README | `README.md` |
| PyProject | `pyproject.toml` |
