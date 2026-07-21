# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-07-21

### Added

- **Haystack 3.x compatibility**: Components now work with `haystack-ai>=3.0.0`.
- **`warm_up()` method**: All components now support lazy initialization via `warm_up()`. The `client` and `async_client` properties auto-initialize on first access.
- **Async support**: All components now have `async def run()` methods for use with Haystack 3.x async pipelines.
- **Async client**: Components use `voyageai.AsyncClient` for async operations with `async_embed()`, `async_rerank()`, `async_multimodal_embed()`, and `async_contextualized_embed()`.

### Changed

- **Breaking**: `VoyageDocumentEmbedder.run()` and `VoyageContextualizedDocumentEmbedder.run()` now return new Document instances via `dataclasses.replace()` instead of mutating documents in-place (Haystack 3.x Document immutability).
- **Breaking**: The `client` is no longer created at `__init__`. It is now lazily initialized on first access via `warm_up()`. This means `__init__` no longer fails if the API key is missing — `warm_up()` or the `client`/`async_client` property will raise the error.
- **Breaking**: All `run()` methods are now `async def run()` for compatibility with Haystack 3.x.
- **Breaking**: Minimum Haystack version is now `haystack-ai>=3.0.0`.

### Migration Notes

- If you were calling `VoyageTextEmbedder.run()` directly, you now need to `await` it:
  ```python
  result = await embedder.run(text="Hello")
  ```
- In Haystack pipelines, the `Pipeline.run_async()` method handles async components automatically.
- If you were checking `embedder.client` at init, note that the client is now lazily created. Call `embedder.warm_up()` if you need it initialized eagerly.

## [1.10.0] - 2026-02-15

### Added

- Support for Voyage 4 model family (`voyage-4`, `voyage-4-large`, `voyage-4-lite`).
- Voyage 4 models support flexible output dimensions (256, 512, 1024, 2048) and multiple output data types (`float`, `int8`, `uint8`, `binary`, `ubinary`).

### Changed

- Updated `VoyageTextEmbedder` and `VoyageDocumentEmbedder` docstrings to document Voyage 4 model support for `output_dimension` and `output_dtype` parameters.
- Updated examples to use `voyage-4` as the default model.

## [1.9.1] - 2026-02-07

### Fixed

- Serialize `chunk_fn` for `VoyageContextualizedDocumentEmbedder` using Haystack's `serialize_callable`/`deserialize_callable`.
- Tighten `chunk_fn` type to `Callable[[str], list[str]]`.
- Add explicit `run()` return types across all components.
- Fix multimodal `_convert_content_item` parameter type.

### Chore

- Add dotenv support to examples and `python-dotenv` to dev dependencies.
- Clean up `pyproject.toml` (remove dead mypy config, redundant ty config).
- Update `CONTRIBUTING.md` with API key setup instructions.

## [1.9.0] - 2026-01-24

### Added

- The new `VoyageMultimodalEmbedder` component supports Voyage's multimodal embedding model (`voyage-multimodal-3.5`).
- Multimodal embeddings can encode text, images, and videos into a shared vector space for cross-modal similarity search.

## [1.8.0] - 2025-11-07

### Added

- The new `VoyageContextualizedDocumentEmbedder` component supports Voyage's contextualized chunk embeddings.
- Contextualized embeddings encode document chunks "in context" with other chunks from the same document, preserving semantic relationships and reducing context loss for improved retrieval accuracy.

## [1.5.0] - 2025-01-22

### Added

- The new `VoyageRanker` component can be used to rerank documents using the `Voyage Reranker` models.
- Matryoshka Embeddings and Quantized Embeddings can now be created using the `output_dimension` and `output_dtype` parameters.

## [1.4.0] - 2024-07-24

### Added

- The maximum timeout and number of retries made by the Client can now be set for the embedders using the `timeout` and `max_retries` parameters.

## [1.3.0] - 2024-03-18

### Changed

- **Breaking Change:** The import path for the embedders has been changed to `haystack_integrations.components.embedders.voyage_embedders`.
- The embedders now use the Haystack `Secret` API for authentication.

## [1.2.0] - 2024-02-02

### Changed

- **Breaking Change:** `VoyageDocumentEmbedder` and `VoyageTextEmbedder` now accept the `model` parameter instead of `model_name`.
- The embedders use the new `voyageai.Client.embed()` method instead of the deprecated `get_embedding` and `get_embeddings` methods.
- Support for the new `truncate` parameter has been added.
- The embedders now return the total number of tokens used as part of the `"total_tokens"` in the metadata.

## [1.1.0] - 2023-12-13

### Added

- Support for `input_type` parameter in `VoyageTextEmbedder` and `VoyageDocumentEmbedder`.

## [1.0.0] - 2023-11-21

### Added

- `VoyageTextEmbedder` and `VoyageDocumentEmbedder` to embed strings and documents.

[unreleased]: https://github.com/awinml/voyage-embedders-haystack/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/awinml/voyage-embedders-haystack/compare/v1.10.0...v2.0.0
[1.10.0]: https://github.com/awinml/voyage-embedders-haystack/compare/v1.9.1...v1.10.0
[1.9.1]: https://github.com/awinml/voyage-embedders-haystack/compare/v1.9.0...v1.9.1
[1.9.0]: https://github.com/awinml/voyage-embedders-haystack/compare/v1.8.0...v1.9.0
[1.8.0]: https://github.com/awinml/voyage-embedders-haystack/compare/v1.5.0...v1.8.0
[1.5.0]: https://github.com/awinml/voyage-embedders-haystack/compare/v1.4.0...v1.5.0
[1.4.0]: https://github.com/awinml/voyage-embedders-haystack/compare/v1.3.0...v1.4.0
[1.3.0]: https://github.com/awinml/voyage-embedders-haystack/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/awinml/voyage-embedders-haystack/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/awinml/voyage-embedders-haystack/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/awinml/voyage-embedders-haystack/releases/tag/v1.0.0
