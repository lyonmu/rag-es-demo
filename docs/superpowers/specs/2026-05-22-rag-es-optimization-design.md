# RAG ES Optimization Design

## Context

This project is a FastAPI-based Chinese RAG service using Elasticsearch 7.10, IK analysis, local BGE-Small-ZH-v1.5 embeddings, SQLite KB metadata, BM25 plus vector retrieval, RRF fusion, and SSE chat responses.

The optimization must keep the Elasticsearch version unchanged and keep the embedding model unchanged. The expected KB scale is uncertain, so the design targets a medium-size default workload while preserving room for larger indexes.

## Goals

- Improve retrieval quality, query latency, ingest stability, and operational clarity without changing ES or the embedding model.
- Prioritize low-risk changes with safe defaults and deterministic tests.
- Allow new KB indexes to benefit from improved mappings.
- Keep existing KBs usable, with full mapping benefits available after rebuilding indexes.

## Non-Goals

- Do not upgrade Elasticsearch.
- Do not replace BGE-Small-ZH-v1.5.
- Do not add an external vector database.
- Do not build a full offline evaluation platform.
- Do not automatically migrate old indexes in the first implementation phase.

## Recommended Approach

Use a phased version of the mapping plus retrieval-chain optimization approach.

Phase 1 keeps existing indexes compatible and adds low-risk code improvements. Phase 2 applies new mapping and chunking behavior to newly created KBs. Phase 3 can add explicit reindex or migration tooling for old KBs.

## Retrieval Architecture

The retrieval unit is a chunk, not a document. The current `_id` format already behaves like `doc_id_chunk_index`, so the optimized design makes this explicit with a `chunk_id` field and tests that describe RRF as chunk-level fusion.

The search flow is:

1. Encode the user query, using an in-process LRU cache for repeated queries.
2. Run BM25 retrieval with an enhanced query over content and heading-aware fields.
3. Run ES 7.10-compatible vector retrieval with `script_score`.
4. Fuse result lists with RRF.
5. Return top chunks to search clients or pass them to chat context assembly.
6. For chat, enforce a prompt context character budget and keep higher-scoring chunks first.

Vector retrieval still uses `script_score` because ES 7.10 has no native kNN. The default mode remains compatible with current behavior. A configurable candidate mode can later restrict vector scoring to a smaller candidate set when indexes grow.

## Index Mapping

New KB indexes should use a new mapping version. Existing fields remain:

- `doc_id`: `keyword`
- `filename`: `keyword`
- `heading_path`: `text`, `ik_max_word`
- `chunk_index`: `integer`
- `content`: `text`, `ik_max_word`
- `embedding`: `dense_vector`, `dims=512`
- `created_at`: `date`

Add these fields:

- `chunk_id`: `keyword`; stable chunk identifier.
- `content_with_heading`: `text`, `ik_max_word`; heading path plus content for heading-aware BM25 recall.
- `content_length`: `integer`; supports diagnostics, abnormal chunk filtering, and chat context budgeting.
- `chunk_text_hash`: `keyword`; supports duplicate detection and future idempotent ingest behavior.
- `mapping_version`: `keyword`; identifies optimized index structure.

Old indexes should remain searchable by falling back to `_id` where `chunk_id` is absent.

## Ingest Design

Ingest continues to accept Markdown and text uploads through the existing service layer.

Chunking changes:

- First split Markdown by heading hierarchy as today.
- Then split oversized sections by configurable character limits.
- Preserve `heading_path` on derived subchunks.
- Use overlap between subchunks to reduce boundary loss.

Recommended defaults:

- `chunk_max_chars=1200`
- `chunk_overlap_chars=150`
- `chunk_min_chars=80`

Embedding behavior:

- Embed the final chunk content.
- Do not prepend large heading text into the embedding input by default, to avoid repeated heading text dominating semantics.
- Use `content_with_heading` for BM25 only.

Bulk write behavior:

- Make the ES refresh policy configurable.
- Default to `refresh="false"` for better upload throughput.
- Allow `wait_for` in development or strict read-after-write scenarios.

Bulk error behavior:

- If all chunks fail, `ingest_markdown` raises an ingest exception and the upload router converts it to the existing unified ES operation error response.
- If some chunks fail, return success with `failed_chunks_count` and log failed chunk ids.
- Increment SQLite `doc_count` only when at least one chunk is indexed successfully.

Idempotency:

- Store `chunk_text_hash` in phase 1.
- Do not automatically skip or delete duplicate chunks yet, because duplicate handling is a product decision.
- Future options include same-filename overwrite and same-hash skip.

## BM25 Retrieval Design

Replace the single-field `match` query with a heading-aware query.

The default query should use `multi_match` over:

- `content` with normal boost.
- `heading_path` with heading boost.
- `content_with_heading` with moderate boost.

Add a phrase-oriented `should` clause for exact or near-exact phrase matches. Keep `minimum_should_match` conservative so semantic/vector retrieval remains useful for broader queries.

Recommended defaults:

- `bm25_content_boost=1.0`
- `bm25_heading_boost=2.0`
- `bm25_phrase_boost=1.5`

## Vector Retrieval Design

Keep ES 7.10-compatible `script_score` with cosine similarity.

Add query embedding cache:

- In-process LRU cache keyed by query text.
- Default cache size: `query_embedding_cache_size=256`.
- Cache only query embeddings, not document embeddings.

Add optional candidate mode:

- `vector_candidate_mode=false` by default.
- `vector_candidate_top_k=200` by default.
- When enabled, restrict vector scoring to a candidate set from BM25 or metadata filters.

Candidate mode is not required for the first implementation, but the configuration and interfaces should not prevent it.

## RRF Fusion Design

RRF remains the default fusion algorithm because it is robust to score scale differences between BM25 and vector search.

Required adjustments:

- Treat `chunk_id` as the primary result identity when present.
- Fall back to ES `_id` for old indexes.
- Keep `doc_id` as source document metadata, not the deduplication key.
- Preserve tie breakers that prefer stronger vector rank and BM25 rank.

## Chat Context Design

Chat should not pass unlimited retrieved content to the LLM prompt.

Add context budgeting:

- `chat_context_max_chars=6000` by default.
- Sort retrieved chunks by final RRF score.
- Add chunks until the budget is reached.
- If one chunk exceeds the remaining budget, truncate it rather than dropping all remaining context.

No-reference behavior:

- If retrieval returns no sources, the chat flow may still call the LLM.
- The prompt should clearly indicate that reference material is empty.
- Returning `NO_REFERENCE_FOUND` can be added later as a configurable strict mode.

## Error Handling

Search should keep the unified `{code, message, data}` API contract.

Retrieval degradation:

- If BM25 fails but vector succeeds, return vector-only results and log the degradation.
- If vector fails but BM25 succeeds, return BM25-only results and log the degradation.
- If both fail, return `SEARCH_ERROR`.

Ingest errors:

- Surface all-failed bulk writes as errors.
- Surface partial failures in response data.
- Keep logs detailed enough to identify failed chunk ids.

## Configuration

Add these settings with safe defaults:

- `chunk_max_chars=1200`
- `chunk_overlap_chars=150`
- `chunk_min_chars=80`
- `query_embedding_cache_size=256`
- `ingest_refresh_policy="false"`
- `bm25_heading_boost=2.0`
- `bm25_content_boost=1.0`
- `bm25_phrase_boost=1.5`
- `chat_context_max_chars=6000`
- `vector_candidate_mode=false`
- `vector_candidate_top_k=200`

## Testing Plan

Update or add tests in the existing pytest structure.

Chunker tests:

- Heading-based chunks still preserve heading paths.
- Oversized sections split into subchunks.
- Overlap is applied.
- Small trailing chunks are handled deterministically.

Ingest tests:

- Bulk documents include `chunk_id`, `content_with_heading`, `content_length`, `chunk_text_hash`, and `mapping_version`.
- Configured refresh policy is passed to ES bulk.
- Partial bulk failures return `failed_chunks_count`.
- All-failed bulk writes raise an ingest exception.

BM25 tests:

- Query body uses `multi_match`.
- Boost values come from settings.
- Phrase `should` clause is present.

Vector tests:

- Repeated identical query reuses the cached embedding.
- Default query remains ES 7.10 `script_score` compatible.
- Candidate-mode body can be tested separately if implemented.

Hybrid tests:

- `chunk_id` is preferred for deduplication.
- Old hits without `chunk_id` fall back to `_id`.
- RRF ordering remains stable.

Chat tests:

- Context assembly respects `chat_context_max_chars`.
- Higher-scoring sources are preserved first.
- Empty retrieval produces a clear empty-reference prompt path.

## Rollout Plan

1. Implement compatibility-safe configuration, chunking, query cache, BM25 body changes, RRF identity handling, and chat context budgeting.
2. Update mapping for newly created KB indexes.
3. Add tests for changed behavior.
4. Document that old KBs continue to work but require reindexing for new mapping fields.
5. Add reindex tooling later if migration becomes necessary.

## Risks and Mitigations

Risk: Smaller chunks may increase ES document count.
Mitigation: Use conservative defaults and keep chunk size configurable.

Risk: BM25 boost tuning may overvalue headings.
Mitigation: Keep boost values configurable and test only query structure, not subjective ranking.

Risk: `refresh="false"` changes read-after-write behavior.
Mitigation: Make refresh policy configurable and document the tradeoff.

Risk: Vector `script_score` remains expensive at large scale.
Mitigation: Preserve candidate-mode extension points for later staged rollout.

Risk: Partial ingest success may surprise clients.
Mitigation: Include explicit `failed_chunks_count` and log failed chunk ids.
