from app.config import Settings


def test_optimization_settings_defaults(monkeypatch):
    for key in [
        "CHUNK_MAX_CHARS",
        "CHUNK_OVERLAP_CHARS",
        "CHUNK_MIN_CHARS",
        "QUERY_EMBEDDING_CACHE_SIZE",
        "INGEST_REFRESH_POLICY",
        "BM25_HEADING_BOOST",
        "BM25_CONTENT_BOOST",
        "BM25_PHRASE_BOOST",
        "CHAT_CONTEXT_MAX_CHARS",
        "VECTOR_CANDIDATE_MODE",
        "VECTOR_CANDIDATE_TOP_K",
    ]:
        monkeypatch.delenv(key, raising=False)

    settings = Settings(_env_file=None)

    assert settings.chunk_max_chars == 1200
    assert settings.chunk_overlap_chars == 150
    assert settings.chunk_min_chars == 80
    assert settings.query_embedding_cache_size == 256
    assert settings.ingest_refresh_policy == "false"
    assert settings.bm25_heading_boost == 2.0
    assert settings.bm25_content_boost == 1.0
    assert settings.bm25_phrase_boost == 1.5
    assert settings.chat_context_max_chars == 6000
    assert settings.vector_candidate_mode is False
    assert settings.vector_candidate_top_k == 200
