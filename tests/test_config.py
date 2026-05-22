from app.config import Settings


def test_settings_defaults():
    settings = Settings()

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
