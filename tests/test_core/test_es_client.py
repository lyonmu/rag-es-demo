from app.core.es_client import KB_INDEX_MAPPING


def test_kb_index_mapping_contains_optimization_fields() -> None:
    properties = KB_INDEX_MAPPING["mappings"]["properties"]

    assert properties["chunk_id"]["type"] == "keyword"
    assert properties["content_with_heading"]["type"] == "text"
    assert properties["content_with_heading"]["analyzer"] == "ik_max_word"
    assert properties["content_length"]["type"] == "integer"
    assert properties["chunk_text_hash"]["type"] == "keyword"
    assert properties["mapping_version"]["type"] == "keyword"
    assert properties["embedding"]["type"] == "dense_vector"
    assert properties["embedding"]["dims"] == 512
