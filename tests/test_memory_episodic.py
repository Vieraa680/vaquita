from pathlib import Path

from vaquita.memory import BaseMemory, EpisodicMemoryStore


def test_episodic_memory_store_implements_protocol(tmp_path: Path) -> None:
    store = EpisodicMemoryStore(db_path=tmp_path / "memory.db")
    assert isinstance(store, BaseMemory)


def test_episodic_memory_store_lifecycle_and_relevance_search(tmp_path: Path) -> None:
    db_file = tmp_path / "memory.db"
    store = EpisodicMemoryStore(db_path=db_file)

    id_1 = store.store(
        content="Configuramos el servidor nginx en el puerto 8080",
        category="system",
        tags=["nginx", "web"],
    )
    id_2 = store.store(
        content="Entrenamos el mini transformer con atencion causal y embeddings",
        category="neural",
        tags=["transformer", "ia"],
    )
    id_3 = store.store(
        content="Compramos cafe de grano para la oficina",
        category="personal",
        tags=["compras"],
    )

    assert id_1 and id_2 and id_3

    # Query matching nginx episode
    nginx_results = store.search("nginx servidor", limit=2)
    assert len(nginx_results) > 0
    assert nginx_results[0].id == id_1
    assert nginx_results[0].score > 0.0
    assert "nginx" in nginx_results[0].tags

    # Category filtered search
    neural_results = store.search("transformer", category="neural")
    assert len(neural_results) == 1
    assert neural_results[0].id == id_2

    # Query with no match
    empty_results = store.search("astronauta luna marte")
    assert empty_results == []

    # Deletion
    assert store.delete(id_3) is True
    assert store.delete("non_existent_id") is False
    assert len(store.search("cafe")) == 0
