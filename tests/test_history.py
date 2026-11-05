from mmrag.history import HistoryStore


def test_history_crud_search_duplicate_inputs_and_clear(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store = HistoryStore(tmp_path / "history.sqlite3")
    first = store.add("Revenue?", {"answer": "120", "evidence_ids": ["c1"]})
    second = store.add("Profit?", {"answer": "20"}, title="Profit question")
    assert [entry.history_id for entry in store.list("Revenue")] == [first.history_id]
    assert store.get(second.history_id) == second
    updated = store.update(first.history_id, title="Pinned revenue", pinned=True)
    assert updated is not None and updated.pinned and updated.title == "Pinned revenue"
    assert store.update("missing", title="none") is None
    assert store.delete(second.history_id)
    assert not store.delete(second.history_id)
    assert store.clear() == 1
    assert store.list() == []
