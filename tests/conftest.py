"""Offline tests never touch the live SurrealDB: memory stays in SQLite and the shared mind is off."""

import pytest


@pytest.fixture(autouse=True)
def offline_stores(monkeypatch):
    monkeypatch.setenv("FRONTIER_N3", "sqlite")
    monkeypatch.setenv("FRONTIER_MIND", "off")
