from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass


@dataclass(frozen=True)
class VectorRecord:
    artifact_id: str
    text: str


class FaissRetrievalStore:
    def __init__(self, embedding_model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self._require("faiss")
        self._require("sentence_transformers")
        sentence_transformers = importlib.import_module("sentence_transformers")
        self._faiss = importlib.import_module("faiss")
        self._encoder = sentence_transformers.SentenceTransformer(embedding_model_name)
        self._index = self._faiss.IndexFlatIP(self._encoder.get_sentence_embedding_dimension())
        self._payloads: dict[str, str] = {}
        self._id_order: list[str] = []

    @staticmethod
    def _require(module_name: str) -> None:
        if importlib.util.find_spec(module_name) is None:
            raise RuntimeError(f"Missing dependency: {module_name}")

    def upsert(self, record: VectorRecord) -> None:
        self._payloads[record.artifact_id] = record.text
        if record.artifact_id not in self._id_order:
            self._id_order.append(record.artifact_id)
        self.rebuild_index()

    def purge_ids(self, ids: set[str]) -> None:
        for artifact_id in ids:
            self._payloads.pop(artifact_id, None)
        self._id_order = [
            artifact_id for artifact_id in self._id_order if artifact_id in self._payloads
        ]
        self.rebuild_index()

    def rebuild_index(self) -> None:
        dim = self._encoder.get_sentence_embedding_dimension()
        self._index = self._faiss.IndexFlatIP(dim)
        if not self._id_order:
            return
        embeddings = self._encoder.encode(
            [self._payloads[k] for k in self._id_order], normalize_embeddings=True
        )
        self._index.add(embeddings)

    def search(self, query: str, top_k: int = 3) -> list[tuple[str, float]]:
        if not self._id_order:
            return []
        query_vec = self._encoder.encode([query], normalize_embeddings=True)
        scores, indices = self._index.search(query_vec, top_k)
        results: list[tuple[str, float]] = []
        for rank, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self._id_order):
                continue
            results.append((self._id_order[idx], float(scores[0][rank])))
        return results

    def export(self) -> dict[str, str]:
        return {artifact_id: self._payloads[artifact_id] for artifact_id in sorted(self._payloads)}
