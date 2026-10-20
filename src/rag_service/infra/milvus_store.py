"""Milvus 向量存储：collection 校验、写入、检索、按文档删除。"""

from __future__ import annotations

from typing import Any

from rag_service.core.exceptions import AppError, ErrorCode
from rag_service.core.logging import get_logger
from rag_service.core.types import IndexedChunk, RetrievedChunk

logger = get_logger(__name__)

_OUTPUT_FIELDS = ["document_id", "chunk_index", "content", "parent_id", "parent_content", "source"]
_REQUIRED_FIELDS = {"parent_id", "parent_content", "sparse"}


class MilvusChunkStore:
    def __init__(
        self,
        uri: str,
        token: str | None,
        collection: str,
        dim: int,
        client: Any | None = None,
        database: str | None = None,
    ) -> None:
        self._collection = collection
        self._dim = dim
        self._client = client or self._build_client(uri, token, database)

    @staticmethod
    def _build_client(uri: str, token: str | None, database: str | None) -> Any:
        from pymilvus import MilvusClient

        kwargs: dict[str, Any] = {"uri": uri}
        if token:
            kwargs["token"] = token
        if database:
            kwargs["db_name"] = database
        return MilvusClient(**kwargs)

    def ensure_collection(self) -> None:
        try:
            if not self._client.has_collection(self._collection):
                self._create_collection()
                logger.info("已创建 Milvus collection: %s", self._collection)
                return
            self._validate_dimension()
        except AppError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("校验 Milvus collection 失败: %s", self._collection)
            raise AppError(ErrorCode.INTERNAL_ERROR, "Milvus collection 初始化失败") from exc

    def _validate_dimension(self) -> None:
        description = self._client.describe_collection(self._collection)
        names = {field.get("name") for field in description.get("fields", [])}
        missing = _REQUIRED_FIELDS - names
        if missing:
            raise AppError(
                ErrorCode.INTERNAL_ERROR,
                "Milvus 集合缺少父子分块字段 "
                f"({', '.join(sorted(missing))})，请运行 scripts/reset_kb.py 重建知识库",
            )
        for field in description.get("fields", []):
            if field.get("name") == "embedding":
                actual = int(field.get("params", {}).get("dim", 0))
                if actual and actual != self._dim:
                    raise AppError(
                        ErrorCode.INTERNAL_ERROR,
                        f"Milvus 集合维度不匹配: 期望 {self._dim}，实际 {actual}，请重建索引",
                    )

    def _create_collection(self) -> None:
        from pymilvus import DataType

        schema = self._client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=36)
        schema.add_field("document_id", DataType.VARCHAR, max_length=36)
        schema.add_field("parent_id", DataType.VARCHAR, max_length=36)
        schema.add_field("parent_content", DataType.VARCHAR, max_length=8192)
        schema.add_field("chunk_index", DataType.INT64)
        schema.add_field("content", DataType.VARCHAR, max_length=8192)
        schema.add_field("source", DataType.VARCHAR, max_length=255)
        schema.add_field("doc_type", DataType.VARCHAR, max_length=16)
        schema.add_field("created_at", DataType.INT64)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=self._dim)
        schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)

        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 200},
        )
        index_params.add_index(
            field_name="sparse",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
        )
        self._client.create_collection(
            collection_name=self._collection,
            schema=schema,
            index_params=index_params,
        )

    def upsert(self, chunks: list[IndexedChunk]) -> None:
        if not chunks:
            return
        rows = [
            {
                "id": chunk.id,
                "document_id": chunk.document_id,
                "parent_id": chunk.parent_id,
                "parent_content": chunk.parent_content,
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
                "source": chunk.source,
                "doc_type": chunk.doc_type,
                "created_at": chunk.created_at,
                "embedding": chunk.embedding,
                "sparse": chunk.sparse,
            }
            for chunk in chunks
        ]
        try:
            self._client.upsert(collection_name=self._collection, data=rows)
        except Exception as exc:  # noqa: BLE001
            logger.exception("写入 Milvus 失败，条数=%s", len(rows))
            raise AppError(ErrorCode.INTERNAL_ERROR, "写入向量库失败") from exc

    def _map_hits(self, results, score_threshold: float | None) -> list[RetrievedChunk]:
        hits: list[RetrievedChunk] = []
        for hit in (results[0] if results else []):
            score = float(hit.get("distance", 0.0))
            if score_threshold is not None and score < score_threshold:
                continue
            entity = hit.get("entity", {})
            parent_id = str(entity.get("parent_id") or hit.get("id"))
            parent_content = str(entity.get("parent_content") or entity.get("content", ""))
            hits.append(
                RetrievedChunk(
                    id=parent_id,
                    document_id=str(entity.get("document_id", "")),
                    content=parent_content,
                    source=str(entity.get("source", "")),
                    chunk_index=int(entity.get("chunk_index", 0)),
                    score=score,
                    snippet=str(entity.get("content", "")),
                )
            )
        hits.sort(key=lambda item: item.score, reverse=True)
        return hits

    def search(
        self,
        vector: list[float],
        top_k: int,
        score_threshold: float,
    ) -> list[RetrievedChunk]:
        try:
            results = self._client.search(
                collection_name=self._collection,
                data=[vector],
                limit=top_k,
                output_fields=_OUTPUT_FIELDS,
                search_params={"metric_type": "COSINE", "params": {}},
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Milvus 检索失败")
            raise AppError(ErrorCode.INTERNAL_ERROR, "向量检索失败") from exc

        return self._map_hits(results, score_threshold)

    def hybrid_search(
        self,
        dense_vector: list[float],
        sparse_vector: dict[int, float],
        branch_limit: int,
        limit: int,
        rrf_k: int = 60,
    ) -> list[RetrievedChunk]:
        """稠密 + 稀疏双路召回，RRF 融合。"""
        from pymilvus import AnnSearchRequest, RRFRanker

        requests = [
            AnnSearchRequest(
                data=[dense_vector],
                anns_field="embedding",
                param={"metric_type": "COSINE", "params": {}},
                limit=branch_limit,
            ),
            AnnSearchRequest(
                data=[sparse_vector],
                anns_field="sparse",
                param={"metric_type": "IP"},
                limit=branch_limit,
            ),
        ]
        try:
            results = self._client.hybrid_search(
                collection_name=self._collection,
                reqs=requests,
                ranker=RRFRanker(rrf_k),
                limit=limit,
                output_fields=_OUTPUT_FIELDS,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Milvus 混合检索失败")
            raise AppError(ErrorCode.INTERNAL_ERROR, "混合检索失败") from exc

        return self._map_hits(results, None)

    def delete_document(self, document_id: str) -> None:
        try:
            self._client.delete(
                collection_name=self._collection,
                filter=f'document_id == "{document_id}"',
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("删除 Milvus 向量失败: document_id=%s", document_id)
            raise AppError(ErrorCode.INTERNAL_ERROR, "删除向量失败") from exc

    def drop_and_recreate(self) -> None:
        try:
            if self._client.has_collection(self._collection):
                self._client.drop_collection(self._collection)
            self._create_collection()
        except Exception as exc:  # noqa: BLE001
            logger.exception("重建 Milvus collection 失败")
            raise AppError(ErrorCode.INTERNAL_ERROR, "重建向量集合失败") from exc

    def ping(self) -> bool:
        return bool(self._client.has_collection(self._collection))
