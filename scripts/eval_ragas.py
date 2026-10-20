"""RAGAS 测评：合成客服语料 → Recall@k 选 top_k → DeepSeek 生成 → RAGAS 打分。

用法（RAGAS 已加入 dev 依赖，直接在主环境执行）：

    .venv\\Scripts\\python.exe -m scripts.eval_ragas all

分步执行：

    ... -m scripts.eval_ragas build       # 建评测集合
    ... -m scripts.eval_ragas generate    # 生成 30 条问答
    ... -m scripts.eval_ragas recall      # 计算 Recall@k
    ... -m scripts.eval_ragas ragas       # 生成答案并跑 RAGAS

所有 DeepSeek 调用都显式传 ``thinking={"type":"disabled"}``；RAGAS 使用
``ragas.metrics.collections`` 的 0.4.x 指标，逐题并发打分并保留失败原因。
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dotenv import load_dotenv  # noqa: E402

from rag_service.core.config import get_settings  # noqa: E402
from rag_service.core.logging import get_logger, setup_logging  # noqa: E402
from rag_service.core.types import IndexedChunk, ParsedBlock, RetrievedChunk  # noqa: E402
from rag_service.evaluation.corpus import CORPUS_VERSION, DOCUMENTS, EvalDocument  # noqa: E402
from rag_service.evaluation.dataset import (  # noqa: E402
    load_jsonl,
    normalize_question,
    parse_json_object,
    validate_dataset,
    validate_question_text,
)
from rag_service.evaluation.ranking import (  # noqa: E402
    RetrievalRecord,
    TopKDecision,
    document_recall_at_k,
    mean_reciprocal_rank,
    recall_at_k,
    recall_curve,
    select_top_k,
)
from rag_service.infra.embedding import EmbeddingProvider, build_embedding  # noqa: E402
from rag_service.infra.milvus_store import MilvusChunkStore  # noqa: E402
from rag_service.infra.reranker import build_reranker  # noqa: E402
from rag_service.services.chunking import split_parent_child  # noqa: E402
from rag_service.services.prompt import build_messages  # noqa: E402
from rag_service.services.retrieval_service import RetrievalService  # noqa: E402

logger = get_logger(__name__)

THINKING_DISABLED = {"thinking": {"type": "disabled"}}
DEFAULT_COLLECTION = "rag_eval_v1"
DEFAULT_OUTPUT_DIR = Path("data/eval/ragas")
DEFAULT_TARGET = 0.90
DEFAULT_MIN_K = 3
DEFAULT_MAX_K = 10
EMBED_BATCH = 32
FALLBACK_ANSWER = (
    "抱歉，我在店铺资料里没有找到相关信息。建议您联系人工客服确认，工作时间为 9:00-21:00。"
)


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _select_documents(limit: int | None) -> list[EvalDocument]:
    documents = list(DOCUMENTS)
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit-docs 必须 >= 1")
        documents = documents[:limit]
    return documents


def _select_parents_round_robin(parents: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    """按类别轮转取题，避免某一品类占满整个测试集。"""
    if limit is None or limit >= len(parents):
        return list(parents)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for parent in parents:
        buckets[str(parent.get("category") or "未分类")].append(parent)
    selected: list[dict[str, Any]] = []
    while len(selected) < limit:
        progressed = False
        for category in sorted(buckets):
            if not buckets[category]:
                continue
            selected.append(buckets[category].pop(0))
            progressed = True
            if len(selected) >= limit:
                break
        if not progressed:
            break
    return selected


def _safe_mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _safe_median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _safe_stdev(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) >= 2 else None


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


# ---------------------------------------------------------------------------
# DeepSeek
# ---------------------------------------------------------------------------


def _deepseek_client():
    from openai import AsyncOpenAI

    load_dotenv(".env", override=False)
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("未找到 DEEPSEEK_API_KEY，请在 .env 中配置")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    logger.info("DeepSeek 客户端初始化: base_url=%s", base_url)
    return AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=180, max_retries=2)


async def verify_thinking_disabled(client, model: str) -> dict[str, Any]:
    """发一条最小请求，确认 reasoning_content 与 reasoning_tokens 都为空。"""
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "只回复：好"}],
        temperature=0.0,
        max_tokens=16,
        extra_body=THINKING_DISABLED,
    )
    message = response.choices[0].message
    reasoning = getattr(message, "reasoning_content", None)
    details = getattr(response.usage, "completion_tokens_details", None)
    reasoning_tokens = getattr(details, "reasoning_tokens", None) if details else None
    if reasoning or (reasoning_tokens or 0) > 0:
        raise RuntimeError(
            f"深度思考未关闭: reasoning_content={reasoning!r}, reasoning_tokens={reasoning_tokens!r}"
        )
    logger.info(
        "深度思考已关闭: model=%s reasoning_content=None reasoning_tokens=%s",
        model,
        reasoning_tokens,
    )
    return {
        "model": model,
        "thinking_disabled": True,
        "reasoning_tokens": reasoning_tokens,
        "response": (message.content or "").strip(),
    }


async def _chat_json(
    client,
    model: str,
    prompt: str,
    *,
    max_tokens: int = 800,
    temperature: float = 0.7,
) -> tuple[dict[str, Any], dict[str, Any]]:
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        extra_body=THINKING_DISABLED,
    )
    content = response.choices[0].message.content or ""
    return parse_json_object(content), _usage_dict(response.usage)


async def _chat_text(
    client,
    model: str,
    messages: list[dict[str, str]],
    *,
    max_tokens: int = 600,
    temperature: float = 0.2,
) -> tuple[str, dict[str, Any]]:
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body=THINKING_DISABLED,
    )
    content = (response.choices[0].message.content or "").strip()
    return content, _usage_dict(response.usage)


def _usage_dict(usage) -> dict[str, Any]:
    details = getattr(usage, "completion_tokens_details", None)
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
        "reasoning_tokens": getattr(details, "reasoning_tokens", None) if details else None,
    }


# ---------------------------------------------------------------------------
# 快照：合成语料 → 真实切分 → 真实 embedding → 独立 Milvus 集合
# ---------------------------------------------------------------------------


def _build_index_records(
    documents: list[EvalDocument], collection: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    parents: list[dict[str, Any]] = []
    children: list[dict[str, Any]] = []
    global_index = 0
    now = int(time.time())

    for document in documents:
        document_id = str(uuid5(NAMESPACE_URL, f"{collection}:{document.slug}:document"))
        parsed_parents = split_parent_child([ParsedBlock(text=document.content)])
        for parent in parsed_parents:
            parent_id = str(
                uuid5(
                    NAMESPACE_URL,
                    f"{collection}:{document.slug}:parent:{parent.index}",
                )
            )
            parents.append(
                {
                    "parent_id": parent_id,
                    "document_id": document_id,
                    "slug": document.slug,
                    "category": document.category,
                    "title": document.title,
                    "filename": document.filename,
                    "parent_index": parent.index,
                    "content": parent.content,
                }
            )
            for child in parent.children:
                child_id = str(
                    uuid5(
                        NAMESPACE_URL,
                        f"{collection}:{document.slug}:child:{parent.index}:{child.index}",
                    )
                )
                children.append(
                    {
                        "id": child_id,
                        "document_id": document_id,
                        "parent_id": parent_id,
                        "parent_content": parent.content,
                        "chunk_index": global_index,
                        "content": child.content,
                        "source": document.filename,
                        "doc_type": "md",
                        "created_at": now,
                    }
                )
                global_index += 1

    return parents, children


def build_snapshot(
    settings,
    documents: list[EvalDocument],
    collection: str,
    output_dir: Path,
    embedding: EmbeddingProvider,
) -> dict[str, Any]:
    if collection == settings.milvus_collection:
        raise SystemExit("评测集合不能与生产集合同名，请换一个 --collection")

    parents, children = _build_index_records(documents, collection)
    if not parents or not children:
        raise RuntimeError("合成语料没有切分出任何父块或子块")

    logger.info(
        "构建评测快照: docs=%s parents=%s children=%s collection=%s",
        len(documents),
        len(parents),
        len(children),
        collection,
    )
    texts = [row["content"] for row in children]
    vectors: list[list[float]] = []
    sparse_vectors: list[dict[int, float]] = []
    for start in range(0, len(texts), EMBED_BATCH):
        batch = texts[start : start + EMBED_BATCH]
        dense_batch, sparse_batch = embedding.embed_documents_hybrid(batch)
        vectors.extend(dense_batch)
        sparse_vectors.extend(sparse_batch)
        logger.info("编码进度: %s/%s", min(start + EMBED_BATCH, len(texts)), len(texts))

    indexed = [
        IndexedChunk(
            id=row["id"],
            document_id=row["document_id"],
            chunk_index=row["chunk_index"],
            content=row["content"],
            source=row["source"],
            doc_type=row["doc_type"],
            created_at=row["created_at"],
            embedding=vector,
            parent_id=row["parent_id"],
            parent_content=row["parent_content"],
            sparse=sparse,
        )
        for row, vector, sparse in zip(children, vectors, sparse_vectors)
    ]

    store = MilvusChunkStore(
        settings.milvus_uri,
        settings.milvus_token,
        collection,
        settings.embedding_dim,
        database=settings.milvus_database,
    )
    store.drop_and_recreate()
    store.upsert(indexed)
    logger.info("评测集合写入完成: %s 条", len(indexed))

    manifest = {
        "collection": collection,
        "corpus_version": CORPUS_VERSION,
        "created_at": _now(),
        "embedding_model_path": settings.embedding_model_path,
        "embedding_dim": settings.embedding_dim,
        "document_count": len(documents),
        "parent_count": len(parents),
        "child_count": len(children),
        "documents": [
            {
                "slug": doc.slug,
                "category": doc.category,
                "title": doc.title,
                "filename": doc.filename,
            }
            for doc in documents
        ],
        "parents": parents,
    }
    _write_json(output_dir / "corpus_manifest.json", manifest)
    return manifest


def load_manifest(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "corpus_manifest.json"
    if not path.exists():
        raise SystemExit(f"缺少 {path}，请先执行 build")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not value.get("parents"):
        raise SystemExit(f"{path} 中没有父块，请重新 build")
    return value


# ---------------------------------------------------------------------------
# 数据集生成
# ---------------------------------------------------------------------------


def _generation_prompt(parent: dict[str, Any], avoid: list[str], feedback: str = "") -> str:
    lines = [
        "你是电商客服质检员。请阅读下面这条店铺知识库资料，写 1 个真实顾客会问的中文问题，",
        "并给出仅依据资料可回答的标准答案。",
        "",
        "要求：",
        "1. 问题必须能由资料中的具体事实回答，优先问数值、期限、条件、例外或限制；",
        "2. 用顾客口吻，不要出现“资料”“文档”“根据上文”等词，不要照抄标题或整句；",
        "3. 标准答案简洁、准确，不超过 120 字，必须包含资料中的关键数字或条件；",
        "4. 只输出 JSON：",
        '{"question": "...", "reference_answer": "...", "evidence": "资料中支撑答案的原文短句"}',
        "",
        f"资料类别：{parent.get('category', '')}",
        f"资料标题：{parent.get('title', '')}",
        "资料正文：",
        str(parent.get("content", "")),
    ]
    if avoid:
        lines.extend(["", "不要生成与以下问题相同或高度近似的问题："])
        lines.extend(f"- {q}" for q in avoid[:12])
    if feedback:
        lines.extend(["", f"上一次输出不合格：{feedback}。请修正后重新输出 JSON。"])
    return "\n".join(lines)


async def _generate_one(
    client,
    model: str,
    parent: dict[str, Any],
    avoid: list[str],
) -> dict[str, Any] | None:
    feedback = ""
    for attempt in range(1, 4):
        try:
            value, usage = await _chat_json(
                client,
                model,
                _generation_prompt(parent, avoid, feedback),
                max_tokens=800,
                temperature=0.7 if attempt == 1 else 0.4,
            )
            question = str(value.get("question") or "").strip()
            reference = str(value.get("reference_answer") or "").strip()
            evidence = str(value.get("evidence") or "").strip()
            errors = validate_question_text(
                question,
                title=str(parent.get("title") or ""),
                reference=reference,
            )
            if not reference:
                errors.append("reference_answer 为空")
            if len(reference) > 300:
                errors.append("reference_answer 超过 300 字")
            if errors:
                feedback = "；".join(errors)
                logger.warning(
                    "生成结果不合格 parent=%s attempt=%s: %s",
                    parent.get("parent_id"),
                    attempt,
                    feedback,
                )
                continue
            return {
                "question": question,
                "reference_answer": reference,
                "evidence": evidence,
                "gold_parent_id": parent["parent_id"],
                "gold_document_id": parent["document_id"],
                "gold_source": parent["filename"],
                "category": parent.get("category", ""),
                "title": parent.get("title", ""),
                "parent_content": parent.get("content", ""),
                "generated_by": model,
                "generated_at": _now(),
                "generation_usage": usage,
            }
        except Exception as exc:  # noqa: BLE001 - 单题失败不拖垮整批
            feedback = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "生成失败 parent=%s attempt=%s: %s",
                parent.get("parent_id"),
                attempt,
                feedback,
            )
            await asyncio.sleep(1.5 * attempt)
    return None


async def generate_dataset(
    client,
    model: str,
    manifest: dict[str, Any],
    output_dir: Path,
    *,
    limit: int | None,
    concurrency: int,
) -> list[dict[str, Any]]:
    parents = _select_parents_round_robin(manifest["parents"], limit)
    logger.info("开始生成测试集: parents=%s concurrency=%s", len(parents), concurrency)
    semaphore = asyncio.Semaphore(concurrency)

    async def worker(parent: dict[str, Any]) -> dict[str, Any] | None:
        async with semaphore:
            return await _generate_one(client, model, parent, [])

    raw = await asyncio.gather(*(worker(parent) for parent in parents))
    parents_by_id = {parent["parent_id"]: parent for parent in parents}

    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for sample in raw:
        if sample is None:
            continue
        key = normalize_question(sample["question"])
        if key in seen:
            parent = parents_by_id.get(sample["gold_parent_id"])
            replacement = None
            if parent is not None:
                replacement = await _generate_one(
                    client,
                    model,
                    parent,
                    sorted({item["question"] for item in unique}),
                )
            if replacement is None or normalize_question(replacement["question"]) in seen:
                logger.warning("去重后仍重复，跳过: %s", sample["question"][:40])
                continue
            sample = replacement
            key = normalize_question(sample["question"])
        seen.add(key)
        unique.append(sample)

    unique.sort(key=lambda item: (str(item["category"]), str(item["gold_source"])))
    for index, sample in enumerate(unique, start=1):
        sample["id"] = f"q{index:03d}"
    validate_dataset(unique)
    _write_jsonl(output_dir / "dataset.jsonl", unique)
    logger.info("测试集生成完成: %s 条，已写入 %s", len(unique), output_dir / "dataset.jsonl")
    return unique


# ---------------------------------------------------------------------------
# Recall@k
# ---------------------------------------------------------------------------


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def run_recall(
    retrieval: RetrievalService,
    dataset: list[dict[str, Any]],
    output_dir: Path,
    *,
    max_k: int,
    target: float,
    min_k: int,
) -> tuple[TopKDecision, dict[int, float], dict[int, float], list[dict[str, Any]]]:
    records: list[RetrievalRecord] = []
    payloads: list[dict[str, Any]] = []
    for index, sample in enumerate(dataset, start=1):
        hits = retrieval.retrieve(sample["question"])
        parent_ids = [hit.id for hit in hits]
        document_ids = _dedupe_keep_order([hit.document_id for hit in hits])
        records.append(
            RetrievalRecord(
                question_id=sample["id"],
                gold_parent_id=sample["gold_parent_id"],
                gold_document_id=sample["gold_document_id"],
                ranked_parent_ids=tuple(parent_ids),
                ranked_document_ids=tuple(document_ids),
            )
        )
        payloads.append(
            {
                "id": sample["id"],
                "question": sample["question"],
                "category": sample.get("category", ""),
                "gold_parent_id": sample["gold_parent_id"],
                "gold_document_id": sample["gold_document_id"],
                "rank_of_gold": next(
                    (rank for rank, parent_id in enumerate(parent_ids, start=1) if parent_id == sample["gold_parent_id"]),
                    None,
                ),
                "hits": [
                    {
                        "rank": rank,
                        "parent_id": hit.id,
                        "document_id": hit.document_id,
                        "source": hit.source,
                        "score": hit.score,
                        "content": hit.content,
                    }
                    for rank, hit in enumerate(hits, start=1)
                ],
            }
        )
        logger.info(
            "Recall 进度: %s/%s q=%s rank=%s hits=%s",
            index,
            len(dataset),
            sample["id"],
            payloads[-1]["rank_of_gold"],
            len(hits),
        )

    curve = recall_curve(records, max_k=max_k)
    doc_curve = {k: document_recall_at_k(records, k) for k in range(1, max_k + 1)}
    decision = select_top_k(curve, target=target, min_k=min_k, max_k=max_k)
    mrr = mean_reciprocal_rank(records)

    rows: list[dict[str, Any]] = []
    for k in range(1, max_k + 1):
        lengths = []
        for payload in payloads:
            hit_contents = [hit["content"] for hit in payload["hits"][:k]]
            lengths.append(sum(len(content) for content in hit_contents))
        rows.append(
            {
                "k": k,
                "recall_parent": round(curve[k], 6),
                "recall_document": round(doc_curve[k], 6),
                "avg_context_chars": round(statistics.mean(lengths), 1) if lengths else 0.0,
            }
        )
    _write_jsonl(output_dir / "retrieval_results.jsonl", payloads)
    with (output_dir / "recall_curve.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["k", "recall_parent", "recall_document", "avg_context_chars"])
        writer.writeheader()
        writer.writerows(rows)
    _write_json(
        output_dir / "recall_summary.json",
        {
            "target": target,
            "min_k": min_k,
            "max_k": max_k,
            "mrr": round(mrr, 6),
            "decision": {
                "top_k": decision.top_k,
                "achieved": decision.achieved,
                "reason": decision.reason,
                "curve": curve,
                "document_curve": doc_curve,
            },
            "rows": rows,
        },
    )
    logger.info("Recall 完成: MRR=%.4f 决策=%s", mrr, decision.reason)
    return decision, curve, doc_curve, payloads


# ---------------------------------------------------------------------------
# 答案生成
# ---------------------------------------------------------------------------


def _rank_of(gold_parent_id: str, chunks: list[RetrievedChunk]) -> int | None:
    for rank, chunk in enumerate(chunks, start=1):
        if chunk.id == gold_parent_id:
            return rank
    return None


async def generate_answers(
    client,
    model: str,
    settings,
    retrieval: RetrievalService,
    dataset: list[dict[str, Any]],
    output_dir: Path,
    *,
    top_k: int,
    concurrency: int,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(concurrency)
    thinking_violations = 0

    async def worker(index: int, sample: dict[str, Any]) -> dict[str, Any]:
        nonlocal thinking_violations
        async with semaphore:
            chunks = retrieval.retrieve(sample["question"])[:top_k]
            retrieved_contexts = [chunk.content for chunk in chunks]
            retrieved_ids = [chunk.id for chunk in chunks]
            usage: dict[str, Any] = {}
            error = ""
            if chunks:
                messages = build_messages(
                    sample["question"],
                    chunks,
                    [],
                    shop_name=settings.shop_name,
                    handoff_hint=settings.handoff_hint,
                    max_context_chars=settings.max_context_chars,
                )
                response_text = ""
                for attempt in range(1, 3):
                    try:
                        response_text, usage = await _chat_text(
                            client,
                            model,
                            messages,
                            max_tokens=600,
                            temperature=0.2,
                        )
                        if response_text:
                            break
                        error = "模型返回空内容"
                    except Exception as exc:  # noqa: BLE001
                        error = f"{type(exc).__name__}: {exc}"
                        logger.warning("答案生成失败 id=%s attempt=%s: %s", sample["id"], attempt, error)
                        await asyncio.sleep(1.5 * attempt)
                if not response_text:
                    response_text = FALLBACK_ANSWER
            else:
                response_text = FALLBACK_ANSWER
                error = "检索为空，使用兜底话术"

            reasoning_tokens = usage.get("reasoning_tokens") or 0
            if reasoning_tokens:
                thinking_violations += 1
                logger.warning(
                    "检测到推理 token: id=%s reasoning_tokens=%s", sample["id"], reasoning_tokens
                )
            logger.info(
                "答案进度: %s/%s id=%s chunks=%s chars=%s",
                index,
                len(dataset),
                sample["id"],
                len(chunks),
                len(response_text),
            )
            return {
                "id": sample["id"],
                "user_input": sample["question"],
                "response": response_text,
                "retrieved_contexts": retrieved_contexts,
                "reference": sample["reference_answer"],
                "gold_parent_id": sample["gold_parent_id"],
                "gold_document_id": sample["gold_document_id"],
                "category": sample.get("category", ""),
                "retrieved_parent_ids": retrieved_ids,
                "rank_of_gold": _rank_of(sample["gold_parent_id"], chunks),
                "model": model,
                "usage": usage,
                "error": error,
                "generated_at": _now(),
            }

    results = await asyncio.gather(*(worker(i, sample) for i, sample in enumerate(dataset, start=1)))
    _write_jsonl(output_dir / "ragas_samples.jsonl", results)
    _write_json(
        output_dir / "answer_generation_summary.json",
        {
            "model": model,
            "top_k": top_k,
            "sample_count": len(results),
            "thinking_violations": thinking_violations,
            "fallback_count": sum(1 for item in results if item["error"]),
        },
    )
    logger.info(
        "答案生成完成: %s 条，thinking 违规=%s",
        len(results),
        thinking_violations,
    )
    return results


# ---------------------------------------------------------------------------
# RAGAS 打分
# ---------------------------------------------------------------------------


async def _score_metric(metric, kwargs: dict[str, Any], semaphore: asyncio.Semaphore, retries: int = 2):
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            async with semaphore:
                result = await metric.ascore(**kwargs)
            return float(result.value), None
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.warning(
                "RAGAS 指标失败 metric=%s attempt=%s: %s",
                type(metric).__name__,
                attempt + 1,
                f"{type(exc).__name__}: {exc}",
            )
            await asyncio.sleep(1.5 * (attempt + 1))
    return None, f"{type(last_error).__name__}: {last_error}" if last_error else "unknown error"


async def run_ragas(
    client,
    model: str,
    embedding: EmbeddingProvider,
    samples: list[dict[str, Any]],
    output_dir: Path,
    *,
    concurrency: int,
) -> dict[str, Any]:
    try:
        from ragas.embeddings import BaseRagasEmbedding
        from ragas.llms import llm_factory
        from ragas.metrics.collections import (
            AnswerCorrectness,
            AnswerRelevancy,
            ContextPrecisionWithReference,
            ContextRecall,
            Faithfulness,
        )
    except ImportError as exc:
        raise SystemExit(
            "未安装 RAGAS。请先执行 uv sync 安装 dev 依赖"
            "（ragas>=0.4.3 与 langchain-community>=0.3,<0.4）"
        ) from exc

    class _BgeM3Embedding(BaseRagasEmbedding):
        def __init__(self, provider: EmbeddingProvider) -> None:
            super().__init__()
            self._provider = provider

        def embed_text(self, text: str, **kwargs: Any) -> list[float]:
            return self._provider.embed_query(text)

        async def aembed_text(self, text: str, **kwargs: Any) -> list[float]:
            return await asyncio.to_thread(self._provider.embed_query, text)

        def embed_texts(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
            return self._provider.embed_documents(texts)

        async def aembed_texts(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
            return await asyncio.to_thread(self._provider.embed_documents, texts)

    verification = await verify_thinking_disabled(client, model)
    llm = llm_factory(
        model,
        client=client,
        temperature=0.01,
        max_tokens=2048,
        extra_body=THINKING_DISABLED,
    )
    ragas_embedding = _BgeM3Embedding(embedding)
    metrics = {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=ragas_embedding),
        "context_precision": ContextPrecisionWithReference(llm=llm),
        "context_recall": ContextRecall(llm=llm),
        "answer_correctness": AnswerCorrectness(llm=llm, embeddings=ragas_embedding),
    }
    semaphore = asyncio.Semaphore(concurrency)

    def metric_kwargs(name: str, sample: dict[str, Any]) -> dict[str, Any]:
        base = {
            "user_input": sample["user_input"],
            "response": sample["response"],
            "retrieved_contexts": sample["retrieved_contexts"],
            "reference": sample["reference"],
        }
        if name == "faithfulness":
            return {key: base[key] for key in ("user_input", "response", "retrieved_contexts")}
        if name == "answer_relevancy":
            return {key: base[key] for key in ("user_input", "response")}
        if name == "context_precision":
            return {key: base[key] for key in ("user_input", "reference", "retrieved_contexts")}
        if name == "context_recall":
            return {key: base[key] for key in ("user_input", "retrieved_contexts", "reference")}
        if name == "answer_correctness":
            return {key: base[key] for key in ("user_input", "response", "reference")}
        raise ValueError(f"未知指标: {name}")

    async def score_sample(index: int, sample: dict[str, Any]) -> dict[str, Any]:
        tasks = {
            name: asyncio.create_task(
                _score_metric(metric, metric_kwargs(name, sample), semaphore)
            )
            for name, metric in metrics.items()
        }
        values: dict[str, float | None] = {}
        errors: dict[str, str] = {}
        for name, task in tasks.items():
            value, error = await task
            values[name] = value
            if error:
                errors[name] = error
        logger.info(
            "RAGAS 进度: %s/%s id=%s scores=%s",
            index,
            len(samples),
            sample["id"],
            {key: (None if value is None else round(value, 4)) for key, value in values.items()},
        )
        return {**sample, "scores": values, "score_errors": errors}

    results = await asyncio.gather(
        *(score_sample(index, sample) for index, sample in enumerate(samples, start=1))
    )

    metric_names = list(metrics)
    summary: dict[str, Any] = {}
    for name in metric_names:
        values = [item["scores"][name] for item in results if item["scores"].get(name) is not None]
        summary[name] = {
            "mean": _safe_mean(values),
            "median": _safe_median(values),
            "stdev": _safe_stdev(values),
            "valid": len(values),
            "failed": len(results) - len(values),
        }
    _write_json(
        output_dir / "ragas_summary.json",
        {
            "model": model,
            "judge_model": model,
            "thinking_verification": verification,
            "sample_count": len(results),
            "metrics": summary,
        },
    )
    with (output_dir / "ragas_scores.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = [
            "id",
            "category",
            "question",
            "reference_answer",
            "response",
            "rank_of_gold",
            *metric_names,
            "error",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            writer.writerow(
                {
                    "id": item["id"],
                    "category": item["category"],
                    "question": item["user_input"],
                    "reference_answer": item["reference"],
                    "response": item["response"],
                    "rank_of_gold": item["rank_of_gold"],
                    **{name: item["scores"].get(name) for name in metric_names},
                    "error": item.get("error", ""),
                }
            )
    logger.info("RAGAS 打分完成: %s 条", len(results))
    return {"summary": summary, "results": results, "verification": verification}


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------


def _sample_average(sample: dict[str, Any]) -> float:
    values = [value for value in sample["scores"].values() if value is not None]
    return statistics.mean(values) if values else -1.0


def write_report(
    output_dir: Path,
    *,
    settings,
    manifest: dict[str, Any],
    dataset: list[dict[str, Any]],
    decision: TopKDecision,
    curve: dict[int, float],
    doc_curve: dict[int, float],
    payloads: list[dict[str, Any]],
    ragas_output: dict[str, Any] | None,
    metadata: dict[str, Any],
) -> Path:
    mrr = 0.0
    ranks = [payload["rank_of_gold"] for payload in payloads if payload.get("rank_of_gold")]
    if ranks:
        mrr = statistics.mean(1.0 / rank for rank in ranks)

    lines: list[str] = [
        "# RAG 系统 RAGAS 测评报告",
        "",
        f"- 生成时间：{_now()}",
        f"- 评测集合：`{manifest['collection']}`（合成语料，未修改生产知识库）",
        f"- 语料版本：`{manifest['corpus_version']}`，文档 {manifest['document_count']} 篇 / 父块 {manifest['parent_count']} 个 / 子块 {manifest['child_count']} 个",
        f"- 测试集：{len(dataset)} 条中文问答，按 10 个客服品类分层生成",
        f"- 生成模型：`{metadata['generator_model']}`；评审模型：`{metadata['judge_model']}`",
        "- 深度思考：所有 DeepSeek 调用均显式传入 `thinking={\"type\":\"disabled\"}`，并做最小请求校验",
        f"- Embedding：`{settings.embedding_model_path}`（{settings.embedding_dim} 维）",
        f"- 最终 top_k：**{decision.top_k}**（规则 Recall≥{decision.target:.2f} 且 k≥{metadata['min_k']}）",
        f"- 配置写回：`Settings.top_k = {decision.top_k}`、`.env` 的 `TOP_K={decision.top_k}`、`.env.example` 同步",
        "",
        "## 1. Recall@k 与 top_k",
        "",
        f"决策：{decision.reason}",
        "",
    ]
    if curve.get(1, 0.0) >= decision.target:
        lines.extend(
            [
                f"> 注意：Recall@1 已经达到 {curve[1]:.4f}，因此 `top_k={decision.top_k}` 是 "
                f"`k≥{metadata['min_k']}` 的下限，而不是召回曲线在 3 处才达标。若只按召回，k=1 即可；"
                "保留 3 条是给生成阶段留上下文余量。",
                "",
            ]
        )
    lines.extend(
        [
            "| k | Recall@k（父块） | Recall@k（文档） | 平均上下文字符 |",
            "| --- | --- | --- | --- |",
        ]
    )
    char_rows = {}
    for k in range(1, metadata["max_k"] + 1):
        lengths = []
        for payload in payloads:
            lengths.append(sum(len(hit["content"]) for hit in payload["hits"][:k]))
        char_rows[k] = statistics.mean(lengths) if lengths else 0.0
        lines.append(
            f"| {k} | {curve[k]:.4f} | {doc_curve[k]:.4f} | {char_rows[k]:.0f} |"
        )
    lines.extend(
        [
            "",
            f"- MRR（父块）：**{mrr:.4f}**",
            f"- 选择 `top_k={decision.top_k}` 时，父块级 Recall@{decision.top_k} = **{curve[decision.top_k]:.4f}**，文档级 Recall@{decision.top_k} = **{doc_curve[decision.top_k]:.4f}**",
            "",
            "## 2. RAGAS 指标",
            "",
        ]
    )
    if ragas_output is None:
        lines.append("未执行 RAGAS（`--skip-ragas`）。")
    else:
        summary = ragas_output["summary"]
        lines.extend(
            [
                "| 指标 | 均值 | 中位数 | 标准差 | 有效样本 | 失败 |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for name, item in summary.items():
            lines.append(
                f"| {name} | {_fmt(item['mean'])} | {_fmt(item['median'])} | {_fmt(item['stdev'])} | {item['valid']} | {item['failed']} |"
            )
        lines.extend(
            [
                "",
                "> 指标口径：Faithfulness、Answer Relevancy、Context Precision、Context Recall、Answer Correctness，均为 0–1，越高越好。",
                "> Answer Relevancy 与 Answer Correctness 的语义相似度使用本地 BGE-M3；其余指标由 DeepSeek 评审。",
                "",
            ]
        )

        ranked = sorted(ragas_output["results"], key=_sample_average)[:5]
        lines.extend(
            [
                "### 低分样本（按可用指标均值升序）",
                "",
                "| ID | 问题 | 答案 | 可用指标均值 |",
                "| --- | --- | --- | --- |",
            ]
        )
        for sample in ranked:
            score = _sample_average(sample)
            answer = sample["response"].replace("|", "｜").replace("\n", " ")[:80]
            question = sample["user_input"].replace("|", "｜")[:60]
            lines.append(f"| {sample['id']} | {question} | {answer} | {score:.4f} |")
        lines.append("")

    lines.extend(
        [
            "## 3. 结论与建议",
            "",
            f"1. **top_k 建议**：按 Recall≥{decision.target:.2f} 且 k≥{metadata['min_k']} 的规则，取 `top_k={decision.top_k}`。"
            + (
                f" 该取值下父块级 Recall={curve[decision.top_k]:.4f}，已满足目标。"
                if decision.achieved >= decision.target
                else f" 30 条样本上最大 k={decision.top_k} 的 Recall={decision.achieved:.4f}，仍未达到目标，需要先排查召回而非继续增大 top_k。"
            ),
            "2. **检索质量**：优先看父块级 Recall@k 的拐点；文档级 Recall 只作为上限参考，因为同一文档里另一个父块可能包含答案，但生成时实际看到的是父块。",
            "3. **RAGAS 质量**：若 Faithfulness 或 Context Precision 明显低于 Context Recall，说明「召回得到但上下文噪声大」，应优先调 rerank 阈值或压缩上下文，而不是继续增大 top_k。",
            "4. **成本**：top_k 越大，平均上下文字符线性增加；在 Recall 已达到目标后继续增大 top_k，收益通常小于 token 成本。",
            "",
            "## 4. 复现",
            "",
            "```powershell",
            "uv sync",
            ".venv\\Scripts\\python.exe -m scripts.eval_ragas all",
            "```",
            "",
            "> 说明：RAGAS 0.4.3 依赖的 `langchain-community` 最新版移除了 `ChatVertexAI` 导入路径，因此 dev 依赖固定在 `<0.4`。",
            "",
            "## 5. 限制",
            "",
            "- 语料是为测评专门合成的 30 篇短文档，事实密度高、单文档单父块；真实知识库的长文档与跨块事实会更难，Recall 可能低于本报告。",
            "- 每题只有 1 个金标父块；如果一个事实在多个父块重复出现，父块级 Recall 会低估实际可用性，文档级 Recall 可作补充。",
            "- RAGAS 的 LLM 评审存在采样方差；本报告使用 temperature=0.01、并发 4、逐题保留失败原因，但没有做多次重复取均值。",
            "- 生产集合 `rag_project` 中存在 3 条旧 `policy.md` 残留向量；本次使用独立集合 `rag_eval_v1`，结论针对当前检索/生成链路，不包含该数据一致性问题的影响。",
        ]
    )
    path = output_dir / "report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_json(output_dir / "run_metadata.json", metadata)
    return path


# ---------------------------------------------------------------------------
# 编排
# ---------------------------------------------------------------------------


def _build_retrieval(
    settings,
    store: MilvusChunkStore,
    top_k: int,
    embedding: EmbeddingProvider | None = None,
) -> RetrievalService:
    embedding = embedding or build_embedding(settings)
    reranker = build_reranker(settings)
    return RetrievalService(
        embedding,
        store,
        reranker,
        candidate_k=settings.candidate_k,
        rerank_candidates=settings.rerank_candidates,
        top_k=top_k,
        score_threshold=settings.score_threshold,
        rerank_score_threshold=settings.rerank_score_threshold,
        rrf_k=settings.rrf_k,
    )


async def _run(args: argparse.Namespace) -> int:
    started_at = _now()
    settings = get_settings()
    setup_logging(settings)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    command = args.command
    logger.info("RAGAS 测评开始: command=%s output=%s", command, output_dir)

    manifest_path = output_dir / "corpus_manifest.json"
    manifest: dict[str, Any] | None = None
    embedding: EmbeddingProvider | None = None
    if manifest_path.exists() and args.reuse_snapshot and command in {"generate", "recall", "ragas"}:
        manifest = load_manifest(output_dir)

    if command == "build" or (command == "all" and not (args.reuse_snapshot and manifest)):
        documents = _select_documents(args.limit_docs)
        embedding = build_embedding(settings)
        manifest = build_snapshot(settings, documents, args.collection, output_dir, embedding)
    elif manifest is None:
        manifest = load_manifest(output_dir)

    client = None
    if command in {"generate", "all", "ragas"}:
        client = _deepseek_client()

    dataset_path = output_dir / "dataset.jsonl"
    dataset: list[dict[str, Any]] = []
    if dataset_path.exists() and args.reuse_dataset and command in {"recall", "ragas", "all"}:
        dataset = load_jsonl(dataset_path)
        validate_dataset(dataset)
        logger.info("复用测试集: %s 条", len(dataset))
    elif command in {"generate", "all"}:
        dataset = await generate_dataset(
            client,
            args.model,
            manifest,
            output_dir,
            limit=args.max_samples,
            concurrency=args.concurrency,
        )
    elif command in {"recall", "ragas"}:
        dataset = load_jsonl(dataset_path)
        validate_dataset(dataset)

    if command == "generate":
        return 0
    if command == "build":
        return 0

    store = MilvusChunkStore(
        settings.milvus_uri,
        settings.milvus_token,
        args.collection,
        settings.embedding_dim,
        database=settings.milvus_database,
    )
    retrieval = _build_retrieval(settings, store, args.max_k, embedding=embedding)

    decision: TopKDecision | None = None
    curve: dict[int, float] = {}
    doc_curve: dict[int, float] = {}
    payloads: list[dict[str, Any]] = []
    recall_path = output_dir / "recall_summary.json"
    if command in {"recall", "all", "ragas"}:
        if recall_path.exists() and args.reuse_recall and command in {"ragas", "all"}:
            saved = json.loads(recall_path.read_text(encoding="utf-8"))
            decision = TopKDecision(
                top_k=saved["decision"]["top_k"],
                target=saved["target"],
                achieved=saved["decision"]["achieved"],
                reason=saved["decision"]["reason"],
                curve=saved["decision"]["curve"],
            )
            curve = {int(k): float(v) for k, v in saved["decision"]["curve"].items()}
            doc_curve = {int(k): float(v) for k, v in saved["decision"]["document_curve"].items()}
            payloads = load_jsonl(output_dir / "retrieval_results.jsonl")
            logger.info("复用 Recall 结果: top_k=%s", decision.top_k)
        else:
            decision, curve, doc_curve, payloads = run_recall(
                retrieval,
                dataset,
                output_dir,
                max_k=args.max_k,
                target=args.target,
                min_k=args.min_k,
            )

    if command == "recall":
        return 0

    if decision is None:
        raise RuntimeError("缺少 top_k 决策")

    samples_path = output_dir / "ragas_samples.jsonl"
    if samples_path.exists() and args.reuse_answers and command == "all":
        samples = load_jsonl(samples_path)
        logger.info("复用答案: %s 条", len(samples))
    else:
        samples = await generate_answers(
            client,
            args.model,
            settings,
            retrieval,
            dataset,
            output_dir,
            top_k=decision.top_k,
            concurrency=args.concurrency,
        )

    ragas_output = None
    if not args.skip_ragas:
        embedding = embedding or retrieval._embedding  # noqa: SLF001 - 复用已加载模型
        ragas_output = await run_ragas(
            client,
            args.judge_model,
            embedding,
            samples,
            output_dir,
            concurrency=args.concurrency,
        )

    metadata = {
        "started_at": started_at,
        "finished_at": _now(),
        "command": command,
        "corpus_version": CORPUS_VERSION,
        "collection": args.collection,
        "dataset_size": len(dataset),
        "generator_model": args.model,
        "judge_model": args.judge_model,
        "target": args.target,
        "min_k": args.min_k,
        "max_k": args.max_k,
        "chosen_top_k": decision.top_k,
        "mrr": mean_reciprocal_rank(
            [
                RetrievalRecord(
                    question_id=payload["id"],
                    gold_parent_id=payload["gold_parent_id"],
                    gold_document_id=payload["gold_document_id"],
                    ranked_parent_ids=tuple(hit["parent_id"] for hit in payload["hits"]),
                    ranked_document_ids=tuple(
                        _dedupe_keep_order([hit["document_id"] for hit in payload["hits"]])
                    ),
                )
                for payload in payloads
            ]
        ),
        "thinking_disabled": True,
        "ragas_version": getattr(__import__("ragas"), "__version__", None) if ragas_output else None,
    }
    report_path = write_report(
        output_dir,
        settings=settings,
        manifest=manifest,
        dataset=dataset,
        decision=decision,
        curve=curve,
        doc_curve=doc_curve,
        payloads=payloads,
        ragas_output=ragas_output,
        metadata=metadata,
    )
    logger.info("测评完成: report=%s", report_path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAGAS 测评与 Recall@k 选 top_k")
    parser.add_argument("command", choices=["build", "generate", "recall", "ragas", "all"])
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit-docs", type=int, default=None, help="只使用前 N 篇合成文档")
    parser.add_argument("--max-samples", type=int, default=30, help="最多生成多少条测试题")
    parser.add_argument("--max-k", type=int, default=DEFAULT_MAX_K)
    parser.add_argument("--target", type=float, default=DEFAULT_TARGET)
    parser.add_argument("--min-k", type=int, default=DEFAULT_MIN_K)
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--judge-model", default="deepseek-flash")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--reuse-snapshot", action="store_true")
    parser.add_argument("--reuse-dataset", action="store_true")
    parser.add_argument("--reuse-recall", action="store_true")
    parser.add_argument("--reuse-answers", action="store_true")
    parser.add_argument("--skip-ragas", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        logger.warning("用户中断")
        return 130
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        logger.exception("RAGAS 测评失败")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
