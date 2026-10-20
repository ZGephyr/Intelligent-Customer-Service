# RAG 电商智能客服 MVP

基于店铺自有知识库的中文智能客服：上传商品表、政策 FAQ、PDF/Word 资料后，按检索结果回答，并给出引用来源；资料不足时明确兜底并建议转人工。

## 架构

```
浏览器 (Vue 3 + Vite)
   │ HTTP + WebSocket
FastAPI (api/)
   │
services/   问答编排、导入、检索、重排、改写、提示词
   │
infra/      LLM、Embedding、Reranker、Milvus、MySQL、Redis、解析器
```

- **检索链路**：问题改写 → bge-m3 同时产出稠密 + 稀疏向量 → Milvus 双路各召回 20 条 → RRF 融合 → 按父块去重 → 子块重排（取 10 条）→ 重排分阈值过滤 → Top-5 父块上下文 → 千问流式生成。
- **父子分块**：父块 1200 字符（上下文），子块 300 字符（向量化与引用），都用递归中文分割器（段落 → 句号/问号/感叹号/分号 → 逗号/顿号）。
- **兜底**：无召回或最高分低于阈值时不调用大模型，直接返回兜底话术，避免编造。
- **会话**：MySQL 持久化全部消息，Redis 缓存最近 20 条（故障自动降级）。
- **引用**：回答中的 `[1][2]` 对应命中的子块，前端折叠展示。

## 目录

```
src/rag_service/
├── main.py            # 应用装配、异常处理、静态托管
├── core/              # 配置、日志、异常、领域类型
├── api/               # HTTP 路由、WebSocket、鉴权
├── services/          # 业务编排
├── infra/             # 外部依赖适配
└── static/            # Vite 构建产物（不提交）
web/                   # Vue 3 前端源码
tests/                 # pytest 单元测试与集成测试
docs/superpowers/      # 方案设计与实施计划
```

## 环境要求

- Python ≥ 3.12、[uv](https://docs.astral.sh/uv/)
- Node ≥ 18（构建前端）
- MySQL 8（`agentic_rag` 库不存在时会自动创建）
- Redis（当前实例为旧版本，客户端固定 RESP2）
- Milvus ≥ 2.4（`192.168.34.97:19530`，库 `rag_hzg`，集合 `rag_project` 自动创建）
- Embedding：本地 `bge-m3`（FlagEmbedding，1024 维稠密 + 稀疏词权重，混合检索）
- 本地模型：`bge-m3`（向量化）、`bge-reranker-v2-m3`（重排）

## 配置

复制 `.env.example` 为 `.env` 并填写。敏感项只放 `.env`：

| 变量 | 说明 |
| --- | --- |
| `DASHSCOPE_API_KEY` | 千问兼容接口 key（也接受 `LLM_API_KEY`） |
| `MYSQL_URL`、`REDIS_URL` | 连接串 |
| `ADMIN_TOKEN` | 管理令牌（`ADMIN_AUTH_ENABLED=true` 时生效） |
| `ADMIN_AUTH_ENABLED` | 默认 false；为 true 时管理接口要求 `X-Admin-Token` |
| `EMBEDDING_MODEL_PATH`、`RERANK_MODEL_PATH` | 本地模型目录 |
| `EMBEDDING_DIM`、`EMBEDDING_BATCH_SIZE` | 默认 1024 / 32 |
| `CANDIDATE_K`、`RRF_K` | 每路召回 20 条、RRF k=60 |
| `RERANK_CANDIDATES`、`RERANK_SCORE_THRESHOLD` | 重排 10 条、相关度阈值 0.3 |

其余项见 `.env.example`，均有默认值。

## 启动

```bash
uv sync --all-groups            # 安装依赖（含 sentence-transformers）
uv run uvicorn rag_service.main:create_app --factory --host 127.0.0.1 --port 8000
```

首次启动会：建库建表 → 创建 / 校验 Milvus collection → 加载 bge-m3（CPU 约 10 秒）。

> 从旧版本（单层分块）升级到父子分块时，先执行一次清空重建：
>
> ```bash
> uv run python scripts/reset_kb.py
> ```
>
> 之后重新上传文档。MySQL 会重建 `kb_documents` / `kb_chunks`，Milvus 集合会按新 schema 重建。

前端：

```bash
cd web
npm install
npm run build                   # 产物输出到 src/rag_service/static/
```

打开 <http://127.0.0.1:8000>。

知识库管理页：<http://127.0.0.1:8000/#/admin>，可选择 / 拖拽文件上传、查看解析状态、删除文档、重建索引。管理接口默认不校验令牌（`ADMIN_AUTH_ENABLED=false`），需要时改成 `true` 并配合 `X-Admin-Token`。

## 管理接口

```bash
# 上传
curl -X POST http://127.0.0.1:8000/api/admin/documents \
  -H "X-Admin-Token: dev-admin-token" \
  -F "file=@policy.md"

# 列表 / 详情 / 删除 / 重建
curl -H "X-Admin-Token: dev-admin-token" "http://127.0.0.1:8000/api/admin/documents?page=1&size=20"
curl -H "X-Admin-Token: dev-admin-token" http://127.0.0.1:8000/api/admin/documents/{id}
curl -X DELETE -H "X-Admin-Token: dev-admin-token" http://127.0.0.1:8000/api/admin/documents/{id}
curl -X POST -H "X-Admin-Token: dev-admin-token" http://127.0.0.1:8000/api/admin/reindex

# 前端用：支持的格式与大小上限
curl http://127.0.0.1:8000/api/admin/config
```

## WebSocket 协议

连接 `ws://127.0.0.1:8000/ws/chat?session_id=可选`。

```json
→ {"type":"message","content":"七天无理由退货谁承担运费？"}
← {"type":"ready","session_id":"..."}
← {"type":"meta","session_id":"...","user_message_id":"...","assistant_message_id":"..."}
← {"type":"sources","sources":[{"index":1,"source":"policy.md","snippet":"...","score":0.82}]}
← {"type":"delta","text":"七天"}
← {"type":"done","status":"ok","latency_ms":1830}
```

错误帧：`{"type":"error","code":"CHAT_BUSY|CHAT_INVALID_MESSAGE|LLM_ERROR","message":"..."}`。

## 测试

```bash
uv run python -m pytest tests/ -v
```

- 单元测试全部注入 fake，不依赖 MySQL/Redis/Milvus/模型。
- 集成测试标记 `integration`，需 `RUN_INTEGRATION=1` 且外部依赖可用。

## 常见问题

| 现象 | 处理 |
| --- | --- |
| 启动报 Milvus 维度不匹配 | 集合维度与 `EMBEDDING_DIM` 不一致，删除集合或调用 `/api/admin/reindex` |
| 上传后状态 failed | 看 `error_message`；扫描版 PDF 无文本需先 OCR |
| 首次请求很慢 | bge-m3 / 重排模型在 CPU 首次加载，属正常 |
| `/health` 中 redis 报 `unknown command 'HELLO'` | 服务端为旧版 Redis，客户端需 `protocol=2`（已内置） |
| 回答总是兜底 | 检查召回分数与 `SCORE_THRESHOLD`，或确认文档已 ready |
