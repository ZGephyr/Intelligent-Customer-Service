# 知店智能客服

> 只懂店里的事 —— 基于店铺自有知识库的中文智能客服。

把商品表、政策文档、PDF/Word 资料导入知识库后，顾客在网页里提问，系统**只依据这些资料**作答并给出引用；资料里没有的内容明确说「不知道」并建议转人工，不编造价格、时效与政策。

**技术栈**：Python 3.12 · FastAPI · LangChain · Milvus · MySQL · Redis（可降级） · bge-m3 + bge-reranker-v2-m3 · Vue 3 + Vite · pytest

---

## 目录

- [功能特性](#功能特性)
- [架构与链路](#架构与链路)
- [目录结构](#目录结构)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [账号与权限](#账号与权限)
- [使用指南](#使用指南)
- [接口一览](#接口一览)
- [WebSocket 协议](#websocket-协议)
- [评测与基线](#评测与基线)
- [测试](#测试)
- [常见问题](#常见问题)
- [已知限制与规划](#已知限制与规划)
- [变更记录](#变更记录)

---

## 功能特性

| 模块 | 能力 |
| --- | --- |
| 问答 | 意图三分类（业务 / 闲聊 / 非业务）→ 高频问答捷径 → 改写与 HyDE → 混合召回 + 重排 → 流式作答；引用可展开原文 |
| 知识库 | 支持 `md / txt / csv / xlsx / pdf / docx`，单文件 ≤ 20MB；sha256 去重；父子分块；解析失败标记 `failed` 并给出原因 |
| 高频问答 | 独立管理页；BM25（jieba + 字符二元组）+ softmax 归一化；别名、分类、启用开关、命中统计、CSV/XLSX 批量导入 |
| 账号 | 自助注册（账号 + 密码 + 算术验证码）、登录/登出、角色（普通用户 / 管理员）、7 天登录态 |
| 会话 | 按用户隔离的多会话：左侧列表、新建、切换、历史恢复、删除；旧会话可继续提问 |
| 运维 | `/health` 四项依赖检查、统一错误码、`request_id` 贯穿日志、按天轮转日志、离线评测脚本 |

---

## 架构与链路

```text
浏览器（Vue 3 + Vite 构建产物，由 FastAPI 托管）
   │  HTTP + WebSocket（Cookie 鉴权）
FastAPI（api/）
   │
services/   问答编排、会话、鉴权、导入、检索、重排、改写、提示词
   │
infra/      LLM、Embedding、Reranker、Milvus、MySQL、Redis、文档解析
```

**一次提问的完整链路**

1. **鉴权**：WebSocket 握手校验登录 Cookie 与会话归属（未登录 `4401`，越权 `4404`）。
2. **意图三分类**：`in_scope`（商品/政策/售后/物流）→ 继续；`chitchat`（问候寒暄）→ 固定闲聊话术；`out_of_scope` → 兜底 + 转人工。分类失败降级为 `in_scope`（宁多查不误拦）。
3. **高频问答**：用原始问题与 FAQ 的**问题文本**做 BM25 + softmax，概率 ≥ 0.8 且查询覆盖率 ≥ 0.30 才命中，直接返回人工答案，**不调用生成模型**。
4. **改写**：一次调用产出「改写问题 + 关键词 + 假设答案」；缩写按 `ABBREVIATIONS` 强制展开。
5. **检索**：bge-m3 产出稠密 + 稀疏向量 → Milvus 双路各召回 20 条 → RRF 融合 → 按父块去重 → 子块重排（取 10）→ 重排分阈值 0.3 → Top-5 父块。
6. **HyDE**：仅当首轮召回为空时，用假设答案再检索一次；仍为空则兜底。
7. **生成**：系统提示约束「只用资料、标注 `[n]`、不足就说不知道」，千问兼容接口流式返回。
8. **落库**：消息、引用、状态、耗时写入 MySQL；Redis 缓存最近 20 条（不可用时自动降级为只读 MySQL）。

**分块策略**：父块 1200 字符（喂模型）、子块 300 字符（向量化与引用），递归中文分割（段落 → 句号/问号 → 逗号/顿号）。

---

## 目录结构

```text
RAG_project/
├── src/rag_service/
│   ├── main.py                 # 应用装配、异常处理、静态托管
│   ├── core/                   # 配置、日志、异常、领域类型
│   ├── api/                    # auth / sessions / admin / faq / health / WebSocket
│   ├── services/               # 问答编排、鉴权、会话、导入、检索、重排、改写、提示词
│   ├── infra/                  # LLM、Embedding、Reranker、Milvus、MySQL、Redis、security、解析器
│   ├── models/tables.py        # SQLAlchemy 模型
│   └── static/                 # 前端构建产物（不提交）
├── web/                        # Vue 3 + Vite 源码
├── tests/                      # pytest（27 个测试文件）
├── scripts/
│   ├── reset_kb.py             # 重建知识库表与 Milvus 集合，并清空上传目录
│   ├── clear_legacy_sessions.py# 一次性清空旧会话（1.1 升级用）
│   └── eval_offline.py         # 意图 / FAQ 离线评测
├── data/
│   ├── eval/                   # 评测集与基线
│   └── uploads/                # 上传文件（运行时生成）
├── docs/                       # 设计文档与实施计划（本地，未纳入版本管理）
└── logs/                       # 运行日志（运行时生成）
```

---

## 环境要求

| 依赖 | 版本 / 说明 |
| --- | --- |
| Python | ≥ 3.12，包管理用 [uv](https://docs.astral.sh/uv/) |
| Node.js | ≥ 18（仅构建前端；仓库使用 npm + `package-lock.json`） |
| MySQL | 8.x；库不存在时启动自动创建（utf8mb4），表结构由 `ensure_schema` 补齐 |
| Redis | 可选；不可用时历史缓存自动降级，不影响问答 |
| Milvus | ≥ 2.4，远程实例；集合不存在时启动自动创建 |
| Embedding | 本地 `bge-m3`（稠密 1024 维 + 稀疏权重） |
| Reranker | 本地 `bge-reranker-v2-m3`（可开关） |
| LLM | OpenAI 兼容接口，默认千问兼容模式 `qwen3.8-flash` |

---

## 快速开始

```bash
# 1. 安装依赖（含本地模型推理依赖）
uv sync --all-groups

# 2. 配置
cp .env.example .env          # Windows: copy .env.example .env
# 至少填：DASHSCOPE_API_KEY（或 LLM_API_KEY）、MYSQL_URL、REDIS_URL、ADMIN_TOKEN

# 3. 构建前端（产物输出到 src/rag_service/static/）
cd web && npm install && npm run build && cd ..

# 4. 启动（注意 --factory：应用是工厂函数）
uv run uvicorn rag_service.main:create_app --factory --host 127.0.0.1 --port 8000
```

打开 <http://127.0.0.1:8000>：

1. 首次使用先**注册**；需要维护知识库的账号在注册页选择「管理员」。
2. 用管理员进入 `#/admin` 上传资料，或到 `#/faq` 维护高频问答。
3. 回到 `#/chat` 提问，左侧可新建 / 切换会话。

首次启动会依次执行：`setup_logging → ensure_database → ensure_schema → ensure_collection → warmup`（加载 bge-m3 与重排模型，CPU 约 10 秒）。

---

## 配置说明

配置集中在 `src/rag_service/core/config.py`（pydantic-settings），敏感项只放 `.env`。下面只列常用项，完整清单见 `.env.example`。

**模型与生成**

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `LLM_API_KEY` / `DASHSCOPE_API_KEY` | 必填 | 兼容接口 key，二者取其一 |
| `LLM_BASE_URL` | 千问兼容地址 | OpenAI 兼容 base_url |
| `LLM_MODEL` | `qwen3.8-flash` | 模型名 |
| `LLM_TIMEOUT_SECONDS` | 60 | 生成超时 |
| `LLM_TEMPERATURE` | 0.2 | 生成温度 |
| `LLM_ENABLE_THINKING` | false | 关闭模型思考模式；开启后小 token 预算容易返回空内容 |
| `EMBEDDING_MODEL_PATH` / `RERANK_MODEL_PATH` | 本机模型目录 | 换机器必须改 |

**意图与话术**

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `INTENT_ENABLED` | true | 关闭后所有问题按 `in_scope` 处理 |
| `INTENT_TIMEOUT_SECONDS` | 2.5 | 意图识别超时 |
| `INTENT_MAX_TOKENS` | 128 | 意图输出上限（需覆盖模型前导消耗） |
| `CHITCHAT_REPLY` | 见上文 | 闲聊固定话术 |
| `OUT_OF_SCOPE_REPLY` | 「不好意思呀…」 | 非业务兜底文案，后面会拼 `HANDOFF_HINT` |

**高频问答**

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `FAQ_ENABLED` | true | 总开关 |
| `FAQ_SCORE_THRESHOLD` | 0.8 | softmax 概率阈值 |
| `FAQ_MIN_COVERAGE` | 0.30 | 查询覆盖率门槛（命中词 IDF / 全查询词 IDF），0 表示关闭 |
| `FAQ_MARGIN` | 0.0 | top1 与 top2 最小分差，0 表示关闭 |
| `FAQ_SOFTMAX_TEMPERATURE` | 1.0 | 越大分布越平 |
| `FAQ_CACHE_TTL_SECONDS` | 60 | 内存快照 TTL；写操作立即失效 |

**检索与改写**

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `CANDIDATE_K` / `RRF_K` | 20 / 60 | 双路召回条数、RRF 参数 |
| `RERANK_ENABLED` / `RERANK_CANDIDATES` / `RERANK_SCORE_THRESHOLD` | true / 10 / 0.3 | 重排开关、候选数、阈值 |
| `TOP_K` / `SCORE_THRESHOLD` | 3 / 0.35 | 最终上下文条数（RAGAS/Recall@k 测评确定）、召回兜底阈值 |
| `REWRITE_MODE` / `REWRITE_ENABLED` | full / true | full=改写+关键词+假设答案；minimal=仅指代 |
| `HYDE_ENABLED` | true | 仅首轮空召回时触发 |
| `ABBREVIATIONS` | 空 | `降噪=主动降噪,hrv=心率变异性` |

**登录鉴权**

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `JWT_SECRET_PATH` | `data/.jwt_secret` | 首次启动自动生成，重启复用 |
| `AUTH_COOKIE_NAME` | `rag_session` | HttpOnly Cookie 名 |
| `AUTH_TOKEN_TTL_DAYS` | 7 | 登录有效期 |
| `COOKIE_SECURE` | false | HTTPS 部署时置 true |
| `ALLOW_ADMIN_SIGNUP` | true | 是否允许自助注册管理员（见风险提示） |
| `CAPTCHA_TTL_SECONDS` | 300 | 验证码有效期 |
| `AUTH_MAX_ATTEMPTS` / `AUTH_LOCK_SECONDS` | 10 / 60 | 登录失败锁定策略 |
| `BCRYPT_ROUNDS` | 12 | 密码哈希成本（测试里用 4） |

---

## 账号与权限

| 能力 | 未登录 | 普通用户 | 管理员 |
| --- | --- | --- | --- |
| 登录 / 注册 | ✅ | ✅ | ✅ |
| 聊天、会话列表、自己的历史 | ❌ | ✅ | ✅ |
| 高频问答管理（`#/faq`、`/api/admin/faqs`） | ❌ | ❌ 403 | ✅ |
| 知识库管理（`#/admin`、`/api/admin/documents`） | ❌ | ❌ 403 | ✅ |
| 健康检查 `/health` | ✅ | ✅ | ✅ |

- 注册页可选角色；**管理员不设邀请码**（由 `ALLOW_ADMIN_SIGNUP` 控制是否开放该选项）。
- 管理员也只能看到**自己**的会话；会话越权一律 404，不暴露存在性。
- 登录态是 HS256 JWT，放在 HttpOnly Cookie 里；前端不接触 token，WebSocket 握手同源自带。

> ⚠️ **部署风险**：注册页对外开放时，任何能访问到的人都能注册成管理员并清空知识库。仅建议内网/受信环境使用；对外前把 `ALLOW_ADMIN_SIGNUP` 设为 `false`，后续版本再补「管理员后台建号」。

---

## 使用指南

**聊天与多会话**

- 左侧是「我的会话」：显示标题（首条问题前 20 字）、消息数、相对时间。
- 「＋ 新建会话」只新增一条，旧会话仍在列表里；点旧会话即可加载历史并继续提问。
- 生成中切换会话会提示确认，避免把回答写进另一个会话。
- 鼠标移到会话行会显示「×」，点击并确认即可删除（会话与消息一起删除，不可恢复）；删除当前会话会自动切到下一条，删空后自动新建。
- 刷新页面保持登录，并恢复上次打开的会话。

**高频问答**

- 页面 `#/faq`：搜索、分类筛选、启用/停用、增删改、批量导入。
- 导入表头：`问题,答案,分类,别名`（别名一行一条，参与匹配）。
- 命中后答案原样返回，不走生成模型；命中次数与最近命中时间可在列表查看。

**知识库**

- 页面 `#/admin`：拖拽/选择上传，查看解析状态、分段数、失败原因；支持删除与全量重建索引。
- 重复文件按 sha256 去重，直接返回已有文档。

**升级到 1.1（旧会话清空）**

旧会话没有 `user_id`，登录后不可见，按约定清空一次（**不可恢复**）：

```bash
uv run python -m scripts.clear_legacy_sessions --yes
```

**重建知识库**（换 embedding 模型或维度变化时）：

```bash
uv run python scripts/reset_kb.py
```

该脚本会删除 `kb_chunks` / `kb_documents` 并重建、按当前 schema 重建 Milvus 集合、清空 `data/uploads/`，之后需要重新上传文档。

---

## 接口一览

统一响应：成功 `{"code":0,"data":...,"request_id":"..."}`；失败 `{"code":"<错误码>","message":"...","request_id":"..."}`。

**鉴权（公开）**

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/auth/captcha` | 取验证码：`{captcha_id, question, allow_admin_signup}` |
| POST | `/api/auth/register` | `{username, password, captcha_id, captcha_answer, role}`，成功即登录 |
| POST | `/api/auth/login` | `{username, password}` |
| POST | `/api/auth/logout` | 清除 Cookie |
| GET | `/api/auth/me` | 当前用户；未登录 401 |

**会话（需登录）**

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/sessions?page=1&size=20` | 我的会话列表（含条数与最近消息时间） |
| POST | `/api/sessions` | 新建会话 |
| GET | /api/sessions/{id}/messages?limit=50 | 历史消息；非本人 404 |
| DELETE | /api/sessions/{id} | 删除会话及其消息；非本人 404 |

**知识库管理（需管理员）**

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/admin/documents` | 上传（multipart 字段 `file`），202 返回 processing |
| GET | `/api/admin/documents?page=1&size=20` | 分页列表 |
| GET | `/api/admin/documents/{id}` | 详情 |
| DELETE | `/api/admin/documents/{id}` | 删除文档与向量 |
| POST | `/api/admin/reindex` | 全量重建索引，202 |
| GET | `/api/admin/config` | 支持格式与大小上限（前端用） |

**高频问答管理（需管理员）**

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/admin/faqs?page&size&keyword&category&enabled` | 分页列表 |
| POST | `/api/admin/faqs` | 新增（问法重复 400 `FAQ_DUPLICATE`） |
| PUT | `/api/admin/faqs/{id}` | 编辑（问题/答案不能改成空） |
| DELETE | `/api/admin/faqs/{id}` | 删除 |
| POST | `/api/admin/faqs/import` | CSV/XLSX 导入，返回新增/更新/跳过 |
| GET | `/api/admin/faqs/stats` | 总数、启用数、累计命中、Top 命中 |

> 兼容：`ADMIN_AUTH_ENABLED=true` 时，管理接口仍接受 `X-Admin-Token` 请求头（便于脚本调用）。

**其他**：`GET /health`（公开，任一依赖异常返回 503 + `degraded`）。

---

## WebSocket 协议

连接 `ws://127.0.0.1:8000/ws/chat?session_id=可选`（同源 Cookie 鉴权；未登录 `4401`，会话越权 `4404`）。

```json
→ {"type":"message","content":"七天无理由退货谁承担运费？"}
← {"type":"ready","session_id":"..."}
← {"type":"meta","session_id":"...","intent":"in_scope","user_message_id":"...","assistant_message_id":"..."}
← {"type":"sources","sources":[{"index":1,"type":"rag","source":"policy.md","snippet":"...","score":0.82}]}
← {"type":"delta","text":"七天"}
← {"type":"done","status":"ok","latency_ms":1830}
```

| `done.status` | 含义 |
| --- | --- |
| `ok` | 资料回答（带引用） |
| `faq` | 高频问答命中，原样返回 |
| `chitchat` | 闲聊固定话术 |
| `fallback` | 资料里没有找到，兜底 + 转人工 |
| `out_of_scope` | 非业务问题，兜底 + 转人工 |

错误帧：`{"type":"error","code":"CHAT_BUSY|CHAT_INVALID_MESSAGE|LLM_ERROR","message":"..."}`。

---

## 评测与基线

```bash
# FAQ 命中率（纯本地，不需要外部服务）
uv run python -m scripts.eval_offline faq

# 意图准确率（需要 .env 中 LLM 可用）
uv run python -m scripts.eval_offline intent --live
```

评测集在 `data/eval/`（40 条 FAQ 用例、30 条意图用例、20 条改写用例）。

**当前基线（2026-10-20）**：FAQ 召回 14/20 = **70%**，误命中 1/20，样本精确率 93.3%；别名覆盖的问法 100% 命中。纯 BM25 对「字面不重叠、又没配别名」的同义问法有天花板，结论与阈值扫描见 `data/eval/baseline.md`。

---

## 测试

```bash
uv run python -m pytest tests/ -v
```

- 当前：**299 passed, 1 skipped**（跳过的是需 `RUN_INTEGRATION=1` 的集成用例）。
- 单元测试全部注入 fake，不依赖 MySQL / Redis / Milvus / 真实模型。
- 覆盖：配置与迁移、解析与切分、检索与重排、意图、FAQ、改写、会话编排、鉴权、会话隔离、WebSocket 鉴权、评测脚本。

### RAGAS 测评

RAGAS 0.4.3 已加入 `dev` 依赖（同时固定 `langchain-community>=0.3,<0.4`，避免新版移除 `ChatVertexAI` 导入路径）。测评脚本会建独立 Milvus 集合、生成合成题集、用 Recall@k 选 `top_k`，再用 DeepSeek（关闭深度思考）做 RAGAS 打分：

```bash
uv sync
.venv\Scripts\python.exe -m scripts.eval_ragas all
```

最近一次结果：Recall@1..10 均为 1.0，按 `Recall≥0.90 且 k≥3` 取 `top_k=3`；RAGAS Faithfulness 0.9494、Answer Relevancy 0.7908、Context Precision 0.9944、Context Recall 1.0000、Answer Correctness 0.7231。完整报告在本地生成：`data/eval/ragas/report.md`（`data/` 不入库）。

---

## 常见问题

| 现象 | 处理 |
| --- | --- |
| 启动报 `Access denied for user 'root'`（1045） | `.env` 的 `MYSQL_URL` 密码与本机 MySQL 不一致；改密码或用独立账号 |
| `/health` 里 redis 报 `unknown command 'HELLO'` | 旧版 Redis；客户端已固定 `protocol=2`，仍报错则确认 `REDIS_URL` 指向的实例 |
| 启动报 Milvus 集合维度不匹配 | `EMBEDDING_DIM` 与集合不一致：执行 `scripts/reset_kb.py` 重建 |
| 首次提问很慢 | bge-m3 / 重排模型冷启动，属正常；启动时已预热 |
| 回答总是兜底 | 看日志 `retrieve` 段的命中数与最高分；确认文档 `ready`、`SCORE_THRESHOLD` 是否过高 |
| 引用里全是 0.5 分 | 重排模型目录缺 `modules.json` 时旧代码会静默加载随机模型；v1.6 起自动改用 transformers 加载，加载失败会在启动报错 |
| 意图总是 `unknown` | 看日志是否有「输出无法解析 / 调用失败」；确认 `INTENT_MAX_TOKENS` 足够、`LLM_ENABLE_THINKING=false` |
| FAQ 明显该命中却没命中 | 看日志里的 `覆盖率=`：低于 `FAQ_MIN_COVERAGE` 就会被拦（字面不重叠的同义问法属此类），补「别名」最有效 |
| 上传后状态 `failed` | 看 `error_message`；扫描版 PDF 无文本，需先 OCR |
| 注册页没有「管理员」选项 | `ALLOW_ADMIN_SIGNUP=false` |
| 端口 8000 被占用 | 换 `--port 8001`，同时把 `WS_ALLOWED_ORIGINS` 改成对应地址 |

---

## 已知限制与规划

**当前限制**

- 注册页开放管理员注册；无找回密码、无用户管理页、无角色变更。
- FAQ 匹配是纯 BM25，同义问法依赖别名维护；语料规模明显变化后阈值需重标定。
- 登录失败限流与 FAQ 缓存是**进程内**实现，多 worker 部署时退化为按进程生效。
- 会话清空脚本不可恢复，执行前请确认。
- `docs/`（设计文档与计划）未纳入版本管理。
- 首字延迟指标（原 P95 ≤5s）暂时作废：当前超时统一 60s，先保证流程正确，延迟优化后再恢复。

**后续方向**

- 1.1.x：管理员邀请码 / 关闭自助注册、用户管理页、找回密码。
- 1.2：FAQ 加一道 embedding 复核，提升同义问法召回；多 worker 下的限流与缓存一致性。

---

## 变更记录

| 版本 | 日期 | 内容 |
| --- | --- | --- |
| 1.1.2 | 2026-10-20 | FAQ 增加查询覆盖率门槛，修复单候选弱匹配被判 ~1.0 相关的误答；召回 70%→55%、误答 1→0 |
| 1.1.1 | 2026-10-20 | 会话删除；确认弹窗改为自绘（不再用浏览器 confirm/alert）；消息按会话内 seq 排序，修复同一秒多条消息乱序 |
| 1.1 | 2026-10-20 | 登录鉴权与角色、多会话（按用户隔离、左侧列表与切换）、视觉 C 改版、更名「知店智能客服」 |
| 1.0 | 2026-10-20 | 意图三分类、高频问答（BM25 + softmax）、改写升级（关键词 / 缩写 / HyDE）、离线评测 |
| MVP | 2026-10-19 | 知识库导入、混合召回 + 重排、流式问答与引用、兜底、WebSocket、Vue 前端 |
