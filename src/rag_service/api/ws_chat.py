"""WebSocket 问答：同步生成器 → 线程 → asyncio 队列 → 前端。"""

from __future__ import annotations

import asyncio
import threading
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from rag_service.core.exceptions import AppError
from rag_service.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


def _pump(generator, queue: asyncio.Queue, cancel: threading.Event, loop: asyncio.AbstractEventLoop) -> None:
    try:
        for event in generator:
            if cancel.is_set():
                break
            loop.call_soon_threadsafe(queue.put_nowait, ("event", event))
    except Exception as exc:  # noqa: BLE001 - 交给主循环转成 error 帧
        loop.call_soon_threadsafe(queue.put_nowait, ("error", exc))
    finally:
        loop.call_soon_threadsafe(queue.put_nowait, ("end", None))


@router.websocket("/ws/chat")
async def chat_socket(websocket: WebSocket) -> None:
    settings = websocket.app.state.settings
    origin = websocket.headers.get("origin")
    if origin and origin not in settings.allowed_origins:
        logger.warning("拒绝未知 Origin 的 WebSocket 连接: %s", origin)
        await websocket.close(code=1008)
        return

    await websocket.accept()
    service = websocket.app.state.services.chat
    session_id = websocket.query_params.get("session_id") or service.create_session()
    audit = uuid.uuid4().hex

    await websocket.send_json({"type": "ready", "session_id": session_id, "request_id": audit})
    logger.info("WebSocket 已连接: session=%s request=%s", session_id, audit)

    task: asyncio.Task | None = None
    busy = False
    cancel = threading.Event()
    loop = asyncio.get_running_loop()

    async def process(content: str) -> None:
        nonlocal cancel, busy
        cancel = threading.Event()
        queue: asyncio.Queue = asyncio.Queue()
        generator = service.stream_answer(session_id, content)
        runner = loop.run_in_executor(None, _pump, generator, queue, cancel, loop)

        try:
            while True:
                kind, payload = await queue.get()
                if kind == "end":
                    break
                if kind == "error":
                    logger.error("问答流异常: session=%s error=%s", session_id, payload)
                    await websocket.send_json(
                        {"type": "error", "code": "LLM_ERROR", "message": "模型响应失败，请重试"}
                    )
                    continue
                # done 发出前就释放忙碌标记，避免客户端立刻追问被误判为 CHAT_BUSY
                if payload.type in ("done", "error"):
                    busy = False
                await websocket.send_json({"type": payload.type, **payload.data})
        finally:
            busy = False
            await runner

    try:
        while True:
            frame = await websocket.receive_json()
            if not isinstance(frame, dict) or frame.get("type") != "message":
                await websocket.send_json(
                    {"type": "error", "code": "CHAT_INVALID_MESSAGE", "message": "消息格式有误"}
                )
                continue

            content = frame.get("content")
            if not isinstance(content, str) or not content.strip():
                await websocket.send_json(
                    {"type": "error", "code": "CHAT_INVALID_MESSAGE", "message": "请输入有效问题"}
                )
                continue

            if busy:
                await websocket.send_json(
                    {"type": "error", "code": "CHAT_BUSY", "message": "上一条回答还在生成中"}
                )
                continue

            busy = True
            task = asyncio.create_task(process(content))
    except WebSocketDisconnect:
        logger.info("WebSocket 断开: session=%s", session_id)
    except AppError as exc:
        await websocket.send_json({"type": "error", "code": exc.code, "message": exc.message})
    finally:
        cancel.set()
        if task is not None and not task.done():
            task.cancel()
