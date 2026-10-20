import { reactive, ref } from "vue";

const STORAGE_PREFIX = "zhidian_messages_";

export function useChatSocket(getSessionId) {
  const messages = ref([]);
  const status = ref("idle");
  const errorText = ref("");

  let socket = null;
  let currentAssistant = null;
  let retries = 0;
  let manual = false;

  function buildUrl() {
    const protocol = location.protocol === "https:" ? "wss" : "ws";
    const sessionId = getSessionId();
    const query = sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : "";
    return `${protocol}://${location.host}/ws/chat${query}`;
  }

  function scheduleReconnect() {
    if (retries >= 5) {
      errorText.value = "连接已断开，请刷新页面重试";
      status.value = "error";
      return;
    }
    const delay = 1000 * 2 ** retries;
    retries += 1;
    setTimeout(connect, delay);
  }

  async function loadHistory(sessionId) {
    if (!sessionId) {
      messages.value = [];
      return;
    }
    try {
      const response = await fetch(`/api/sessions/${sessionId}/messages`, {
        credentials: "same-origin",
      });
      if (!response.ok) return;
      const payload = await response.json();
      // 切换会话后旧请求可能才返回，丢弃避免串台
      if (getSessionId() !== sessionId) return;
      const items = payload?.data?.items ?? [];
      const mapped = items.map((item) => ({
        id: item.id,
        seq: item.seq ?? null,
        role: item.role,
        content: item.content,
        sources: item.sources ?? [],
        status: item.status,
      }));
      const ordered = mapped.every((item) => typeof item.seq === "number")
        ? mapped.slice().sort((a, b) => a.seq - b.seq)
        : mapped;
      messages.value = ordered;
    } catch {
      /* 历史拉取失败不阻断对话 */
    }
  }

  function handle(frame) {
    if (frame.type === "ready") {
      return;
    }
    if (frame.type === "meta") {
      if (currentAssistant) {
        currentAssistant.id = frame.assistant_message_id;
        currentAssistant.intent = frame.intent ?? "";
      }
      return;
    }
    if (frame.type === "sources") {
      if (currentAssistant) currentAssistant.sources = frame.sources ?? [];
      return;
    }
    if (frame.type === "delta") {
      if (currentAssistant) {
        currentAssistant.content += frame.text ?? "";
        status.value = "streaming";
      }
      return;
    }
    if (frame.type === "done") {
      if (currentAssistant) currentAssistant.status = frame.status;
      status.value = "idle";
      currentAssistant = null;
      localStorage.removeItem(`${STORAGE_PREFIX}${getSessionId()}`);
      return;
    }
    if (frame.type === "error") {
      errorText.value = frame.message || "服务异常，请重试";
      status.value = "error";
      currentAssistant = null;
    }
  }

  function connect() {
    status.value = "connecting";
    errorText.value = "";
    socket = new WebSocket(buildUrl());
    socket.onopen = () => {
      retries = 0;
      status.value = "idle";
    };
    socket.onmessage = (event) => {
      try {
        handle(JSON.parse(event.data));
      } catch {
        errorText.value = "收到无法解析的消息";
      }
    };
    socket.onclose = () => {
      if (!manual) scheduleReconnect();
    };
    socket.onerror = () => {
      errorText.value = "连接出错，正在重试";
    };
  }

  function disconnect() {
    manual = true;
    if (socket) socket.close();
    socket = null;
    currentAssistant = null;
  }

  function switchSession(sessionId) {
    disconnect();
    errorText.value = "";
    status.value = "idle";
    messages.value = [];
    loadHistory(sessionId).then(() => {
      manual = false;
      connect();
    });
  }

  function send(text) {
    const content = text.trim();
    if (!content) return false;
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      errorText.value = "连接尚未就绪，请稍后再试";
      return false;
    }
    messages.value.push({
      id: `u-${Date.now()}`,
      role: "user",
      content,
      sources: [],
      status: "ok",
    });
    currentAssistant = reactive({
      id: `a-${Date.now()}`,
      role: "assistant",
      content: "",
      sources: [],
      status: "streaming",
    });
    messages.value.push(currentAssistant);
    socket.send(JSON.stringify({ type: "message", content }));
    return true;
  }

  return {
    messages,
    status,
    errorText,
    connect,
    disconnect,
    switchSession,
    loadHistory,
    send,
  };
}
