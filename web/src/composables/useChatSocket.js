import { onBeforeUnmount, reactive, ref } from "vue";

const STORAGE_KEY = "rag_session_id";

export function useChatSocket() {
  const messages = ref([]);
  const status = ref("idle");
  const errorText = ref("");
  const sessionId = ref(localStorage.getItem(STORAGE_KEY) || "");

  let socket = null;
  let currentAssistant = null;
  let retries = 0;
  let manualClose = false;

  function buildUrl() {
    const protocol = location.protocol === "https:" ? "wss" : "ws";
    const query = sessionId.value ? `?session_id=${encodeURIComponent(sessionId.value)}` : "";
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

  async function loadHistory() {
    if (!sessionId.value || currentAssistant) return;
    try {
      const response = await fetch(`/api/sessions/${sessionId.value}/messages`);
      const payload = await response.json();
      const items = payload?.data?.items ?? [];
      if (!items.length) return;

      const mapped = items.map((item) => ({
        id: item.id,
        role: item.role,
        content: item.content,
        sources: item.sources ?? [],
        status: item.status,
      }));

      // 合并而不是覆盖：避免把已经开始的流式气泡冲掉
      const seen = new Set(messages.value.map((message) => message.id));
      const missing = mapped.filter((message) => !seen.has(message.id));
      if (missing.length) {
        messages.value = [...missing, ...messages.value];
      }
    } catch {
      /* 历史拉取失败不阻断对话 */
    }
  }

  function handle(frame) {
    if (frame.type === "ready") {
      sessionId.value = frame.session_id;
      localStorage.setItem(STORAGE_KEY, frame.session_id);
      if (!messages.value.length) loadHistory();
      return;
    }
    if (frame.type === "meta") {
      if (currentAssistant) currentAssistant.id = frame.assistant_message_id;
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
      if (!manualClose) scheduleReconnect();
    };
    socket.onerror = () => {
      errorText.value = "连接出错，正在重试";
    };
  }

  function send(text) {
    const content = text.trim();
    if (!content || !socket || socket.readyState !== WebSocket.OPEN) {
      errorText.value = "连接尚未就绪，请稍后再试";
      return false;
    }
    messages.value.push({ id: `u-${Date.now()}`, role: "user", content, sources: [], status: "ok" });
    // 必须是响应式对象：直接 push 普通对象后改它，Vue 不会触发子组件重渲染
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

  function newSession() {
    manualClose = true;
    if (socket) socket.close();
    localStorage.removeItem(STORAGE_KEY);
    sessionId.value = "";
    messages.value = [];
    currentAssistant = null;
    errorText.value = "";
    status.value = "idle";
    setTimeout(() => {
      manualClose = false;
      connect();
    }, 100);
  }

  onBeforeUnmount(() => {
    manualClose = true;
    if (socket) socket.close();
  });

  return { messages, status, errorText, sessionId, connect, send, newSession };
}
