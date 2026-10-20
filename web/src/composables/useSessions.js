import { ref } from "vue";

import { useAuth } from "./useAuth";

const sessions = ref([]);
const currentId = ref("");
const STORAGE_KEY = "zhidian_last_session";

export function useSessions() {
  const { request } = useAuth();

  async function load() {
    const data = await request("/api/sessions?page=1&size=50");
    sessions.value = data.items;
    return data.items;
  }

  async function create() {
    const data = await request("/api/sessions", { method: "POST" });
    await load();
    open(data.session_id);
    return data.session_id;
  }

  function open(sessionId) {
    currentId.value = sessionId;
    localStorage.setItem(STORAGE_KEY, sessionId);
  }

  function restore() {
    const saved = localStorage.getItem(STORAGE_KEY) || "";
    const exists = sessions.value.some((item) => item.id === saved);
    return exists ? saved : sessions.value[0]?.id ?? "";
  }

  function reset() {
    sessions.value = [];
    currentId.value = "";
    localStorage.removeItem(STORAGE_KEY);
  }

  return { sessions, currentId, load, create, open, restore, reset };
}
