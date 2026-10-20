import { ref } from "vue";

const user = ref(null);
const loading = ref(true);

async function request(url, options = {}) {
  const response = await fetch(url, {
    credentials: "same-origin",
    ...options,
    headers: {
      ...(options.body ? { "Content-Type": "application/json" } : {}),
      ...(options.headers || {}),
    },
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(payload.message || "请求失败");
    error.code = payload.code;
    error.status = response.status;
    throw error;
  }
  return payload.data;
}

export function useAuth() {
  async function fetchMe() {
    try {
      user.value = await request("/api/auth/me");
    } catch {
      user.value = null;
    } finally {
      loading.value = false;
    }
    return user.value;
  }

  async function login(username, password) {
    user.value = await request("/api/auth/login", {
      method: "POST",
      body: JSON.stringify({ username, password }),
    });
    return user.value;
  }

  async function register(payload) {
    user.value = await request("/api/auth/register", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    return user.value;
  }

  async function logout() {
    try {
      await request("/api/auth/logout", { method: "POST" });
    } finally {
      user.value = null;
    }
  }

  return { user, loading, fetchMe, login, register, logout, request };
}
