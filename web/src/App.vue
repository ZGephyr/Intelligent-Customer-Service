<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";

import AdminPanel from "./components/AdminPanel.vue";
import AppShell from "./components/AppShell.vue";
import AuthPage from "./components/AuthPage.vue";
import ChatView from "./components/ChatView.vue";
import ConfirmDialog from "./components/ConfirmDialog.vue";
import FaqPanel from "./components/FaqPanel.vue";
import { useAuth } from "./composables/useAuth";
import { useDialog } from "./composables/useDialog";
import { useSessions } from "./composables/useSessions";

const { user, loading, fetchMe, logout, request } = useAuth();
const { sessions, currentId, load, create, open, restore, reset } = useSessions();
const { confirm } = useDialog();

const route = ref(window.location.hash || "#/chat");
const notice = ref("");

const isAdmin = computed(() => user.value?.role === "admin");
const page = computed(() =>
  route.value.startsWith("#/faq") ? "faq" : route.value.startsWith("#/admin") ? "admin" : "chat"
);
const authMode = computed(() => (route.value.startsWith("#/register") ? "register" : "login"));
const sessionTitle = computed(
  () => sessions.value.find((item) => item.id === currentId.value)?.title || "新会话"
);

function go(hash) {
  if (window.location.hash === hash) {
    route.value = hash;
    return;
  }
  window.location.hash = hash;
}

function syncRoute() {
  route.value = window.location.hash || "#/chat";
}

function guard() {
  if (!user.value) {
    if (!route.value.startsWith("#/login") && !route.value.startsWith("#/register")) {
      go("#/login");
    }
    return;
  }
  if (route.value.startsWith("#/login") || route.value.startsWith("#/register")) {
    go("#/chat");
    return;
  }
  if ((page.value === "faq" || page.value === "admin") && !isAdmin.value) {
    notice.value = "没有管理权限，已返回对话";
    go("#/chat");
  }
}

async function boot() {
  await load();
  if (!sessions.value.length) {
    await create();
    return;
  }
  const restored = restore();
  if (restored) open(restored);
}

async function refreshSessions() {
  try {
    await load();
  } catch {
    /* 列表刷新失败不影响对话 */
  }
}

async function onNewSession() {
  notice.value = "";
  await create();
  if (page.value !== "chat") go("#/chat");
}

async function onDeleteSession(id) {
  const target = sessions.value.find((item) => item.id === id);
  const label = target?.title || "新会话";
  const confirmed = await confirm({
    title: "删除会话",
    message: `「${label}」及其全部消息会被删除，且不可恢复。`,
    confirmText: "删除",
    danger: true,
  });
  if (!confirmed) return;

  try {
    await request(`/api/sessions/${id}`, { method: "DELETE" });
  } catch (error) {
    notice.value = `删除失败：${error.message || "请稍后重试"}`;
    return;
  }

  const wasCurrent = id === currentId.value;
  await load();
  if (!wasCurrent) return;

  if (sessions.value.length) {
    open(sessions.value[0].id);
  } else {
    await create();
  }
}

function onSelect(id) {
  if (id !== currentId.value) open(id);
  // 在问答/知识库页时，即使点的是当前会话，也要回到对话页
  if (page.value !== "chat") go("#/chat");
}

async function onLogout() {
  await logout();
  reset();
  notice.value = "";
  go("#/login");
}

async function onAuthDone() {
  await boot();
  go("#/chat");
}

onMounted(async () => {
  window.addEventListener("hashchange", syncRoute);
  await fetchMe();
  if (user.value) await boot();
  guard();
});

onBeforeUnmount(() => {
  window.removeEventListener("hashchange", syncRoute);
});

watch([route, user], () => guard());
</script>

<template>
  <ConfirmDialog />

  <div v-if="loading" class="auth-screen">
    <p class="hint">正在加载…</p>
  </div>

  <AuthPage
    v-else-if="!user"
    :mode="authMode"
    :notice="notice"
    @done="onAuthDone"
    @switch="(mode) => go(mode === 'register' ? '#/register' : '#/login')"
  />

  <AppShell
    v-else
    :title="page === 'chat' ? sessionTitle : page === 'faq' ? '高频问答' : '知识库'"
    :subtitle="
      page === 'chat'
        ? '只依据店铺资料作答，可在左侧切换会话'
        : page === 'faq'
          ? '命中后直接返回人工答案，不调用模型'
          : '支持 md / txt / csv / xlsx / pdf / docx，单文件 ≤ 20MB'
    "
    :sessions="sessions"
    :current-id="currentId"
    :user="user"
    :active="page"
    :body-class="page === 'chat' ? 'content-chat' : ''"
    :notice="notice"
    @new-session="onNewSession"
    @select="onSelect"
    @delete-session="onDeleteSession"
    @dismiss-notice="notice = ''"
    @logout="onLogout"
  >
    <ChatView v-if="page === 'chat'" :session-id="currentId" @finished="refreshSessions" />
    <FaqPanel v-else-if="page === 'faq'" />
    <AdminPanel v-else />
  </AppShell>
</template>
