<script setup>
import { computed } from "vue";

const props = defineProps({
  title: { type: String, default: "" },
  subtitle: { type: String, default: "" },
  sessions: { type: Array, default: () => [] },
  currentId: { type: String, default: "" },
  user: { type: Object, default: null },
  active: { type: String, default: "chat" },
  notice: { type: String, default: "" },
  bodyClass: { type: String, default: "" },
});

defineEmits(["new-session", "select", "delete-session", "dismiss-notice", "logout"]);

const isAdmin = computed(() => props.user?.role === "admin");
const initial = computed(() => (props.user?.username || "?").slice(0, 1).toUpperCase());

function relative(value) {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  const diff = Date.now() - date.getTime();
  const minutes = Math.floor(diff / 60000);
  if (minutes < 1) return "刚刚";
  if (minutes < 60) return `${minutes} 分钟前`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours} 小时前`;
  const days = Math.floor(hours / 24);
  if (days < 30) return `${days} 天前`;
  return date.toLocaleDateString();
}
</script>

<template>
  <div class="shell">
    <aside class="sidebar">
      <div class="brand">
        <div class="brand-logo">知</div>
        <div>
          <div class="brand-name">知店智能客服</div>
          <div class="brand-sub">只懂店里的事</div>
        </div>
      </div>

      <button class="new-session" type="button" @click="$emit('new-session')">＋ 新建会话</button>

      <div class="sidebar-title">我的会话</div>
      <div class="session-list">
        <div v-for="item in sessions" :key="item.id" class="session-row">
          <button
            type="button"
            class="session-item"
            :class="{ active: active === 'chat' && item.id === currentId }"
            :aria-current="active === 'chat' && item.id === currentId ? 'true' : undefined"
            @click="$emit('select', item.id)"
          >
            <span class="title">{{ item.title }}</span>
            <span class="meta">{{ item.message_count }} 条 · {{ relative(item.last_message_at || item.created_at) }}</span>
          </button>
          <button
            type="button"
            class="session-delete"
            title="删除这个会话"
            aria-label="删除会话"
            @click.stop="$emit('delete-session', item.id)"
          >
            ×
          </button>
        </div>
        <p v-if="!sessions.length" class="sidebar-title">还没有会话，点上面新建</p>
      </div>

      <nav class="sidebar-nav">
        <a href="#/chat" :class="{ active: active === 'chat' }">对话</a>
        <a v-if="isAdmin" href="#/faq" :class="{ active: active === 'faq' }">高频问答</a>
        <a v-if="isAdmin" href="#/admin" :class="{ active: active === 'admin' }">知识库</a>
      </nav>

      <div class="sidebar-user">
        <div class="avatar">{{ initial }}</div>
        <div>
          <div class="name">{{ user?.username }}</div>
          <div class="role">{{ isAdmin ? "管理员" : "普通用户" }}</div>
        </div>
        <button class="logout" type="button" @click="$emit('logout')">退出</button>
      </div>
    </aside>

    <main class="main">
      <header class="topbar">
        <div>
          <h1>{{ title }}</h1>
          <div v-if="subtitle" class="sub">{{ subtitle }}</div>
        </div>
        <slot name="actions" />
      </header>
      <div v-if="notice" class="notice-bar">
        <span>{{ notice }}</span>
        <button type="button" class="notice-close" @click="$emit('dismiss-notice')">×</button>
      </div>
      <div :class="['content', bodyClass]">
        <slot />
      </div>
    </main>
  </div>
</template>
