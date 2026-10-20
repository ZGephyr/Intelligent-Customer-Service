<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";
import AdminPanel from "./components/AdminPanel.vue";
import ChatMessage from "./components/ChatMessage.vue";
import Composer from "./components/Composer.vue";
import { useChatSocket } from "./composables/useChatSocket";

const { messages, status, errorText, connect, send, newSession } = useChatSocket();

const draft = ref("");
const listRef = ref(null);
const route = ref(window.location.hash || "#/");
const isAdmin = computed(() => route.value.startsWith("#/admin"));

const examples = [
  "七天无理由退货谁承担运费？",
  "这款耳机保修多久？",
  "发票怎么开？",
];

function syncRoute() {
  route.value = window.location.hash || "#/";
}

onMounted(() => {
  window.addEventListener("hashchange", syncRoute);
  connect();
});

onBeforeUnmount(() => {
  window.removeEventListener("hashchange", syncRoute);
});

watch(
  () => messages.value.length,
  async () => {
    await nextTick();
    if (listRef.value) listRef.value.scrollTop = listRef.value.scrollHeight;
  }
);

function submit() {
  if (send(draft.value)) draft.value = "";
}

function fill(text) {
  draft.value = text;
}
</script>

<template>
  <AdminPanel v-if="isAdmin" />

  <div v-else class="page">
    <header class="header">
      <h1>店铺智能客服</h1>
      <div class="header-right">
        <a class="nav-link" href="#/admin">知识库</a>
        <span class="badge" :data-status="status">{{ status }}</span>
      </div>
    </header>

    <main ref="listRef" class="messages">
      <div v-if="!messages.length" class="empty">
        <p>您好，我是店铺智能客服，可以回答商品、退换货、发票等问题。</p>
        <div class="examples">
          <button v-for="item in examples" :key="item" type="button" @click="fill(item)">
            {{ item }}
          </button>
        </div>
      </div>
      <ChatMessage v-for="message in messages" :key="message.id" :message="message" />
    </main>

    <p v-if="errorText" class="error">{{ errorText }}</p>

    <Composer
      v-model="draft"
      :disabled="status === 'connecting' || status === 'streaming'"
      @send="submit"
      @new-session="newSession"
    />
  </div>
</template>
