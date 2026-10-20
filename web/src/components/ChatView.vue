<script setup>
import { nextTick, ref, watch } from "vue";

import ChatMessage from "./ChatMessage.vue";
import { useChatSocket } from "../composables/useChatSocket";

const props = defineProps({
  sessionId: { type: String, default: "" },
  title: { type: String, default: "新会话" },
});

const emit = defineEmits(["finished"]);

const draft = ref("");
const listRef = ref(null);
const connecting = ref(false);

const { messages, status, errorText, connect, switchSession, send } = useChatSocket(() => props.sessionId);

const examples = ["七天无理由退货谁承担运费？", "这款耳机保修多久？", "发票怎么开？"];

watch(
  () => props.sessionId,
  async (sessionId) => {
    if (!sessionId) {
      messages.value = [];
      return;
    }
    connecting.value = true;
    switchSession(sessionId);
    connecting.value = false;
    await nextTick();
    scrollToBottom();
  },
  { immediate: true }
);

let wasStreaming = false;
watch(
  () => status.value,
  (value) => {
    if (wasStreaming && value === "idle") emit("finished");
    wasStreaming = value === "streaming";
  }
);

watch(
  () => messages.value.length,
  async () => {
    await nextTick();
    scrollToBottom();
  }
);

function scrollToBottom() {
  if (listRef.value) listRef.value.scrollTop = listRef.value.scrollHeight;
}

function submit() {
  if (send(draft.value)) draft.value = "";
}

function fill(text) {
  draft.value = text;
}
</script>

<template>
  <div class="chat">
    <div ref="listRef" class="messages">
      <div v-if="!messages.length" class="empty">
        <h2>您好，我是知店智能客服</h2>
        <p>可以回答商品、退换货、发票、物流等问题，只依据店铺资料作答。</p>
        <div class="examples">
          <button v-for="item in examples" :key="item" type="button" @click="fill(item)">
            {{ item }}
          </button>
        </div>
      </div>
      <ChatMessage v-for="message in messages" :key="message.id" :message="message" />
    </div>

    <p v-if="errorText" class="error">{{ errorText }}</p>

    <form class="composer" @submit.prevent="submit">
      <input
        v-model="draft"
        type="text"
        maxlength="2000"
        placeholder="请输入您的问题，例如：七天无理由退货谁承担运费？"
        :disabled="connecting || status === 'streaming'"
      />
      <button class="btn" type="submit" :disabled="connecting">发送</button>
    </form>
  </div>
</template>
