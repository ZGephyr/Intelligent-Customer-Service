<script setup>
import { computed } from "vue";
import SourceCard from "./SourceCard.vue";

const props = defineProps({
  message: { type: Object, required: true },
});

const LABELS = {
  ok: "资料回答",
  faq: "高频问答",
  chitchat: "闲聊",
  fallback: "未找到相关信息",
  out_of_scope: "非业务问题",
  error: "出错",
};

const label = computed(() =>
  props.message.role === "assistant" ? LABELS[props.message.status] ?? "" : ""
);

const typing = computed(() => props.message.role === "assistant" && !props.message.content);

const showSources = computed(
  () => props.message.status !== "out_of_scope" && (props.message.sources?.length ?? 0) > 0
);
</script>

<template>
  <div class="message" :class="[message.role, message.status]">
    <div class="who">{{ message.role === "user" ? "我" : "知" }}</div>
    <div class="body">
      <div class="bubble">
        <span v-if="label" class="tag" :data-kind="message.status">{{ label }}</span>
        <p v-for="(line, index) in message.content.split('\n')" :key="index">{{ line }}</p>
        <span v-if="typing" class="typing">正在输入…</span>
      </div>
      <SourceCard v-if="showSources" :sources="message.sources" />
    </div>
  </div>
</template>
