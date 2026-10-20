<script setup>
import SourceCard from "./SourceCard.vue";

defineProps({
  message: { type: Object, required: true },
});
</script>

<template>
  <div class="message" :class="[message.role, message.status]">
    <div class="bubble">
      <p v-for="(line, index) in message.content.split('\n')" :key="index">{{ line }}</p>
      <span v-if="message.role === 'assistant' && !message.content" class="typing">正在输入…</span>
    </div>
    <SourceCard v-if="message.sources && message.sources.length" :sources="message.sources" />
  </div>
</template>
