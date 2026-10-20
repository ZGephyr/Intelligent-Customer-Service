<script setup>
import { ref } from "vue";

defineProps({
  sources: { type: Array, required: true },
});

const open = ref(false);
</script>

<template>
  <div class="sources">
    <button type="button" class="sources-toggle" @click="open = !open">
      引用来源（{{ sources.length }}）{{ open ? "▲" : "▼" }}
    </button>
    <ul v-if="open" class="sources-list">
      <li v-for="source in sources" :key="source.index">
        <template v-if="source.type === 'faq'">
          <strong>[{{ source.index }}] 高频问答 · {{ source.question }}</strong>
          <span class="score">相关度 {{ source.score }}</span>
          <p>{{ source.snippet }}</p>
        </template>
        <template v-else>
          <strong>[{{ source.index }}] {{ source.source }}</strong>
          <span class="score">相关度 {{ source.score }}</span>
          <p>{{ source.snippet }}</p>
        </template>
      </li>
    </ul>
  </div>
</template>
