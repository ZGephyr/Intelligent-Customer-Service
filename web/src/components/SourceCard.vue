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
        <strong>[{{ source.index }}] {{ source.source }}</strong>
        <span class="score">相关度 {{ source.score }}</span>
        <p>{{ source.snippet }}</p>
      </li>
    </ul>
  </div>
</template>
