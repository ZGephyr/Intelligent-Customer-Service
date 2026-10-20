<script setup>
import { nextTick, onBeforeUnmount, onMounted, ref, watch } from "vue";

import { useDialog } from "../composables/useDialog";

const { state, close } = useDialog();
const confirmButton = ref(null);

watch(
  () => state.open,
  async (open) => {
    if (!open) return;
    await nextTick();
    confirmButton.value?.focus();
  }
);

function onKeydown(event) {
  if (!state.open) return;
  if (event.key === "Escape") close(false);
}

onMounted(() => window.addEventListener("keydown", onKeydown));
onBeforeUnmount(() => window.removeEventListener("keydown", onKeydown));
</script>

<template>
  <Teleport to="body">
    <Transition name="modal">
      <div v-if="state.open" class="modal-overlay" @click.self="close(false)">
        <div class="modal-card" role="dialog" aria-modal="true" :aria-label="state.title">
          <div class="modal-icon" :class="{ danger: state.danger }">{{ state.danger ? "!" : "?" }}</div>
          <h2 class="modal-title">{{ state.title }}</h2>
          <p v-if="state.message" class="modal-message">{{ state.message }}</p>
          <div class="modal-actions">
            <button
              v-if="state.mode === 'confirm'"
              type="button"
              class="btn ghost"
              @click="close(false)"
            >
              {{ state.cancelText }}
            </button>
            <button
              ref="confirmButton"
              type="button"
              class="btn"
              :class="{ danger: state.danger }"
              @click="close(true)"
            >
              {{ state.confirmText }}
            </button>
          </div>
        </div>
      </div>
    </Transition>
  </Teleport>
</template>
