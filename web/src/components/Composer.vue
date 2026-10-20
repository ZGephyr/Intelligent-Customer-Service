<script setup>
defineProps({
  modelValue: { type: String, default: "" },
  disabled: { type: Boolean, default: false },
});

defineEmits(["update:modelValue", "send", "new-session"]);
</script>

<template>
  <form class="composer" @submit.prevent="$emit('send')">
    <input
      :value="modelValue"
      type="text"
      maxlength="2000"
      placeholder="请输入您的问题，例如：七天无理由退货谁承担运费？"
      :disabled="disabled"
      @input="$emit('update:modelValue', $event.target.value)"
    />
    <button type="submit" :disabled="disabled">发送</button>
    <button type="button" class="ghost" @click="$emit('new-session')">新建会话</button>
  </form>
</template>
