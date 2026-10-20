<script setup>
import { computed, onMounted, reactive, ref, watch } from "vue";

import { useAuth } from "../composables/useAuth";

const props = defineProps({
  mode: { type: String, default: "login" },
  notice: { type: String, default: "" },
});

const emit = defineEmits(["done", "switch"]);

const { login, register } = useAuth();

const tab = ref(props.mode === "register" ? "register" : "login");
const errorText = ref("");
const submitting = ref(false);
const allowAdminSignup = ref(false);
const captcha = reactive({ id: "", question: "" });

const form = reactive({
  username: "",
  password: "",
  captchaAnswer: "",
  role: "user",
});

const isRegister = computed(() => tab.value === "register");

async function refreshCaptcha() {
  try {
    const response = await fetch("/api/auth/captcha", { credentials: "same-origin" });
    const payload = await response.json();
    captcha.id = payload.data.captcha_id;
    captcha.question = payload.data.question;
    allowAdminSignup.value = payload.data.allow_admin_signup;
    form.captchaAnswer = "";
  } catch {
    errorText.value = "验证码获取失败，请刷新重试";
  }
}

function switchTab(next) {
  if (next === tab.value) return;
  tab.value = next;
  errorText.value = "";
  emit("switch", next);
  if (next === "register") refreshCaptcha();
}

const MESSAGES = {
  AUTH_INVALID_CREDENTIALS: "账号或密码不正确",
  AUTH_USERNAME_TAKEN: "用户名已被占用",
  AUTH_CAPTCHA_INVALID: "验证码不正确，请重新输入",
  AUTH_CAPTCHA_EXPIRED: "验证码已过期，请刷新",
  AUTH_TOO_MANY_ATTEMPTS: "尝试过于频繁，请稍后再试",
  AUTH_FORBIDDEN: "当前不允许该操作",
};

async function submit() {
  errorText.value = "";
  submitting.value = true;
  try {
    if (isRegister.value) {
      await register({
        username: form.username.trim(),
        password: form.password,
        captcha_id: captcha.id,
        captcha_answer: form.captchaAnswer.trim(),
        role: form.role,
      });
    } else {
      await login(form.username.trim(), form.password);
    }
    emit("done");
  } catch (error) {
    errorText.value = MESSAGES[error.code] || error.message || "操作失败，请重试";
    if (isRegister.value) refreshCaptcha();
  } finally {
    submitting.value = false;
  }
}

watch(
  () => props.mode,
  (mode) => {
    tab.value = mode === "register" ? "register" : "login";
    if (isRegister.value) refreshCaptcha();
  }
);

onMounted(() => {
  if (isRegister.value) refreshCaptcha();
});
</script>

<template>
  <div class="auth-screen">
    <div class="auth-card">
      <div class="auth-brand">
        <div class="brand-logo" style="width: 34px; height: 34px">知</div>
        <div>
          <div class="brand-name">知店智能客服</div>
          <div class="brand-sub" style="color: var(--muted)">只懂店里的事</div>
        </div>
      </div>

      <div class="auth-tabs">
        <button type="button" :class="{ on: !isRegister }" @click="switchTab('login')">登录</button>
        <button type="button" :class="{ on: isRegister }" @click="switchTab('register')">注册</button>
      </div>

      <p v-if="notice" class="hint" style="margin-top: 0">{{ notice }}</p>
      <p v-if="errorText" class="form-error">{{ errorText }}</p>

      <form @submit.prevent="submit">
        <div class="field">
          <label>账号</label>
          <input v-model="form.username" autocomplete="username" placeholder="3–32 位字母、数字或下划线" />
        </div>
        <div class="field">
          <label>密码</label>
          <input
            v-model="form.password"
            type="password"
            :autocomplete="isRegister ? 'new-password' : 'current-password'"
            placeholder="至少 8 位"
          />
        </div>

        <template v-if="isRegister">
          <div class="field">
            <label>验证码</label>
            <div class="captcha-row">
              <input v-model="form.captchaAnswer" placeholder="请输入计算结果" />
              <button type="button" class="captcha" :title="'点击刷新'" @click="refreshCaptcha">
                {{ captcha.question || "加载中" }}
              </button>
            </div>
          </div>
          <div v-if="allowAdminSignup" class="field">
            <label>账号类型</label>
            <div class="role-row">
              <label><input v-model="form.role" type="radio" value="user" /> 普通用户（仅聊天）</label>
              <label><input v-model="form.role" type="radio" value="admin" /> 管理员（可管知识库）</label>
            </div>
          </div>
        </template>

        <button class="btn auth-submit" type="submit" :disabled="submitting">
          {{ submitting ? "处理中…" : isRegister ? "注册并登录" : "登录" }}
        </button>
      </form>

      <div class="auth-foot">
        {{ isRegister ? "已有账号？点上方「登录」" : "没有账号？点上方「注册」，只需账号、密码和验证码" }}
      </div>
    </div>
  </div>
</template>
