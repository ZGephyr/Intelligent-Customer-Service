<script setup>
import { computed, onMounted, reactive, ref } from "vue";

import { useDialog } from "../composables/useDialog";

const { confirm } = useDialog();
const stats = ref({ total: 0, enabled: 0, hit_total: 0, top: [] });
const items = ref([]);
const total = ref(0);
const page = ref(1);
const size = 20;
const loading = ref(false);
const errorText = ref("");
const keyword = ref("");
const category = ref("");
const enabledFilter = ref("all");
const importSummary = ref("");

const form = reactive({
  id: "",
  question: "",
  answer: "",
  category: "",
  aliases: "",
  enabled: true,
});
const editing = ref(false);
const saving = ref(false);

const totalPages = computed(() => Math.max(1, Math.ceil(total.value / size)));

function statusLabel(value) {
  return value ? "启用" : "停用";
}

async function requestJson(url, options) {
  const response = await fetch(url, options);
  const payload = await response.json().catch(() => ({}));
  if (response.status === 401) {
    throw new Error("后端已开启管理鉴权，请关闭 ADMIN_AUTH_ENABLED 或改用带令牌的接口调用");
  }
  if (!response.ok) {
    throw new Error(payload.message || "请求失败");
  }
  return payload.data;
}

async function loadStats() {
  try {
    stats.value = await requestJson("/api/admin/faqs/stats");
  } catch (error) {
    errorText.value = String(error.message || error);
  }
}

async function loadList() {
  loading.value = true;
  try {
    const params = new URLSearchParams({ page: String(page.value), size: String(size) });
    if (keyword.value.trim()) params.set("keyword", keyword.value.trim());
    if (category.value.trim()) params.set("category", category.value.trim());
    if (enabledFilter.value !== "all") params.set("enabled", enabledFilter.value);

    const data = await requestJson(`/api/admin/faqs?${params.toString()}`);
    items.value = data.items;
    total.value = data.total;
    errorText.value = "";
  } catch (error) {
    errorText.value = String(error.message || error);
  } finally {
    loading.value = false;
  }
}

async function refresh() {
  await Promise.all([loadList(), loadStats()]);
}

function resetForm() {
  form.id = "";
  form.question = "";
  form.answer = "";
  form.category = "";
  form.aliases = "";
  form.enabled = true;
}

function startCreate() {
  resetForm();
  editing.value = true;
}

function startEdit(row) {
  form.id = row.id;
  form.question = row.question;
  form.answer = row.answer;
  form.category = row.category ?? "";
  form.aliases = row.aliases ?? "";
  form.enabled = row.enabled;
  editing.value = true;
}

function cancelEdit() {
  editing.value = false;
  resetForm();
}

async function save() {
  if (!form.question.trim() || !form.answer.trim()) {
    errorText.value = "问题与答案都不能为空";
    return;
  }
  saving.value = true;
  try {
    const body = JSON.stringify({
      question: form.question.trim(),
      answer: form.answer.trim(),
      category: form.category.trim() || null,
      aliases: form.aliases.trim() || null,
      enabled: form.enabled,
    });
    if (form.id) {
      await requestJson(`/api/admin/faqs/${form.id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body,
      });
    } else {
      await requestJson("/api/admin/faqs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body,
      });
    }
    editing.value = false;
    resetForm();
    await refresh();
  } catch (error) {
    errorText.value = String(error.message || error);
  } finally {
    saving.value = false;
  }
}

async function remove(row) {
  const confirmed = await confirm({
    title: "删除问答",
    message: `「${row.question}」会被永久删除。`,
    confirmText: "删除",
    danger: true,
  });
  if (!confirmed) return;
  try {
    await requestJson(`/api/admin/faqs/${row.id}`, { method: "DELETE" });
    await refresh();
  } catch (error) {
    errorText.value = String(error.message || error);
  }
}

async function toggle(row) {
  try {
    await requestJson(`/api/admin/faqs/${row.id}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enabled: !row.enabled }),
    });
    await refresh();
  } catch (error) {
    errorText.value = String(error.message || error);
  }
}

async function onImport(event) {
  const file = event.target.files?.[0];
  if (!file) return;
  importSummary.value = "";
  try {
    const body = new FormData();
    body.append("file", file);
    const data = await requestJson("/api/admin/faqs/import", { method: "POST", body });
    importSummary.value = `导入完成：新增 ${data.created}，更新 ${data.updated}，跳过 ${data.skipped}`;
    await refresh();
  } catch (error) {
    errorText.value = String(error.message || error);
  } finally {
    event.target.value = "";
  }
}

function search() {
  page.value = 1;
  loadList();
}

function go(delta) {
  const next = page.value + delta;
  if (next < 1 || next > totalPages.value) return;
  page.value = next;
  loadList();
}

onMounted(refresh);
</script>

<template>
  <div class="panel-page">
    <section class="faq-stats">
      <div class="stat"><span>{{ stats.total }}</span>总条数</div>
      <div class="stat"><span>{{ stats.enabled }}</span>启用中</div>
      <div class="stat"><span>{{ stats.hit_total }}</span>累计命中</div>
    </section>

    <section class="faq-toolbar">
      <input v-model="keyword" type="search" placeholder="搜索问题或别名" @keyup.enter="search" />
      <input v-model="category" type="text" placeholder="分类，可空" @keyup.enter="search" />
      <select v-model="enabledFilter" @change="search">
        <option value="all">全部状态</option>
        <option value="true">启用</option>
        <option value="false">停用</option>
      </select>
      <button type="button" @click="search">搜索</button>
      <button type="button" class="primary" @click="startCreate">新增问答</button>
      <label class="import">
        导入 CSV/XLSX
        <input type="file" accept=".csv,.xlsx" @change="onImport" />
      </label>
    </section>

    <p v-if="importSummary" class="hint">{{ importSummary }}</p>
    <p v-if="errorText" class="error">{{ errorText }}</p>

    <table class="faq-table">
      <thead>
        <tr>
          <th>问题</th>
          <th>分类</th>
          <th>状态</th>
          <th>命中</th>
          <th>最近命中</th>
          <th>操作</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="row in items" :key="row.id">
          <td>
            <strong>{{ row.question }}</strong>
            <p class="snippet">{{ row.answer }}</p>
          </td>
          <td>{{ row.category || "—" }}</td>
          <td>{{ statusLabel(row.enabled) }}</td>
          <td>{{ row.hit_count }}</td>
          <td>{{ row.last_hit_at ? row.last_hit_at.replace("T", " ").slice(0, 19) : "—" }}</td>
          <td class="actions">
            <button type="button" @click="startEdit(row)">编辑</button>
            <button type="button" @click="toggle(row)">{{ row.enabled ? "停用" : "启用" }}</button>
            <button type="button" class="danger" @click="remove(row)">删除</button>
          </td>
        </tr>
        <tr v-if="!items.length && !loading">
          <td colspan="6" class="empty">还没有高频问答，点「新增问答」或导入表格。</td>
        </tr>
      </tbody>
    </table>

    <div class="pager">
      <button type="button" :disabled="page <= 1" @click="go(-1)">上一页</button>
      <span>{{ page }} / {{ totalPages }}（共 {{ total }} 条）</span>
      <button type="button" :disabled="page >= totalPages" @click="go(1)">下一页</button>
    </div>

    <section v-if="editing" class="faq-form">
      <h2>{{ form.id ? "编辑问答" : "新增问答" }}</h2>
      <label>问题<input v-model="form.question" type="text" maxlength="500" /></label>
      <label>答案<textarea v-model="form.answer" rows="4" /></label>
      <label>分类<input v-model="form.category" type="text" maxlength="32" /></label>
      <label>别名（一行一条）<textarea v-model="form.aliases" rows="3" /></label>
      <label class="inline">
        <input v-model="form.enabled" type="checkbox" />
        启用
      </label>
      <div class="form-actions">
        <button type="button" class="primary" :disabled="saving" @click="save">保存</button>
        <button type="button" @click="cancelEdit">取消</button>
      </div>
    </section>
  </div>
</template>

<style scoped>
.faq-stats {
  display: flex;
  gap: 12px;
  padding: 0 24px 8px;
}

.stat {
  flex: 1;
  padding: 12px;
  border: 1px solid #e2e8f0;
  border-radius: 10px;
  color: #64748b;
  font-size: 13px;
}

.stat span {
  display: block;
  font-size: 22px;
  font-weight: 600;
  color: #0f172a;
}

.faq-toolbar {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  padding: 8px 24px;
  align-items: center;
}

.faq-toolbar input[type="search"],
.faq-toolbar input[type="text"],
.faq-toolbar select {
  padding: 6px 10px;
  border: 1px solid #cbd5e1;
  border-radius: 8px;
}

.import {
  padding: 6px 12px;
  border: 1px dashed #94a3b8;
  border-radius: 8px;
  cursor: pointer;
  font-size: 13px;
}

.import input {
  display: none;
}

.faq-table {
  width: calc(100% - 48px);
  margin: 8px 24px;
  border-collapse: collapse;
  font-size: 14px;
}

.faq-table th,
.faq-table td {
  padding: 8px 10px;
  border-bottom: 1px solid #e2e8f0;
  text-align: left;
  vertical-align: top;
}

.snippet {
  margin: 2px 0 0;
  color: #64748b;
  font-size: 12px;
}

.actions {
  display: flex;
  gap: 6px;
}

.empty {
  text-align: center;
  color: #94a3b8;
  padding: 24px;
}

.pager {
  display: flex;
  gap: 12px;
  align-items: center;
  padding: 8px 24px;
  font-size: 13px;
}

.faq-form {
  margin: 12px 24px 32px;
  padding: 16px;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  display: grid;
  gap: 10px;
  max-width: 640px;
}

.faq-form label {
  display: grid;
  gap: 4px;
  font-size: 13px;
  color: #475569;
}

.faq-form input[type="text"],
.faq-form textarea {
  padding: 8px 10px;
  border: 1px solid #cbd5e1;
  border-radius: 8px;
  font: inherit;
}

.faq-form .inline {
  display: flex;
  align-items: center;
  gap: 8px;
}

.form-actions {
  display: flex;
  gap: 8px;
}
</style>
