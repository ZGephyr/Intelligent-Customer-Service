<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from "vue";

import { useDialog } from "../composables/useDialog";

const { confirm, alert: showAlert } = useDialog();
const config = ref(null);
const queue = ref([]);
const documents = ref([]);
const loading = ref(false);
const uploading = ref(false);
const errorText = ref("");
let timer = null;
let seq = 0;

const accept = computed(() =>
  (config.value?.supported_extensions ?? []).map((ext) => `.${ext}`).join(",")
);

const hasProcessing = computed(() => documents.value.some((item) => item.status === "processing"));

function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function statusLabel(status) {
  return { processing: "解析中", ready: "可用", failed: "失败" }[status] ?? status;
}

async function loadConfig() {
  try {
    const response = await fetch("/api/admin/config");
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.message || "读取配置失败");
    config.value = payload.data;
  } catch (error) {
    errorText.value = String(error.message || error);
  }
}

async function loadDocuments() {
  loading.value = true;
  try {
    const response = await fetch("/api/admin/documents?page=1&size=100");
    const payload = await response.json();
    if (response.status === 401) {
      errorText.value = "后端已开启管理鉴权，请关闭 ADMIN_AUTH_ENABLED 或改用带令牌的接口调用";
      documents.value = [];
      return;
    }
    if (!response.ok) throw new Error(payload.message || "读取文档失败");
    documents.value = payload.data.items;
    errorText.value = "";
  } catch (error) {
    errorText.value = String(error.message || error);
  } finally {
    loading.value = false;
  }
}

function startPolling() {
  stopPolling();
  if (!hasProcessing.value) return;
  timer = setInterval(async () => {
    await loadDocuments();
    if (!hasProcessing.value) stopPolling();
  }, 2000);
}

function stopPolling() {
  if (timer) {
    clearInterval(timer);
    timer = null;
  }
}

function validate(file) {
  const extension = file.name.split(".").pop()?.toLowerCase() ?? "";
  const allowed = config.value?.supported_extensions ?? [];
  if (!allowed.includes(extension)) {
    return `不支持的格式 .${extension}，仅支持 ${allowed.join(" / ")}`;
  }
  const limit = (config.value?.max_upload_mb ?? 20) * 1024 * 1024;
  if (file.size > limit) {
    return `文件超过 ${config.value?.max_upload_mb ?? 20}MB`;
  }
  return "";
}

function addFiles(fileList) {
  for (const file of Array.from(fileList)) {
    const message = validate(file);
    queue.value.push({
      id: `f-${seq++}`,
      name: file.name,
      size: file.size,
      raw: file,
      status: message ? "invalid" : "pending",
      message,
    });
  }
}

function onFileInput(event) {
  addFiles(event.target.files);
  event.target.value = "";
}

function onDrop(event) {
  addFiles(event.dataTransfer.files);
}

async function uploadAll() {
  const pending = queue.value.filter((item) => item.status === "pending");
  if (!pending.length) return;
  uploading.value = true;
  errorText.value = "";

  for (const item of pending) {
    item.status = "uploading";
    const form = new FormData();
    form.append("file", item.raw, item.name);
    try {
      const response = await fetch("/api/admin/documents", { method: "POST", body: form });
      if (response.status === 202) {
        item.status = "accepted";
        item.message = "已提交，后台解析中";
      } else {
        const payload = await response.json().catch(() => ({}));
        item.status = "error";
        item.message = payload.message || `上传失败（${response.status}）`;
      }
    } catch (error) {
      item.status = "error";
      item.message = "网络错误，请重试";
    }
  }

  uploading.value = false;
  queue.value = queue.value.filter((item) => item.status !== "accepted");
  await loadDocuments();
  startPolling();
}

async function removeDocument(item) {
  const confirmed = await confirm({
    title: "删除文档",
    message: `「${item.filename}」的向量与分块会被一并清理。`,
    confirmText: "删除",
    danger: true,
  });
  if (!confirmed) return;
  const response = await fetch(`/api/admin/documents/${item.id}`, { method: "DELETE" });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    errorText.value = payload.message || "删除失败";
    return;
  }
  await loadDocuments();
}

async function reindex() {
  const confirmed = await confirm({
    title: "重建索引",
    message: "将清空现有向量并用 MySQL 中的分块重建，期间问答可能暂时检索不到内容。",
    confirmText: "开始重建",
    danger: true,
  });
  if (!confirmed) return;
  const response = await fetch("/api/admin/reindex", { method: "POST" });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    errorText.value = payload.message || "重建失败";
    return;
  }
  await showAlert({
    title: "重建完成",
    message: `共 ${payload.data.chunks} 条分块已重新向量化。`,
    confirmText: "知道了",
  });
  await loadDocuments();
}

onMounted(async () => {
  await loadConfig();
  await loadDocuments();
  startPolling();
});

onBeforeUnmount(stopPolling);
</script>

<template>
  <div class="panel-page">
    <main>
      <p class="hint">
        支持
        <strong>{{ (config?.supported_extensions ?? []).join(" / ") }}</strong>
        ，单文件不超过 {{ config?.max_upload_mb ?? 20 }}MB；相同文件只会入库一次。
      </p>

      <section
        class="dropzone"
        @dragover.prevent
        @drop.prevent="onDrop"
      >
        <p>把文件拖到这里，或</p>
        <label class="file-button">
          选择文件
          <input
            type="file"
            multiple
            :accept="accept"
            hidden
            @change="onFileInput"
          />
        </label>
        <p class="muted">文件会先校验格式与大小，再逐个上传</p>
      </section>

      <section v-if="queue.length" class="queue">
        <div v-for="item in queue" :key="item.id" class="queue-item" :class="item.status">
          <span class="file-name">{{ item.name }}</span>
          <span class="file-size">{{ formatSize(item.size) }}</span>
          <span class="file-status">{{ item.message || item.status }}</span>
        </div>
        <button
          class="primary"
          :disabled="uploading || !queue.some((item) => item.status === 'pending')"
          @click="uploadAll"
        >
          {{ uploading ? "上传中…" : "上传全部" }}
        </button>
      </section>

      <p v-if="errorText" class="error">{{ errorText }}</p>

      <section class="documents">
        <div class="documents-header">
          <h2>已入库文档</h2>
          <div class="actions">
            <button class="ghost" :disabled="loading" @click="loadDocuments">刷新</button>
            <button class="ghost" @click="reindex">重建索引</button>
          </div>
        </div>

        <p v-if="!documents.length" class="muted">还没有文档，先上传一份试试。</p>

        <table v-else class="doc-table">
          <thead>
            <tr>
              <th>文件名</th>
              <th>类型</th>
              <th>大小</th>
              <th>状态</th>
              <th>分块</th>
              <th>操作</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="item in documents" :key="item.id">
              <td :title="item.error_message || ''">{{ item.filename }}</td>
              <td>{{ item.file_ext }}</td>
              <td>{{ formatSize(item.file_size) }}</td>
              <td>
                <span class="status" :class="item.status">{{ statusLabel(item.status) }}</span>
                <div v-if="item.error_message" class="row-error">{{ item.error_message }}</div>
              </td>
              <td>{{ item.chunk_count }}</td>
              <td>
                <button class="danger" @click="removeDocument(item)">删除</button>
              </td>
            </tr>
          </tbody>
        </table>
      </section>
    </main>
  </div>
</template>
