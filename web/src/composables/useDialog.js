import { reactive } from "vue";

const state = reactive({
  open: false,
  mode: "confirm",
  title: "确认操作",
  message: "",
  confirmText: "确定",
  cancelText: "取消",
  danger: false,
  resolve: null,
});

function show(options = {}) {
  Object.assign(state, {
    open: true,
    mode: "confirm",
    title: "确认操作",
    message: "",
    confirmText: "确定",
    cancelText: "取消",
    danger: false,
    ...options,
  });
  return new Promise((resolve) => {
    state.resolve = resolve;
  });
}

function close(result) {
  state.open = false;
  const resolve = state.resolve;
  state.resolve = null;
  if (resolve) resolve(result);
}

export function useDialog() {
  return {
    state,
    confirm: (options) => show({ ...options, mode: "confirm" }),
    alert: (options) => show({ ...options, mode: "alert" }),
    close,
  };
}
