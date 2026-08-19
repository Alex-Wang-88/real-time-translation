const $ = (selector) => document.querySelector(selector);
const THEME_STORAGE_KEY = "realtime-meeting-theme";

const RECOMMENDED_ASR_SETTINGS = Object.freeze({
  silence_ms: 850,
  vad_minimum_speech_ms: 200,
});

const RECOMMENDED_MEETING_SETTINGS = Object.freeze({
  ...RECOMMENDED_ASR_SETTINGS,
  volume_threshold_percent: 1.0,
  speech_start_ms: 80,
  audio_pre_roll_ms: 400,
  vad_minimum_speech_ratio: 0.06,
  max_utterance_seconds: 18,
  partial_interval_ms: 800,
  audio_segment_minutes: 30,
  translation_beam_size: 2,
  translation_max_decoding_length: 384,
  translation_repetition_penalty: 1.05,
  keep_audio: true,
});

const MEETING_NUMBER_FIELDS = [
  { key: "volume_threshold_percent", slider: "volumeThreshold", input: "volumeThresholdValue", min: 0, max: 30, step: 0.1 },
  { key: "silence_ms", slider: "asrSilenceMs", input: "asrSilenceMsValue", min: 160, max: 2000, step: 10 },
  { key: "partial_interval_ms", slider: "asrPartialIntervalMs", input: "asrPartialIntervalMsValue", min: 100, max: 5000, step: 50 },
  { key: "audio_segment_minutes", slider: "audioSegmentMinutes", input: "audioSegmentMinutesValue", min: 1, max: 120, step: 1 },
];

// Keep the old name as a small compatibility surface for extensions and
// static checks from the first settings panel version.
const ASR_SETTING_FIELDS = MEETING_NUMBER_FIELDS.filter((field) => [
  "silence_ms", "vad_minimum_speech_ms",
].includes(field.key));

// The live microphone level is normalized to 0-100%.  Keep this separate
// from the 0-30% background-noise filter threshold range.
const MICROPHONE_METER_MAX_PERCENT = 100;
const MICROPHONE_LEVEL_SETTLE_EPSILON = 0.0025;
const MICROPHONE_LEVEL_ATTACK_MS = 55;
const MICROPHONE_LEVEL_RELEASE_MS = 130;

const state = {
  meetings: [],
  meeting: null,
  ws: null,
  stream: null,
  audioContext: null,
  audioSource: null,
  audioNode: null,
  audioReady: false,
  reconnectTimer: null,
  reconnectAttempt: 0,
  intentionalClose: false,
  streamGeneration: 0,
  audioFlushWaiters: new Map(),
  pingTimer: null,
  transcript: new Map(),
  transcriptNodes: new Map(),
  loadedTranscriptRevision: 0,
  transcriptLoadedFor: null,
  transcriptLoadToken: 0,
  draft: null,
  draftNode: null,
  transcriptNearBottom: true,
  timer: null,
  renameMeetingId: null,
  confirmResolver: null,
  authResolver: null,
  authPromise: null,
  volumeThresholdPercent: 2.2,
  asrSettings: { ...RECOMMENDED_ASR_SETTINGS },
  meetingSettings: { ...RECOMMENDED_MEETING_SETTINGS },
  microphoneLevelPercent: 0,
  pendingMicrophoneLevel: 0,
  microphoneLevelFrame: null,
  microphoneLevelFrameAt: 0,
  audioStreamingEnabled: false,
  inputDevicesRefreshId: 0,
  pendingInputDevices: null,
  inputDevicePickerOpen: false,
};

const dom = {
  welcome: $("#welcomePanel"),
  meetingPanel: $("#meetingPanel"),
  meetingList: $("#meetingList"),
  search: $("#meetingSearch"),
  pageTitle: $("#pageTitle"),
  pageSubtitle: $("#pageSubtitle"),
  startMeeting: $("#startMeeting"),
  newMeeting: $("#newMeeting"),
  connectionBadge: $("#connectionBadge"),
  themeToggle: $("#themeToggle"),
  meetingTitle: $("#meetingTitle"),
  transcriptList: $("#transcriptList"),
  transcriptEmpty: $("#transcriptEmpty"),
  utteranceCount: $("#utteranceCount"),
  jumpLatest: $("#jumpLatest"),
  recordingIndicator: $("#recordingIndicator"),
  recordingState: $("#recordingState"),
  recordingHint: $("#recordingHint"),
  timer: $("#timer"),
  startRecordingButton: $("#startRecordingButton"),
  recordButton: $("#recordButton"),
  levelBar: $("#levelBar"),
  levelText: $("#levelText"),
  inputDevice: $("#inputDevice"),
  inputDevicePicker: $("#inputDevicePicker"),
  inputDeviceTrigger: $("#inputDeviceTrigger"),
  inputDeviceOptions: $("#inputDeviceOptions"),
  notice: $("#notice"),
  refinedBadge: $("#refinedBadge"),
  refinedTranscriptText: $("#refinedTranscriptText"),
  summaryBadge: $("#summaryBadge"),
  summaryText: $("#summaryText"),
  retrySummary: $("#retrySummary"),
  downloadSummary: $("#downloadSummary"),
  todoBadge: $("#todoBadge"),
  todoText: $("#todoText"),
  downloadTodo: $("#downloadTodo"),
  filesCard: $("#filesCard"),
  fileLinks: $("#fileLinks"),
  statusDot: $("#statusDot"),
  systemStatus: $("#systemStatus"),
  dialog: $("#newMeetingDialog"),
  dialogForm: $("#newMeetingForm"),
  dialogTitle: $("#dialogTitle"),
  renameDialog: $("#renameMeetingDialog"),
  renameDialogForm: $("#renameMeetingForm"),
  renameDialogTitle: $("#renameMeetingTitle"),
  renameMeetingSubmit: $("#renameMeetingSubmit"),
  confirmDialog: $("#confirmDialog"),
  confirmDialogForm: $("#confirmDialogForm"),
  confirmDialogTitle: $("#confirmDialogTitle"),
  confirmDialogMessage: $("#confirmDialogMessage"),
  confirmDialogSubmit: $("#confirmDialogSubmit"),
  authDialog: $("#authDialog"),
  authDialogForm: $("#authDialogForm"),
  authToken: $("#authToken"),
  volumeThreshold: $("#volumeThreshold"),
  volumeThresholdValue: $("#volumeThresholdValue"),
  thresholdMeter: $("#thresholdMeter"),
  microphoneLevelFill: $("#microphoneLevelFill"),
  microphoneLevelMarker: $("#microphoneLevelMarker"),
  microphoneLevelValue: $("#microphoneLevelValue"),
  microphoneLevelStatus: $("#microphoneLevelStatus"),
  inputDeviceSummary: $("#inputDeviceSummary"),
  volumeThresholdSummary: $("#volumeThresholdSummary"),
  settingsAudioSection: $("#settingsAudioSection"),
  openAsrSettings: $("#openAsrSettings"),
  asrSettingsDialog: $("#asrSettingsDialog"),
  asrSettingsForm: $("#asrSettingsForm"),
  asrSettingsNotice: $("#asrSettingsNotice"),
  resetAsrSettings: $("#resetAsrSettings"),
  saveAsrSettings: $("#saveAsrSettings"),
};

function storedTheme() {
  try {
    return window.localStorage.getItem(THEME_STORAGE_KEY) === "dark" ? "dark" : "light";
  } catch {
    return "light";
  }
}

function applyTheme(theme, { persist = true } = {}) {
  const selected = theme === "dark" ? "dark" : "light";
  document.documentElement.dataset.theme = selected;
  if (persist) {
    try { window.localStorage.setItem(THEME_STORAGE_KEY, selected); } catch {}
  }
  const dark = selected === "dark";
  if (dom.themeToggle) {
    dom.themeToggle.setAttribute("aria-pressed", String(dark));
    const label = dark ? "切换到浅色模式" : "切换到深色模式";
    dom.themeToggle.setAttribute("aria-label", label);
    dom.themeToggle.title = label;
    const icon = dom.themeToggle.querySelector(".theme-toggle-icon");
    if (icon) icon.textContent = dark ? "☀" : "☾";
  }
}

const recordingLabels = {
  created: "待开始",
  starting: "准备录音",
  recording: "正在录音",
  finalizing: "正在保存",
  complete: "录音完成",
  error: "录音异常",
};
const summaryLabels = { idle: "等待手动生成", queued: "排队中", running: "生成中", complete: "已完成", error: "生成失败" };
const todoLabels = { waiting_summary: "等待手动生成", queued: "排队中", running: "生成中", complete: "已完成", stale: "等待重新生成", error: "生成失败" };
const refinedLabels = { idle: "等待手动生成", queued: "排队中", running: "精修中", complete: "已完成", error: "生成失败" };
const languageLabels = { zh: "中文", en: "英文", de: "德文", unknown: "自动判断" };
const variantLabels = {
  mandarin: "普通话",
  anhui: "安徽方言",
  dongbei: "东北方言",
  fujian: "福建方言",
  gansu: "甘肃方言",
  guizhou: "贵州方言",
  hebei: "河北方言",
  henan: "河南方言",
  hubei: "湖北方言",
  hunan: "湖南方言",
  jiangxi: "江西方言",
  ningxia: "宁夏方言",
  shandong: "山东方言",
  shaanxi: "陕西方言",
  shanxi: "山西方言",
  sichuan: "四川方言",
  tianjin: "天津方言",
  yunnan: "云南方言",
  zhejiang: "浙江方言",
  cantonese_hong_kong: "粤语（香港口音）",
  cantonese_guangdong: "粤语（广东口音）",
  cantonese_unknown: "粤语（口音未确认）",
  wu: "吴语",
  minnan: "闽南语",
  unknown: "方言未确认",
};

function paragraphLanguageLabel(item) {
  const language = languageLabels[item?.language] || languageLabels.unknown;
  const variant = item?.language === "zh" && item?.speech_variant && item.speech_variant !== "unknown"
    ? ` · ${variantLabels[item.speech_variant] || item.speech_variant}`
    : "";
  return `${language}${variant}`;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderInlineMarkdown(value) {
  return escapeHtml(value)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/(?<!\*)\*([^*]+)\*(?!\*)/g, "<em>$1</em>");
}

function markdownToHtml(markdown) {
  const lines = String(markdown || "").split(/\r?\n/);
  const result = [];
  let tableRows = [];

  const tableCells = (line) => line.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((cell) => cell.trim().replaceAll("\\|", "|"));
  const flushTable = () => {
    if (!tableRows.length) return;
    const hasHeader = tableRows.length >= 2 && tableRows[1].every((cell) => /^:?-{3,}:?$/.test(cell));
    const rows = hasHeader ? [tableRows[0], ...tableRows.slice(2)] : tableRows;
    const head = hasHeader ? `<thead><tr>${rows[0].map((cell) => `<th>${renderInlineMarkdown(cell)}</th>`).join("")}</tr></thead>` : "";
    const bodyRows = hasHeader ? rows.slice(1) : rows;
    const body = bodyRows.length
      ? `<tbody>${bodyRows.map((row) => `<tr>${row.map((cell) => `<td>${renderInlineMarkdown(cell)}</td>`).join("")}</tr>`).join("")}</tbody>`
      : "";
    result.push(`<table>${head}${body}</table>`);
    tableRows = [];
  };

  const flushList = (items, tag) => {
    if (!items.length) return;
    result.push(`<${tag}>${items.map((item) => `<li>${renderInlineMarkdown(item)}</li>`).join("")}</${tag}>`);
    items.length = 0;
  };

  let unorderedItems = [];
  let orderedItems = [];
  for (const rawLine of lines) {
    const line = rawLine.trimEnd();
    if (/^\|.*\|$/.test(line)) {
      flushList(unorderedItems, "ul");
      flushList(orderedItems, "ol");
      tableRows.push(tableCells(line));
      continue;
    }
    flushTable();
    if (!line.trim()) {
      flushList(unorderedItems, "ul");
      flushList(orderedItems, "ol");
    } else if (/^#{1,6} /.test(line)) {
      flushList(unorderedItems, "ul");
      flushList(orderedItems, "ol");
      const match = line.match(/^(#{1,6})\s+(.+)$/);
      const level = Math.min(6, match[1].length + 2);
      result.push(`<h${level}>${renderInlineMarkdown(match[2])}</h${level}>`);
    } else if (/^[-*+] /.test(line)) {
      flushList(orderedItems, "ol");
      unorderedItems.push(line.slice(2));
    } else if (/^\d+[.)] /.test(line)) {
      flushList(unorderedItems, "ul");
      orderedItems.push(line.replace(/^\d+[.)]\s+/, ""));
    } else if (/^> ?/.test(line)) {
      flushList(unorderedItems, "ul");
      flushList(orderedItems, "ol");
      result.push(`<blockquote>${renderInlineMarkdown(line.replace(/^> ?/, ""))}</blockquote>`);
    } else if (/^(---+|\*\*\*+|___+)\s*$/.test(line)) {
      flushList(unorderedItems, "ul");
      flushList(orderedItems, "ol");
      result.push("<hr>");
    } else {
      flushList(unorderedItems, "ul");
      flushList(orderedItems, "ol");
      result.push(`<p>${renderInlineMarkdown(line)}</p>`);
    }
  }
  flushTable();
  flushList(unorderedItems, "ul");
  flushList(orderedItems, "ol");
  return result.join("");
}

function requestConfirmation(title, message, confirmLabel = "确认") {
  return new Promise((resolve) => {
    state.confirmResolver = resolve;
    dom.confirmDialogTitle.textContent = title;
    dom.confirmDialogMessage.textContent = message;
    dom.confirmDialogSubmit.textContent = confirmLabel;
    dom.confirmDialog.showModal();
    dom.confirmDialogSubmit.focus();
  });
}

function requestAuthToken() {
  if (state.authPromise) return state.authPromise;
  state.authPromise = new Promise((resolve) => {
    state.authResolver = resolve;
    dom.authToken.value = "";
    dom.authDialog.showModal();
    dom.authToken.focus();
  });
  return state.authPromise;
}

async function requestJson(path, options = {}) {
  const { authRetried = false, ...fetchOptions } = options;
  const response = await fetch(path, {
    ...fetchOptions,
    credentials: "same-origin",
    headers: { ...(options.body ? { "Content-Type": "application/json" } : {}), ...(options.headers || {}) },
  });
  if (response.status === 401 && path !== "/api/v2/auth/session" && !authRetried) {
    const token = await requestAuthToken();
    if (!token) throw new Error("需要有效的会议服务访问令牌");
    const login = await fetch("/api/v2/auth/session", {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ token }),
    });
    if (!login.ok) throw new Error("访问令牌无效");
    return requestJson(path, { ...options, authRetried: true });
  }
  const text = await response.text();
  let payload = null;
  try { payload = text ? JSON.parse(text) : null; } catch { payload = { detail: text }; }
  if (!response.ok) throw new Error(payload?.detail || `请求失败（${response.status}）`);
  return payload;
}

function setNotice(message, kind = "info") {
  dom.notice.hidden = !message;
  dom.notice.className = `notice ${kind}`;
  dom.notice.textContent = message || "";
}

function setSystemStatus(message, good = false) {
  dom.systemStatus.textContent = message;
  dom.statusDot.classList.toggle("ready", good);
  dom.statusDot.classList.toggle("error", !good);
  if (dom.startMeeting) dom.startMeeting.disabled = !good;
  if (dom.newMeeting) dom.newMeeting.disabled = !good;
}

function setConnection(message, kind = "neutral") {
  dom.connectionBadge.textContent = message;
  dom.connectionBadge.className = `badge ${kind}`;
}

function setVolumeThreshold(value, propagate = true) {
  const threshold = Math.max(0, Math.min(30, Number(value) || 0));
  state.volumeThresholdPercent = Math.round(threshold * 10) / 10;
  state.meetingSettings.volume_threshold_percent = state.volumeThresholdPercent;
  if (dom.volumeThreshold) dom.volumeThreshold.value = String(state.volumeThresholdPercent);
  if (dom.volumeThresholdValue) dom.volumeThresholdValue.value = state.volumeThresholdPercent.toFixed(1);
  if (dom.volumeThresholdSummary) dom.volumeThresholdSummary.textContent = `${state.volumeThresholdPercent.toFixed(1)}%`;
  renderMicrophoneLevel(state.microphoneLevelPercent, Boolean(state.stream));
  if (!propagate) return;
  state.audioNode?.port.postMessage({ type: "volume_threshold", percent: state.volumeThresholdPercent });
  if (state.meeting) state.meeting.volume_threshold_percent = state.volumeThresholdPercent;
  if (state.ws?.readyState === WebSocket.OPEN) {
    try { state.ws.send(JSON.stringify({ type: "audio_threshold", percent: state.volumeThresholdPercent })); } catch {}
  }
}

function clampMeetingValue(field, value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return Number(RECOMMENDED_MEETING_SETTINGS[field.key]);
  const step = Number(field.step) || 1;
  const stepped = Math.round((numeric - field.min) / step) * step + field.min;
  const precision = String(step).includes(".") ? String(step).split(".")[1].length : 0;
  return Number(Math.max(field.min, Math.min(field.max, stepped)).toFixed(precision));
}

function setMeetingField(field, value, { updateState = true } = {}) {
  const normalized = clampMeetingValue(field, value);
  const slider = $(`#${field.slider}`);
  const input = $(`#${field.input}`);
  if (slider) slider.value = String(normalized);
  if (input) input.value = String(normalized);
  if (updateState) state.meetingSettings[field.key] = normalized;
  if (field.key === "volume_threshold_percent") {
    state.volumeThresholdPercent = normalized;
    if (dom.volumeThresholdSummary) dom.volumeThresholdSummary.textContent = `${normalized.toFixed(1)}%`;
    renderMicrophoneLevel(state.microphoneLevelPercent, Boolean(state.stream));
  }
}

function renderMeetingSettings(settings = state.meetingSettings, { updateState = true } = {}) {
  const values = { ...RECOMMENDED_MEETING_SETTINGS, ...(settings || {}) };
  if (updateState) {
    state.meetingSettings = { ...state.meetingSettings, ...values };
  }
  for (const field of MEETING_NUMBER_FIELDS) setMeetingField(field, values[field.key], { updateState });
  if (updateState) setVolumeThreshold(values.volume_threshold_percent, false);
  const toggles = {
    keep_audio: "settingKeepAudio",
  };
  for (const [key, id] of Object.entries(toggles)) {
    const input = $(`#${id}`);
    if (input) input.checked = Boolean(values[key]);
    if (updateState) state.meetingSettings[key] = Boolean(values[key]);
  }
  const editable = state.meeting?.recording_state === "created";
  dom.openAsrSettings.disabled = !state.meeting;
  dom.openAsrSettings.title = editable ? "调整识别设置" : "识别设置只能在录音开始前调整";
  for (const field of MEETING_NUMBER_FIELDS) {
    const slider = $(`#${field.slider}`);
    const input = $(`#${field.input}`);
    if (slider) slider.disabled = !editable && field.key !== "volume_threshold_percent";
    if (input) input.disabled = !editable && field.key !== "volume_threshold_percent";
  }
  for (const id of ["settingKeepAudio"]) {
    const input = $(`#${id}`);
    if (input) input.disabled = !editable;
  }
  // The audio threshold is live even though other meeting settings are locked
  // at the start of recording.
  const audioThresholdEditable = ["created", "starting", "recording"].includes(state.meeting?.recording_state);
  if (dom.volumeThreshold) dom.volumeThreshold.disabled = !audioThresholdEditable;
  if (dom.volumeThresholdValue) dom.volumeThresholdValue.disabled = !audioThresholdEditable;
  dom.resetAsrSettings.disabled = !editable;
  dom.saveAsrSettings.disabled = !editable;
}

function renderAsrSettings(settings = state.asrSettings) {
  renderMeetingSettings({ ...state.meetingSettings, ...RECOMMENDED_ASR_SETTINGS, ...(settings || {}) });
}

function moveAudioSettingsIntoDialog() {
  if (!dom.settingsAudioSection) return;
  const nodes = [
    document.querySelector('label[for="inputDevice"]'),
    document.querySelector(".device-row"),
    document.querySelector('label[for="volumeThreshold"]'),
    dom.thresholdMeter,
    document.querySelector(".threshold-feedback"),
    document.querySelector("#volumeThresholdHint"),
  ];
  for (const node of nodes) {
    if (node) dom.settingsAudioSection.append(node);
  }
}

function readAsrSettings() {
  return Object.fromEntries(ASR_SETTING_FIELDS.map((field) => [field.key, Number($(`#${field.input}`).value)]));
}

function readMeetingSettings() {
  const values = { ...state.meetingSettings };
  for (const field of MEETING_NUMBER_FIELDS) values[field.key] = clampMeetingValue(field, $(`#${field.input}`)?.value);
  values.keep_audio = Boolean($("#settingKeepAudio")?.checked);
  return values;
}

function resetMeetingSettings() {
  if (!state.meeting || state.meeting.recording_state !== "created") return;
  renderMeetingSettings(RECOMMENDED_MEETING_SETTINGS);
  if (dom.asrSettingsNotice) dom.asrSettingsNotice.textContent = "已恢复推荐值，点击“保存设置”后生效。";
}

function openAsrSettings() {
  if (!state.meeting) return;
  renderMeetingSettings(state.meetingSettings);
  dom.asrSettingsNotice.textContent = state.meeting.recording_state === "created"
    ? "推荐值适合中文会议和实时识别；音频输入调整会立即应用，识别参数需要点击“保存设置”。"
    : "本次会议已经开始或结束，识别设置已锁定。请新建会议后再调整。";
  dom.asrSettingsDialog.showModal();
}

async function saveAsrSettings() {
  if (!state.meeting || state.meeting.recording_state !== "created") return;
  dom.saveAsrSettings.disabled = true;
  try {
    const snapshot = await requestJson(`/api/v2/meetings/${encodeURIComponent(state.meeting.id)}/settings`, {
      method: "PATCH",
      body: JSON.stringify({ settings: readMeetingSettings(), asr_settings: readAsrSettings() }),
    });
    applySnapshot(snapshot, false);
    dom.asrSettingsDialog.close();
    setNotice("识别设置已保存，开始录音后生效。", "info");
  } catch (error) {
    dom.asrSettingsNotice.textContent = error.message || "识别设置保存失败，请重试。";
    dom.saveAsrSettings.disabled = false;
  }
}

function renderMicrophoneLevel(levelPercent, live = true) {
  const level = Math.max(0, Math.min(100, Number(levelPercent) || 0));
  const meterMax = MICROPHONE_METER_MAX_PERCENT;
  const position = Math.min(100, level / meterMax * 100);
  state.microphoneLevelPercent = level;
  dom.microphoneLevelFill.style.width = `${position}%`;
  dom.microphoneLevelMarker.style.left = `${position}%`;
  dom.thresholdMeter.classList.toggle("live", live);
  dom.microphoneLevelValue.textContent = live ? `当前音量 ${level.toFixed(1)}%` : "当前音量 --";
  const passing = live && level >= state.volumeThresholdPercent;
  dom.thresholdMeter.classList.toggle("filtered", live && !passing);
  dom.microphoneLevelStatus.className = passing ? "passing" : live ? "filtered" : "";
  dom.microphoneLevelStatus.textContent = !live ? "等待麦克风" : passing ? "声音会被保留" : "低于阈值，将被过滤";
}

function queueMicrophoneLevel(level) {
  state.pendingMicrophoneLevel = Math.max(0, Math.min(1, Number(level) || 0));
  if (state.microphoneLevelFrame != null) return;
  state.microphoneLevelFrame = window.requestAnimationFrame(renderQueuedMicrophoneLevel);
}

function renderQueuedMicrophoneLevel() {
  const now = performance.now();
  state.microphoneLevelFrame = null;
  const target = state.pendingMicrophoneLevel;
  const current = Math.max(0, Math.min(1, state.microphoneLevelPercent / 100));
  const delta = target - current;
  const elapsed = state.microphoneLevelFrameAt
    ? Math.min(100, Math.max(1, now - state.microphoneLevelFrameAt))
    : 16.7;
  state.microphoneLevelFrameAt = now;
  const timeConstant = delta >= 0 ? MICROPHONE_LEVEL_ATTACK_MS : MICROPHONE_LEVEL_RELEASE_MS;
  const factor = 1 - Math.exp(-elapsed / timeConstant);
  const value = Math.abs(delta) <= MICROPHONE_LEVEL_SETTLE_EPSILON ? target : current + delta * factor;
  renderMicrophoneLevel(value * 100, true);
  dom.levelBar.style.width = `${Math.round(value * 100)}%`;
  dom.levelText.textContent = value > 0.03 ? "正在采集声音" : "等待说话";
  if (state.meeting) state.meeting.audio_level = value;
  if (Math.abs(target - value) > MICROPHONE_LEVEL_SETTLE_EPSILON && state.stream) {
    state.microphoneLevelFrame = window.requestAnimationFrame(renderQueuedMicrophoneLevel);
  }
}

function formatDate(value) {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleString("zh-CN", { month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" });
}

function formatTime(seconds) {
  const total = Math.max(0, Math.floor(Number(seconds) || 0));
  const hours = String(Math.floor(total / 3600)).padStart(2, "0");
  const minutes = String(Math.floor((total % 3600) / 60)).padStart(2, "0");
  const secs = String(total % 60).padStart(2, "0");
  return `${hours}:${minutes}:${secs}`;
}

function stateText(value) {
  return String(value || "").replaceAll("_", " ");
}

function renderMeetings() {
  const query = dom.search.value.trim().toLowerCase();
  const meetings = state.meetings.filter((meeting) => !query || `${meeting.title} ${meeting.id}`.toLowerCase().includes(query));
  dom.meetingList.replaceChildren();
  if (!meetings.length) {
    const empty = document.createElement("div");
    empty.className = "sidebar-empty";
    empty.textContent = query ? "没有匹配的会议" : "还没有会议记录";
    dom.meetingList.append(empty);
    return;
  }
  for (const meeting of meetings) {
    const entry = document.createElement("article");
    entry.className = `meeting-entry ${state.meeting?.id === meeting.id ? "active" : ""}`;
    entry.dataset.meetingId = meeting.id;

    const selectButton = document.createElement("button");
    selectButton.type = "button";
    selectButton.className = "meeting-entry-select";
    selectButton.setAttribute("aria-label", `打开会议：${meeting.title || "未命名会议"}`);
    selectButton.innerHTML = `<span class="meeting-entry-title"></span><span class="meeting-entry-meta"></span><span class="meeting-entry-state"></span>`;
    selectButton.querySelector(".meeting-entry-title").textContent = meeting.title || "未命名会议";
    selectButton.querySelector(".meeting-entry-meta").textContent = `${formatDate(meeting.started_at)} · ${formatTime(meeting.duration_seconds)}`;
    const stateNode = selectButton.querySelector(".meeting-entry-state");
    stateNode.textContent = recordingLabels[meeting.recording_state] || stateText(meeting.recording_state);
    stateNode.classList.toggle("live", meeting.recording_state === "recording");

    const actions = document.createElement("div");
    actions.className = "meeting-entry-actions";
    const renameButton = document.createElement("button");
    renameButton.type = "button";
    renameButton.className = "meeting-action-button rename";
    renameButton.setAttribute("aria-label", `重命名会议：${meeting.title || "未命名会议"}`);
    renameButton.title = "重命名会议";
    renameButton.innerHTML = '<span class="meeting-action-icon" aria-hidden="true">✎</span>';
    renameButton.addEventListener("click", () => openRenameDialog(meeting));

    const deleteButton = document.createElement("button");
    deleteButton.type = "button";
    deleteButton.className = "meeting-action-button delete";
    deleteButton.setAttribute("aria-label", `删除会议：${meeting.title || "未命名会议"}`);
    deleteButton.title = "删除会议";
    deleteButton.textContent = "×";
    deleteButton.addEventListener("click", () => deleteMeeting(meeting.id, meeting.title));

    selectButton.addEventListener("click", () => selectMeeting(meeting.id));
    actions.append(renameButton, deleteButton);
    entry.append(selectButton, actions);
    dom.meetingList.append(entry);
  }
}

function clearTranscript() {
  state.transcriptLoadToken += 1;
  clearDraft();
  state.transcript.clear();
  state.transcriptNodes.clear();
  state.loadedTranscriptRevision = 0;
  state.transcriptLoadedFor = null;
  dom.transcriptList.replaceChildren();
  dom.transcriptList.append(dom.transcriptEmpty);
  dom.transcriptEmpty.hidden = false;
}

function clearDraft() {
  state.draft = null;
  state.draftNode?.remove();
  state.draftNode = null;
  if (dom.transcriptEmpty && !state.transcript.size) dom.transcriptEmpty.hidden = false;
}

function draftMatchesUtterance(utterance) {
  if (!state.draft || !utterance) return false;
  const revision = String(state.draft.revision ?? "");
  const sourceSegmentId = String(utterance.source_segment_id ?? "");
  const segmentId = String(utterance.segment_id ?? "");
  return revision !== "" && (revision === sourceSegmentId || segmentId.startsWith(`${revision}:`));
}

function upsertUtterance(utterance, transcript = state.transcript) {
  if (!utterance || !utterance.segment_id) return false;
  if (utterance.deleted) {
    const current = transcript.get(utterance.segment_id);
    const revision = Number(utterance.revision || 1);
    const sourceRevision = Number(utterance.source_revision || 1);
    const currentRevision = Number(current?.revision || 1);
    const currentSourceRevision = Number(current?.source_revision || 1);
    if (!current || revision > currentRevision || (revision === currentRevision && sourceRevision >= currentSourceRevision)) {
      transcript.delete(utterance.segment_id);
      return Boolean(current);
    }
    return false;
  }
  const previous = transcript.get(utterance.segment_id);
  const revision = Number(utterance.revision || 1);
  const sourceRevision = Number(utterance.source_revision || 1);
  const previousRevision = Number(previous?.revision || 1);
  const previousSourceRevision = Number(previous?.source_revision || 1);
  const newer = !previous
    || revision > previousRevision
    || (revision === previousRevision && sourceRevision > previousSourceRevision)
    || (revision === previousRevision && sourceRevision === previousSourceRevision && Boolean(utterance.closed) && !Boolean(previous.closed));
  if (newer) {
    transcript.set(utterance.segment_id, utterance);
    return !previous || JSON.stringify(previous) !== JSON.stringify(utterance);
  }
  return false;
}

function createTranscriptNode() {
  const article = document.createElement("article");
  article.className = "transcript-item transcript-message";
  article.innerHTML = `<div class="transcript-meta"><span class="paragraph-avatar" aria-hidden="true"></span><span class="language-label"></span><time></time><span class="language-tag"></span></div><div class="transcript-bubbles"><p class="transcript-bubble original-bubble"></p><div class="transcript-bubble translation-bubble" hidden><span class="translation-label">中文翻译</span><b></b><span class="translation-pending" hidden><i></i><i></i><i></i></span></div></div>`;
  return article;
}

function updateTranscriptNode(article, item) {
  const lang = item.language || "unknown";
  article.dataset.segmentId = item.segment_id;
  article.dataset.language = lang;
  article.dataset.variant = item.speech_variant || "unknown";
  article.querySelector("time").textContent = `${formatTime(item.start).slice(3)} – ${formatTime(item.end).slice(3)}`;
  article.querySelector(".language-label").textContent = paragraphLanguageLabel(item);
  article.querySelector(".language-tag").textContent = lang.toUpperCase();
  article.querySelector(".original-bubble").textContent = item.text || "";
  const translation = article.querySelector(".translation-bubble");
  const translated = Boolean(item.translation_zh && item.translation_status !== "not_needed");
  const pending = !translated && ["pending", "streaming"].includes(item.translation_status);
  const unsupported = !translated && item.translation_status === "unsupported";
  const failed = !translated && item.translation_status === "failed";
  translation.hidden = (!translated && !pending && !failed) || unsupported;
  translation.classList.toggle("is-pending", pending);
  translation.querySelector("b").textContent = translated ? item.translation_zh : "";
  translation.querySelector(".translation-label").textContent = pending ? "中文翻译" : "中文翻译";
  translation.querySelector(".translation-pending").hidden = !pending;
}

function createDraftNode() {
  const article = document.createElement("article");
  article.className = "transcript-item transcript-message transcript-draft";
  article.innerHTML = `<div class="transcript-meta"><span class="paragraph-avatar draft-avatar" aria-hidden="true">✦</span><span class="language-label">实时识别</span><time></time><span class="language-tag"></span></div><div class="transcript-bubbles"><p class="transcript-bubble original-bubble"><span class="draft-text"></span><span class="streaming-cursor" aria-hidden="true"></span></p><div class="draft-status"><span class="status-pulse"></span><span>正在识别</span></div></div>`;
  return article;
}

function renderDraft(draft) {
  if (!draft?.text) {
    clearDraft();
    return;
  }
  if (!state.draftNode) {
    state.draftNode = createDraftNode();
    dom.transcriptList.append(state.draftNode);
  }
  state.draft = { ...draft };
  state.draftNode.querySelector(".draft-text").textContent = draft.text;
  state.draftNode.querySelector("time").textContent = draft.start != null ? formatTime(draft.start).slice(3) : "实时";
  state.draftNode.querySelector(".language-tag").textContent = draft.language ? String(draft.language).toUpperCase() : "";
  dom.transcriptEmpty.hidden = true;
  updateTranscriptViewport(true);
}

function updateTranscriptViewport(autoScroll, itemCount = state.transcript.size) {
  const hasDraft = Boolean(state.draftNode);
  dom.transcriptEmpty.hidden = itemCount > 0 || hasDraft;
  dom.utteranceCount.textContent = `${itemCount} 个段落`;
  const shouldScroll = autoScroll || state.transcriptNearBottom;
  if (shouldScroll) requestAnimationFrame(() => { dom.transcriptList.scrollTop = dom.transcriptList.scrollHeight; });
  dom.jumpLatest.hidden = shouldScroll || !itemCount;
}

function renderTranscriptItem(segmentId, autoScroll = false) {
  const item = state.transcript.get(segmentId);
  if (!item) return;
  let article = state.transcriptNodes.get(segmentId);
  if (!article) {
    article = createTranscriptNode();
    state.transcriptNodes.set(segmentId, article);
    const following = [...state.transcript.values()]
      .sort((a, b) => (a.start || 0) - (b.start || 0))
      .find((candidate) => (candidate.start || 0) > (item.start || 0) && state.transcriptNodes.has(candidate.segment_id));
    dom.transcriptList.insertBefore(article, following ? state.transcriptNodes.get(following.segment_id) : null);
  }
  updateTranscriptNode(article, item);
  updateTranscriptViewport(autoScroll);
}

function removeTranscriptItem(segmentId) {
  state.transcriptNodes.get(segmentId)?.remove();
  state.transcriptNodes.delete(segmentId);
  if (!state.transcript.size && !state.draftNode && !dom.transcriptEmpty.isConnected) dom.transcriptList.append(dom.transcriptEmpty);
  updateTranscriptViewport(false);
}

function renderTranscript(autoScroll = false) {
  clearDraft();
  const items = [...state.transcript.values()].sort((a, b) => (a.start || 0) - (b.start || 0));
  state.transcriptNodes.clear();
  dom.transcriptList.replaceChildren();
  if (!items.length) dom.transcriptList.append(dom.transcriptEmpty);
  for (const item of items) {
    const article = createTranscriptNode();
    updateTranscriptNode(article, item);
    state.transcriptNodes.set(item.segment_id, article);
    dom.transcriptList.append(article);
  }
  updateTranscriptViewport(autoScroll, items.length);
}

function renderRefinedTranscript(items, summaryState = "idle", markdown = "") {
  const values = Array.isArray(items) ? items : [];
  const markdownValue = String(markdown || "").trim();
  if (dom.refinedBadge) {
    dom.refinedBadge.textContent = refinedLabels[summaryState] || stateText(summaryState);
    dom.refinedBadge.className = `badge ${summaryState === "complete" ? "success" : summaryState === "error" ? "danger" : summaryState === "running" ? "live" : "neutral"}`;
  }
  if (!dom.refinedTranscriptText) return;
  dom.refinedTranscriptText.replaceChildren();
  dom.refinedTranscriptText.classList.toggle("empty-result", !values.length && !markdownValue);
  if (markdownValue) {
    dom.refinedTranscriptText.innerHTML = markdownToHtml(markdownValue);
    return;
  }
  if (!values.length) {
    dom.refinedTranscriptText.textContent = summaryState === "running" || summaryState === "queued"
      ? "智能体正在处理录音，请稍候。"
      : summaryState === "error"
        ? "精修转写生成失败，请重试。"
        : "会议停止并完成翻译后，点击“生成三段结果”查看精修转写。";
    return;
  }
  const list = document.createElement("div");
  list.className = "refined-list";
  for (const item of values) {
    const row = document.createElement("article");
    row.className = "refined-item";
    const start = Number(item.start);
    const end = Number(item.end);
    const time = Number.isFinite(start) && Number.isFinite(end)
      ? `${formatTime(start).slice(3)} – ${formatTime(end).slice(3)}`
      : "时间待确认";
    row.innerHTML = `<div class="refined-meta"><strong></strong><span class="refined-language"></span><time></time></div><div class="refined-source"><span>原文</span><p></p></div><div class="refined-translation"><span>中文翻译</span><p></p></div>`;
    row.querySelector("strong").textContent = item[["spea", "ker"].join("")] || "待确认";
    row.querySelector(".refined-language").textContent = String(item.language || "unknown").toUpperCase();
    row.querySelector("time").textContent = time;
    row.querySelector(".refined-source p").textContent = item.original || "";
    row.querySelector(".refined-translation p").textContent = item.translation_zh || "";
    list.append(row);
  }
  dom.refinedTranscriptText.append(list);
}

function renderSummary(summary, summaryState) {
  const value = String(summary || "").trim();
  const preprocessingReady = state.meeting?.recording_state === "complete" && state.meeting?.translation_pending !== true;
  dom.summaryBadge.textContent = summaryLabels[summaryState] || stateText(summaryState);
  dom.summaryBadge.className = `badge ${summaryState === "complete" ? "success" : summaryState === "error" ? "danger" : summaryState === "running" ? "live" : "neutral"}`;
  dom.summaryText.classList.toggle("empty-result", !value);
  dom.summaryText.innerHTML = value
    ? markdownToHtml(value)
      : preprocessingReady
      ? "转写和翻译已经完成，请点击下方“生成三段结果”按钮。"
      : "停止会议并完成翻译后，可以手动生成三段结果。";
  dom.retrySummary.hidden = !(preprocessingReady && ["idle", "error", "complete"].includes(summaryState));
  dom.retrySummary.disabled = !preprocessingReady || summaryState === "running";
  dom.retrySummary.textContent = summaryState === "complete" ? "重新生成三段结果" : "生成三段结果";
  dom.downloadSummary.hidden = !value || !state.meeting?.files?.includes("meeting_minutes.md");
}

function renderTodo(todo, todoState, markdown = "") {
  dom.todoBadge.textContent = todoLabels[todoState] || stateText(todoState);
  dom.todoBadge.className = `badge ${todoState === "complete" ? "success" : todoState === "error" ? "danger" : "neutral"}`;
  const items = todo?.items || [];
  const markdownValue = String(markdown || "").trim();
  dom.todoText.replaceChildren();
  dom.todoText.classList.toggle("empty-result", !todo && !markdownValue);
  if (markdownValue) {
    dom.todoText.innerHTML = markdownToHtml(markdownValue);
  } else if (!todo) {
    dom.todoText.textContent = "会议停止并完成翻译后，请点击“生成三段结果”。";
  } else if (!items.length) {
    dom.todoText.textContent = "没有提取到明确行动项。";
  } else {
    const list = document.createElement("div");
    list.className = "todo-list";
    for (const item of items) {
      const row = document.createElement("article");
      row.className = "todo-item";
      row.innerHTML = `<div class="todo-check" aria-hidden="true">✓</div><div class="todo-body"><strong></strong><div class="todo-meta"><span class="todo-chip owner"></span><span class="todo-chip due"></span><span class="todo-chip priority"></span></div><small class="todo-evidence"></small></div>`;
      row.querySelector("strong").textContent = item.task || "未命名任务";
      row.querySelector(".owner").textContent = item.owner || "负责人待确认";
      row.querySelector(".due").textContent = item.due_date || "截止时间待确认";
      row.querySelector(".priority").textContent = item.priority || "待确认";
      const evidence = row.querySelector(".todo-evidence");
      evidence.textContent = item.evidence || "";
      evidence.hidden = !item.evidence;
      list.append(row);
    }
    dom.todoText.append(list);
  }
  dom.downloadTodo.hidden = !todo || !state.meeting?.files?.includes("todo_list.json");
}

function renderFiles(files = []) {
  dom.fileLinks.replaceChildren();
  const visible = files.filter((name) => !name.startsWith("original_") || ["original_zh.md", "original_en.md", "original_de.md"].includes(name));
  for (const name of visible) {
    const link = document.createElement("a");
    link.href = `/api/v2/meetings/${encodeURIComponent(state.meeting.id)}/files/${name.split("/").map(encodeURIComponent).join("/")}`;
    link.rel = "noopener";
    link.download = name.split("/").pop() || name;
    link.textContent = name;
    dom.fileLinks.append(link);
  }
  dom.filesCard.hidden = !visible.length;
}

function applySnapshot(snapshot, replace = false) {
  if (!snapshot) return;
  if (state.meeting?.id === snapshot.id && (
    Number(snapshot.snapshot_revision || 0) < Number(state.meeting.snapshot_revision || 0)
    || Number(snapshot.transcript_revision || 0) < Number(state.meeting.transcript_revision || 0)
  )) return;
  const changed = state.meeting?.id !== snapshot.id;
  state.meeting = { ...(state.meeting || {}), ...snapshot };
  if (changed || replace || state.meeting.recording_state !== "recording") clearDraft();
  if (changed && snapshot.volume_threshold_percent != null) setVolumeThreshold(snapshot.volume_threshold_percent, false);
  if (snapshot.asr_settings) {
    state.asrSettings = { ...RECOMMENDED_ASR_SETTINGS, ...snapshot.asr_settings };
  }
  if (snapshot.meeting_settings) {
    state.meetingSettings = { ...RECOMMENDED_MEETING_SETTINGS, ...snapshot.meeting_settings };
  } else {
    state.meetingSettings = {
      ...RECOMMENDED_MEETING_SETTINGS,
      ...state.meetingSettings,
      ...state.asrSettings,
      ...(snapshot.volume_threshold_percent != null ? { volume_threshold_percent: snapshot.volume_threshold_percent } : {}),
    };
  }
  const isCreated = state.meeting.recording_state === "created";
  const canAdjustAudio = ["created", "starting", "recording"].includes(state.meeting.recording_state);
  const canStop = ["starting", "recording"].includes(state.meeting.recording_state);
  dom.volumeThreshold.disabled = !canAdjustAudio;
  if (!dom.asrSettingsDialog.open) {
    renderMeetingSettings(state.meetingSettings);
  }
  const meetingIndex = state.meetings.findIndex((meeting) => meeting.id === snapshot.id);
  if (meetingIndex >= 0) state.meetings[meetingIndex] = { ...state.meetings[meetingIndex], ...snapshot };
  else state.meetings.unshift(snapshot);
  let transcriptChanged = changed || replace;
  if (transcriptChanged) {
    clearTranscript();
  }
  for (const item of snapshot.paragraphs || snapshot.recent_utterances || []) transcriptChanged = upsertUtterance(item) || transcriptChanged;
  dom.pageTitle.textContent = state.meeting.title || "未命名会议";
  dom.pageSubtitle.textContent = state.meeting.recording_state === "recording"
    ? "实时保留原文；英文和德文句子会异步补充中文翻译。"
    : isCreated ? "会议已创建，可以先调整输入设备和背景声过滤，再手动开始录音。" : "这场会议的录音、原文、纪要和行动项已保存到本机。";
  dom.welcome.hidden = true;
  dom.meetingPanel.hidden = false;
  dom.recordingState.textContent = recordingLabels[state.meeting.recording_state] || stateText(state.meeting.recording_state);
  dom.recordingIndicator.classList.toggle("active", state.meeting.recording_state === "recording");
  dom.startRecordingButton.hidden = !isCreated;
  dom.startRecordingButton.disabled = !isCreated;
  dom.recordButton.hidden = !canStop;
  dom.recordButton.disabled = !canStop;
  dom.recordButton.setAttribute("aria-label", "停止会议");
  dom.recordingHint.textContent = state.meeting.error || (
    isCreated ? "请先确认输入设备和背景声过滤设置，再点击“开始录音”。"
      : state.meeting.recording_state === "recording" ? "正在接收麦克风音频，连续语音会合并为段落。"
        : "录音已经结束，翻译完成后请点击“生成三段结果”。"
  );
  dom.levelBar.style.width = `${Math.round((state.meeting.audio_level || 0) * 100)}%`;
  if (transcriptChanged) renderTranscript();
  renderRefinedTranscript(state.meeting?.refined_transcript, state.meeting?.summary_state, state.meeting?.refined_transcript_markdown);
  renderSummary(state.meeting.summary, state.meeting.summary_state);
  renderTodo(state.meeting.todo, state.meeting.todo_state, state.meeting.todo_markdown);
  renderFiles(state.meeting.files || []);
  renderMeetings();
  updateTimer();
}

function updateTimer() {
  if (!state.meeting) {
    dom.timer.textContent = "00:00:00";
    return;
  }
  const seconds = state.meeting.recording_state === "recording" && state.meeting.started_at
    ? (Date.now() - new Date(state.meeting.started_at).getTime()) / 1000
    : state.meeting.duration_seconds;
  dom.timer.textContent = formatTime(seconds);
}

async function loadMeetings() {
  state.meetings = await requestJson("/api/v2/meetings");
  renderMeetings();
}

async function loadFullTranscript(id, expectedGeneration = null) {
  const loadToken = ++state.transcriptLoadToken;
  let offset = 0;
  const limit = 1000;
  const transcript = new Map();
  while (loadToken === state.transcriptLoadToken && state.meeting?.id === id && (expectedGeneration == null || state.streamGeneration === expectedGeneration)) {
    const page = await requestJson(`/api/v2/meetings/${encodeURIComponent(id)}/transcript?offset=${offset}&limit=${limit}`);
    const items = page.paragraphs || page.items || [];
    for (const item of items) upsertUtterance(item, transcript);
    offset += items.length;
    if (!page.has_more || !items.length) break;
  }
  if (loadToken === state.transcriptLoadToken && state.meeting?.id === id && (expectedGeneration == null || state.streamGeneration === expectedGeneration)) {
    for (const item of state.transcript.values()) upsertUtterance(item, transcript);
    state.transcript = transcript;
    state.transcriptLoadedFor = id;
    state.loadedTranscriptRevision = Number(state.meeting.transcript_revision || 0);
    renderTranscript();
  }
}

async function refreshCurrentMeeting() {
  if (!state.meeting) return;
  try {
    const previousRevision = Number(state.meeting.snapshot_revision || 0);
    const previousTranscriptRevision = Number(state.loadedTranscriptRevision || 0);
    const snapshot = await requestJson(`/api/v2/meetings/${encodeURIComponent(state.meeting.id)}`);
    if (state.meeting?.id === snapshot.id) {
      applySnapshot(snapshot, false);
      if (
        (snapshot.recording_state === "complete" && state.transcriptLoadedFor !== snapshot.id)
        || Number(snapshot.transcript_revision || 0) > previousTranscriptRevision
        || (Number(snapshot.snapshot_revision || 0) > previousRevision && state.transcriptLoadedFor !== snapshot.id)
      ) {
        // Post-processing can replace or delete utterances. Replacing the map
        // from the paginated projection removes stale rows that a recent-only
        // snapshot cannot describe.
        await loadFullTranscript(snapshot.id);
      }
    }
  } catch {}
}

async function selectMeeting(id) {
  if (state.meeting?.recording_state === "recording" && state.meeting.id !== id) {
    setNotice("当前会议正在录音，请先结束当前会议。", "warning");
    return;
  }
  try {
    const snapshot = await requestJson(`/api/v2/meetings/${encodeURIComponent(id)}`);
    const oldId = state.meeting?.id;
    if (oldId !== id) {
      clearDraft();
      if (state.stream) stopAudioCapture();
      closeStream(false);
    }
    state.meeting = null;
    applySnapshot(snapshot, true);
    await loadFullTranscript(snapshot.id);
    // The stream transports live audio and realtime events. Once recording has
    // finished, post-processing progress is delivered by the existing snapshot
    // polling instead of keeping an unnecessary websocket alive.
    if (["recording", "starting", "finalizing"].includes(snapshot.recording_state)) {
      await connectStream(snapshot.id);
    } else if (snapshot.recording_state === "created") {
      await startMicrophonePreview();
    }
    setConnection(
      snapshot.recording_state === "created" ? "等待开始" : snapshot.recording_state === "recording" ? "等待连接" : "本地已保存",
      snapshot.recording_state === "recording" ? "warning" : "neutral",
    );
  } catch (error) { setNotice(error.message, "error"); }
}

async function checkHealth() {
  let healthResponseReceived = false;
  try {
    const health = await requestJson("/api/v2/health");
    healthResponseReceived = true;
    setSystemStatus(health.status === "ready" ? "本机服务已就绪" : health.message || "模型正在加载", health.status === "ready");
    setConnection(health.status === "ready" ? "服务就绪" : "模型加载中", health.status === "ready" ? "success" : "warning");
    if (health.status !== "ready") throw new Error(health.message || "模型尚未就绪");
    return health;
  } catch (error) {
    if (healthResponseReceived) {
      setSystemStatus(error.message || "模型尚未就绪", false);
      setConnection("模型未就绪", "warning");
      throw error;
    }
    setSystemStatus("本机服务未连接", false);
    setConnection("服务不可用", "danger");
    throw error;
  }
}

function inputDeviceMenuIsOpen() {
  return Boolean(state.inputDevicePickerOpen);
}

function inputDeviceOptionButtons() {
  return dom.inputDeviceOptions ? [...dom.inputDeviceOptions.querySelectorAll('[role="option"]')] : [];
}

function updateInputDevicePicker() {
  if (!dom.inputDevice || !dom.inputDeviceTrigger || !dom.inputDeviceOptions) return;
  const selected = dom.inputDevice.value;
  const options = document.createDocumentFragment();
  dom.inputDeviceTrigger.textContent = dom.inputDevice.selectedOptions[0]?.textContent || "系统默认设备";
  for (const option of dom.inputDevice.options) {
    const item = document.createElement("button");
    item.type = "button";
    item.className = "device-picker-option";
    item.setAttribute("role", "option");
    item.setAttribute("aria-selected", String(option.value === selected));
    item.dataset.value = option.value;
    item.textContent = option.textContent;
    options.append(item);
  }
  dom.inputDeviceOptions.replaceChildren(options);
}

function focusInputDeviceOption(index) {
  const options = inputDeviceOptionButtons();
  if (!options.length) return;
  const nextIndex = Math.max(0, Math.min(options.length - 1, index));
  options[nextIndex].focus();
  options[nextIndex].scrollIntoView({ block: "nearest" });
}

function openInputDevicePicker() {
  if (!dom.inputDeviceTrigger || !dom.inputDeviceOptions) return;
  flushPendingInputDevices();
  state.inputDevicePickerOpen = true;
  dom.inputDeviceTrigger.setAttribute("aria-expanded", "true");
  dom.inputDeviceOptions.hidden = false;
}

function closeInputDevicePicker({ focusTrigger = false } = {}) {
  if (!dom.inputDeviceTrigger || !dom.inputDeviceOptions) return;
  state.inputDevicePickerOpen = false;
  dom.inputDeviceTrigger.setAttribute("aria-expanded", "false");
  dom.inputDeviceOptions.hidden = true;
  flushPendingInputDevices();
  if (focusTrigger) dom.inputDeviceTrigger.focus();
}

function chooseInputDevice(value) {
  const option = [...dom.inputDevice.options].find((candidate) => candidate.value === value);
  if (!option) return;
  const changed = dom.inputDevice.value !== option.value;
  dom.inputDevice.value = option.value;
  updateInputDevicePicker();
  closeInputDevicePicker({ focusTrigger: true });
  if (changed) dom.inputDevice.dispatchEvent(new Event("change", { bubbles: true }));
}

function handleInputDeviceTriggerKeydown(event) {
  const options = inputDeviceOptionButtons();
  const selectedIndex = Math.max(0, options.findIndex((option) => option.getAttribute("aria-selected") === "true"));
  if (["Enter", " ", "ArrowDown"].includes(event.key)) {
    event.preventDefault();
    openInputDevicePicker();
    if (event.key === "ArrowDown") focusInputDeviceOption(selectedIndex);
  } else if (event.key === "ArrowUp") {
    event.preventDefault();
    openInputDevicePicker();
    focusInputDeviceOption(selectedIndex);
  } else if (event.key === "Escape" && state.inputDevicePickerOpen) {
    event.preventDefault();
    closeInputDevicePicker({ focusTrigger: true });
  }
}

function handleInputDeviceOptionsKeydown(event) {
  const current = event.target.closest?.('[role="option"]');
  const options = inputDeviceOptionButtons();
  const index = Math.max(0, options.indexOf(current));
  if (event.key === "ArrowDown") {
    event.preventDefault();
    focusInputDeviceOption(index + 1);
  } else if (event.key === "ArrowUp") {
    event.preventDefault();
    focusInputDeviceOption(index - 1);
  } else if (event.key === "Home") {
    event.preventDefault();
    focusInputDeviceOption(0);
  } else if (event.key === "End") {
    event.preventDefault();
    focusInputDeviceOption(options.length - 1);
  } else if (["Enter", " "].includes(event.key) && current) {
    event.preventDefault();
    chooseInputDevice(current.dataset.value || "");
  } else if (event.key === "Escape") {
    event.preventDefault();
    closeInputDevicePicker({ focusTrigger: true });
  }
}

function renderInputDevices(microphones) {
  const selected = dom.inputDevice.value;
  const options = document.createDocumentFragment();
  options.append(new Option("系统默认设备", ""));
  microphones.forEach((device, index) => {
    options.append(new Option(device.label || `麦克风 ${index + 1}`, device.deviceId));
  });
  // Commit the complete list in one DOM operation. Replacing the select one
  // option at a time can make the native popup repaint while it is opening.
  dom.inputDevice.replaceChildren(options);
  if ([...dom.inputDevice.options].some((option) => option.value === selected)) dom.inputDevice.value = selected;
  if (dom.inputDeviceSummary) dom.inputDeviceSummary.textContent = dom.inputDevice.selectedOptions[0]?.textContent || "系统默认设备";
  updateInputDevicePicker();
}

function flushPendingInputDevices() {
  if (!state.pendingInputDevices || inputDeviceMenuIsOpen()) return;
  const microphones = state.pendingInputDevices;
  state.pendingInputDevices = null;
  renderInputDevices(microphones);
}

async function refreshDevices() {
  if (!navigator.mediaDevices?.enumerateDevices) return;
  const refreshId = ++state.inputDevicesRefreshId;
  const devices = await navigator.mediaDevices.enumerateDevices();
  if (refreshId !== state.inputDevicesRefreshId) return;
  const microphones = devices.filter((device) => device.kind === "audioinput");
  if (inputDeviceMenuIsOpen()) {
    state.pendingInputDevices = microphones;
    return;
  }
  state.pendingInputDevices = null;
  renderInputDevices(microphones);
}

async function prepareMicrophone() {
  if (state.stream) {
    const hasLiveTrack = state.stream.getAudioTracks().some((track) => track.readyState === "live");
    if (hasLiveTrack) return;
    stopAudioCapture();
  }
  if (!navigator.mediaDevices?.getUserMedia) throw new Error("当前浏览器不支持麦克风采集，请使用 Chrome 或 Edge。");
  const selected = dom.inputDevice.value;
  const request = navigator.mediaDevices.getUserMedia({ audio: selected ? { deviceId: { exact: selected }, channelCount: 1 } : { channelCount: 1 } });
  const timeoutMs = 10000;
  let timedOut = false;
  let timeoutId;
  request.then((stream) => {
    if (timedOut) stream.getTracks().forEach((track) => track.stop());
  }).catch(() => {});
  try {
    const timeout = new Promise((_, reject) => {
      timeoutId = window.setTimeout(() => {
        timedOut = true;
        const error = new Error("MIC_PERMISSION_TIMEOUT");
        error.code = "MIC_PERMISSION_TIMEOUT";
        reject(error);
      }, timeoutMs);
    });
    state.stream = await Promise.race([request, timeout]);
  } catch (error) {
    if (error?.code === "MIC_PERMISSION_TIMEOUT") {
      throw new Error("麦克风权限请求超时，请在浏览器地址栏允许麦克风后重试。", { cause: error });
    }
    if (error?.name === "NotAllowedError" || error?.name === "PermissionDeniedError") {
      throw new Error("麦克风权限被拒绝，请在浏览器地址栏允许麦克风后重试。", { cause: error });
    }
    if (error?.name === "NotFoundError") {
      throw new Error("没有检测到可用麦克风，请连接设备后重试。", { cause: error });
    }
    throw new Error("无法访问麦克风，请检查设备和浏览器权限后重试。", { cause: error });
  } finally {
    window.clearTimeout(timeoutId);
  }
  await refreshDevices();
  const stream = state.stream;
  for (const track of stream.getAudioTracks()) {
    track.addEventListener("ended", () => {
      if (state.stream !== stream) return;
      state.audioStreamingEnabled = false;
      if (["starting", "recording"].includes(state.meeting?.recording_state)) {
        stopMeeting().catch(() => {});
      }
      stopAudioCapture();
      setConnection("楹﹀厠椋庡凡鏂紑", "danger");
      setNotice("楹﹀厠椋庤澶囧凡鏂紑锛岃閲嶆柊杩炴帴璁惧鍚庡啀寮€濮嬪綍闊?", "error");
    }, { once: true });
  }
}

async function startAudioCapture() {
  if (!state.stream) return;
  if (state.audioReady && state.audioContext && state.audioNode && state.audioSource) return;
  if (state.audioContext || state.audioNode || state.audioSource) stopAudioCapture();
  try {
    state.audioContext = new AudioContext();
    await state.audioContext.audioWorklet.addModule("/static/audio-worklet.js?v=8");
    state.audioSource = state.audioContext.createMediaStreamSource(state.stream);
    state.audioNode = new AudioWorkletNode(state.audioContext, "meeting-capture-processor", { processorOptions: { targetRate: 16000, packetSamples: 640, thresholdPercent: state.volumeThresholdPercent } });
    state.audioNode.port.onmessage = (event) => {
      if (event.data?.type === "level") {
        const level = Math.min(1, Number(event.data.value) || 0);
        if (state.stream) queueMicrophoneLevel(level);
      } else if (event.data?.type === "audio" && state.audioStreamingEnabled && state.ws?.readyState === WebSocket.OPEN) {
        try { state.ws.send(event.data.buffer); } catch { state.audioStreamingEnabled = false; }
      }
    };
    state.audioSource.connect(state.audioNode);
    const silentGain = state.audioContext.createGain();
    silentGain.gain.value = 0;
    state.audioNode.connect(silentGain);
    silentGain.connect(state.audioContext.destination);
    await state.audioContext.resume();
    state.audioReady = true;
  } catch (error) {
    stopAudioCapture();
    throw error;
  }
}

function stopAudioCapture() {
  state.audioReady = false;
  state.audioStreamingEnabled = false;
  try { state.audioNode?.disconnect(); } catch {}
  try { state.audioSource?.disconnect(); } catch {}
  state.audioNode = null;
  state.audioSource = null;
  if (state.audioContext) state.audioContext.close().catch(() => {});
  state.audioContext = null;
  const stream = state.stream;
  state.stream = null;
  if (stream) stream.getTracks().forEach((track) => track.stop());
  if (state.microphoneLevelFrame != null) window.cancelAnimationFrame(state.microphoneLevelFrame);
  state.microphoneLevelFrame = null;
  state.microphoneLevelFrameAt = 0;
  renderMicrophoneLevel(0, false);
  dom.levelBar.style.width = "0%";
  dom.levelText.textContent = "麦克风已停止";
}

async function startMicrophonePreview() {
  if (state.meeting?.recording_state !== "created") return;
  state.audioStreamingEnabled = false;
  try {
    await prepareMicrophone();
    await startAudioCapture();
    dom.levelText.textContent = "麦克风预览中";
  } catch (error) {
    stopAudioCapture();
    throw error;
  }
}

async function restartMicrophonePreview() {
  if (state.meeting?.recording_state !== "created") return;
  stopAudioCapture();
  await startMicrophonePreview();
}

function websocketUrl(id) {
  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${location.host}/api/v2/meetings/${encodeURIComponent(id)}/stream`;
}

function scheduleStreamReconnect(id) {
  if (state.reconnectTimer || state.intentionalClose || state.meeting?.id !== id) return;
  if (!["starting", "recording", "finalizing"].includes(state.meeting?.recording_state)) return;
  /*
  setConnection("杩炴帴鏂紑锛屾鍦ㄩ噸杩?, "warning");
  */
  setConnection("reconnecting", "warning");
  const delay = Math.min(5000, 400 * 2 ** state.reconnectAttempt++);
  state.reconnectTimer = window.setTimeout(() => {
    state.reconnectTimer = null;
    connectStream(id).catch(() => {});
  }, delay);
}

async function connectStream(id, { initial = false } = {}) {
  if (state.reconnectTimer) { clearTimeout(state.reconnectTimer); state.reconnectTimer = null; }
  const generation = ++state.streamGeneration;
  const previous = state.ws;
  state.ws = null;
  state.audioStreamingEnabled = false;
  if (previous) {
    try { previous.close(); } catch {}
  }
  state.intentionalClose = false;
  let ticket;
  try {
    ticket = await requestJson(`/api/v2/meetings/${encodeURIComponent(id)}/stream-ticket`, { method: "POST" });
  } catch (error) {
    if (!initial) scheduleStreamReconnect(id);
    throw error;
  }
  if (generation !== state.streamGeneration || state.intentionalClose || state.meeting?.id !== id) return false;
  const socket = new WebSocket(websocketUrl(id));
  socket.binaryType = "arraybuffer";
  // A reconnect must complete auth + audio_config before the existing
  // worklet is allowed to send binary packets again.
  state.audioStreamingEnabled = false;
  state.ws = socket;
  setConnection("正在连接会议", "warning");
  socket.onopen = () => {
    if (generation !== state.streamGeneration || state.ws !== socket) {
      try { socket.close(); } catch {}
      return;
    }
    socket.send(JSON.stringify({ type: "auth", ticket: ticket.ticket }));
    setConnection("会议连接中", "warning");
  };
  socket.onmessage = async (event) => {
    if (generation !== state.streamGeneration || state.ws !== socket) return;
    let payload;
    try { payload = JSON.parse(event.data); } catch { return; }
    await handleEvent(payload, socket, id, generation);
  };
  /*
  socket.onerror = () => setConnection("连接异常", "danger");
  socket.onclose = () => {
    if (state.ws === socket) state.ws = null;
    state.audioStreamingEnabled = false;
    if (!state.intentionalClose && state.meeting?.id === id && ["starting", "recording", "finalizing"].includes(state.meeting.recording_state)) {
      setConnection("连接断开，正在重连", "warning");
      const delay = Math.min(5000, 400 * 2 ** state.reconnectAttempt++);
      state.reconnectTimer = setTimeout(() => connectStream(id).catch(() => {}), delay);
    } else if (state.meeting?.recording_state !== "recording") {
      setConnection("本地已保存", "neutral");
    }
  };
  */
  socket.onerror = () => {
    if (generation === state.streamGeneration && state.ws === socket) setConnection("杩炴帴寮傚父", "danger");
  };
  socket.onclose = () => {
    if (generation !== state.streamGeneration || state.ws !== socket) return;
    state.ws = null;
    state.audioStreamingEnabled = false;
    if (!state.intentionalClose && state.meeting?.id === id && ["starting", "recording", "finalizing"].includes(state.meeting.recording_state)) {
      scheduleStreamReconnect(id);
    } else if (state.meeting?.recording_state !== "recording") {
      /*
      setConnection("鏈湴宸蹭繚瀛?, "neutral");
      */
      setConnection("saved locally", "neutral");
    }
  };
  return true;
}

function closeStream(intentional = true) {
  if (state.reconnectTimer) { clearTimeout(state.reconnectTimer); state.reconnectTimer = null; }
  state.streamGeneration += 1;
  state.intentionalClose = intentional;
  state.audioStreamingEnabled = false;
  for (const waiter of state.audioFlushWaiters.values()) {
    window.clearTimeout(waiter.timeoutId);
    waiter.resolve(false);
  }
  state.audioFlushWaiters.clear();
  if (state.ws) {
    try { state.ws.close(); } catch {}
    state.ws = null;
  }
  if (intentional) stopAudioCapture();
}

async function flushAudioStream() {
  const socket = state.ws;
  if (!socket || socket.readyState !== WebSocket.OPEN) {
    state.audioStreamingEnabled = false;
    return false;
  }
  const requestId = globalThis.crypto?.randomUUID?.() || `flush-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  const timeoutId = window.setTimeout(() => {
    const waiter = state.audioFlushWaiters.get(requestId);
    if (!waiter) return;
    state.audioFlushWaiters.delete(requestId);
    waiter.resolve(false);
  }, 1500);
  const waiterPromise = new Promise((resolve) => {
    state.audioFlushWaiters.set(requestId, { resolve, timeoutId });
  });
  const deadline = Date.now() + 1000;
  while (socket.readyState === WebSocket.OPEN && socket.bufferedAmount > 0 && Date.now() < deadline) {
    await new Promise((resolve) => window.setTimeout(resolve, 20));
  }
  if (socket.readyState !== WebSocket.OPEN) {
    state.audioStreamingEnabled = false;
    window.clearTimeout(timeoutId);
    state.audioFlushWaiters.delete(requestId);
    return false;
  }
  state.audioStreamingEnabled = false;
  try {
    socket.send(JSON.stringify({ type: "audio_flush", request_id: requestId }));
  } catch {
    window.clearTimeout(timeoutId);
    state.audioFlushWaiters.delete(requestId);
    return false;
  }
  return waiterPromise;
}

async function handleEvent(payload, sourceSocket = null, meetingId = null, sourceGeneration = null) {
  if (sourceSocket && (
    state.ws !== sourceSocket
    || state.meeting?.id !== meetingId
    || (sourceGeneration != null && state.streamGeneration !== sourceGeneration)
  )) return;
  const type = payload.type;
  if (type === "audio_flush_ack") {
    const waiter = state.audioFlushWaiters.get(payload.request_id);
    if (waiter) {
      state.audioFlushWaiters.delete(payload.request_id);
      window.clearTimeout(waiter.timeoutId);
      waiter.resolve(Boolean(payload.drained));
    }
    return;
  }
  if (type === "auth_ok") {
    state.reconnectAttempt = 0;
    if (["starting", "recording"].includes(state.meeting?.recording_state)) {
      if (state.ws?.readyState === WebSocket.OPEN) state.ws.send(JSON.stringify({ type: "audio_config", sample_rate: 16000, channels: 1, encoding: "pcm_s16le", packet_ms: 40, sequence_header: true, volume_threshold_percent: state.volumeThresholdPercent }));
      setConnection("实时连接", "success");
    } else {
      setConnection("处理任务连接", "success");
    }
  } else if (type === "audio_config_ack") {
    if (sourceSocket && (state.ws !== sourceSocket || state.meeting?.id !== meetingId)) return;
    state.audioStreamingEnabled = false;
    try {
      await startAudioCapture();
      if (sourceGeneration != null && state.streamGeneration !== sourceGeneration) return;
      state.audioStreamingEnabled = true;
    } catch (error) {
      if (sourceGeneration != null && state.streamGeneration !== sourceGeneration) return;
      state.audioStreamingEnabled = false;
      setNotice(error.message || "鏃犳硶鍚姩闊抽鎹曡幏", "error");
      if (["starting", "recording"].includes(state.meeting?.recording_state)) {
        await requestJson(`/api/v2/meetings/${encodeURIComponent(meetingId || state.meeting.id)}/stop`, { method: "POST" }).catch(() => {});
      }
      closeStream(true);
    }
  } else if (type === "audio_threshold_ack") {
    setVolumeThreshold(payload.percent, false);
  } else if (type === "audio_threshold_error") {
    setNotice(payload.message || "音量阈值设置无效", "error");
  } else if (type === "snapshot") {
    applySnapshot(payload.meeting || payload, false);
    const snapshot = payload.meeting || payload;
    if (snapshot?.id && (
      snapshot.recording_state === "complete"
      || Number(snapshot.transcript_revision || 0) > Number(state.loadedTranscriptRevision || 0)
    )) {
      await loadFullTranscript(snapshot.id, sourceGeneration).catch(() => {});
    }
    if (state.meeting?.recording_state === "recording") setConnection("实时连接", "success");
  } else if (type === "meeting_renamed") {
    if (payload.meeting) applySnapshot(payload.meeting, false);
  } else if (type === "meeting_state") {
    if (payload.meeting) applySnapshot(payload.meeting, false);
    if (payload.message) dom.recordingHint.textContent = payload.message;
  } else if (type === "paragraph_update") {
    const paragraph = payload.paragraph;
    if (state.meeting && payload.transcript_revision != null) {
      const revision = Number(payload.transcript_revision) || 0;
      state.meeting.transcript_revision = Math.max(Number(state.meeting.transcript_revision || 0), revision);
      state.loadedTranscriptRevision = Math.max(state.loadedTranscriptRevision, revision);
    }
    const changed = upsertUtterance(paragraph);
    if (!changed) return;
    clearDraft();
    if (state.meeting) {
      state.meeting.paragraph_count = state.transcript.size;
      if (paragraph?.closed) {
        if (state.meeting.active_paragraph?.segment_id === paragraph.segment_id) state.meeting.active_paragraph = null;
      } else {
        state.meeting.active_paragraph = paragraph;
        state.meeting.current_language = paragraph?.language || state.meeting.current_language;
        state.meeting.current_variant = paragraph?.speech_variant || state.meeting.current_variant;
      }
    }
    if (paragraph?.segment_id) renderTranscriptItem(paragraph.segment_id, true);
  } else if (type === "audio_input") {
    if (state.meeting) Object.assign(state.meeting, payload);
  } else if (type === "recording_complete") {
    clearDraft();
    stopAudioCapture();
    applySnapshot(payload.meeting, false);
    await loadFullTranscript(payload.meeting?.id, sourceGeneration).catch(() => {});
    closeStream(true);
    setConnection("录音已完成", "success");
  } else if (type === "agent_progress") {
    if (payload.phase === "request") setNotice("正在调用会议智能体…", "info");
  } else if (type === "agent_complete") {
    const result = payload.result || {};
    if (state.meeting) Object.assign(state.meeting, {
      summary: payload.summary || result.summary_markdown || state.meeting.summary,
      summary_state: "complete",
      todo: payload.todo || { schema_version: "1.0", items: result.todo || [] },
      todo_state: "complete",
      refined_transcript: payload.refined_transcript || result.transcript || [],
      refined_transcript_markdown: payload.refined_transcript_markdown || result.transcript_markdown || state.meeting.refined_transcript_markdown,
      todo_markdown: payload.todo_markdown || result.todo_markdown || state.meeting.todo_markdown,
      summary_revision: payload.summary_revision ?? state.meeting.summary_revision,
      files: payload.files || state.meeting.files,
    });
    renderRefinedTranscript(state.meeting?.refined_transcript, "complete", state.meeting?.refined_transcript_markdown);
    renderSummary(state.meeting?.summary, "complete");
    renderTodo(state.meeting?.todo, "complete", state.meeting?.todo_markdown);
    renderFiles(state.meeting?.files || []);
  } else if (type === "warning") {
    setNotice(payload.message || "处理出现警告", "warning");
  } else if (type === "error") {
    setNotice(payload.message || "处理失败", "error");
    if (state.meeting && payload.code === "agent_failed") Object.assign(state.meeting, {
      summary_state: "error",
      summary: payload.summary ?? state.meeting.summary,
      summary_revision: payload.summary_revision ?? state.meeting.summary_revision,
      todo_state: payload.agent ? "error" : state.meeting.todo_state,
      refined_transcript: payload.agent ? [] : state.meeting.refined_transcript,
      refined_transcript_markdown: payload.agent ? "" : state.meeting.refined_transcript_markdown,
      todo_markdown: payload.agent ? "" : state.meeting.todo_markdown,
    });
    if (state.meeting && payload.code === "agent_failed") state.meeting.todo_state = "error";
    if (payload.agent) renderRefinedTranscript(state.meeting?.refined_transcript, "error", state.meeting?.refined_transcript_markdown);
    renderSummary(state.meeting?.summary, state.meeting?.summary_state || "error");
    renderTodo(state.meeting?.todo, state.meeting?.todo_state || "error", state.meeting?.todo_markdown);
  }
  renderMeetings();
}

async function createMeeting(title) {
  if (["created", "starting", "recording"].includes(state.meeting?.recording_state)) {
    setNotice("当前会议尚未结束，请先开始或结束它。", "warning");
    return;
  }
  setNotice("正在创建会议…", "info");
  try {
    const health = await checkHealth();
    if (health.meeting_start_mode !== "manual") {
      throw new Error("服务版本尚未支持创建后手动开始，请重启会记服务后重试。");
    }
    const snapshot = await requestJson("/api/v2/meetings", { method: "POST", body: JSON.stringify({ title: title.trim() || "未命名会议" }) });
    if (snapshot.recording_state !== "created") {
      if (["starting", "recording"].includes(snapshot.recording_state)) {
        await requestJson(`/api/v2/meetings/${encodeURIComponent(snapshot.id)}/stop`, { method: "POST" }).catch(() => {});
      }
      throw new Error("会议创建后意外进入录音状态，已停止录音；请重启会记服务后重试。");
    }
    state.meeting = null;
    applySnapshot(snapshot, true);
    setConnection("等待开始", "neutral");
    setNotice("会议已创建，正在连接麦克风以预览实时音量…", "info");
    try {
      await startMicrophonePreview();
      setNotice("麦克风仅在本地预览，确认设备和阈值后点击“开始录音”。", "info");
    } catch (error) {
      setNotice(`会议已创建，但麦克风预览不可用：${error.message}`, "warning");
    }
  } catch (error) {
    setNotice(error.message || "无法创建会议", "error");
  }
}

async function startRecording() {
  if (!state.meeting || state.meeting.recording_state !== "created") return;
  dom.startRecordingButton.disabled = true;
  const meetingId = state.meeting.id;
  let backendStarted = false;
  let startAttempted = false;
  setNotice("正在申请麦克风权限…", "info");
  try {
    await checkHealth();
    await prepareMicrophone();
    state.audioStreamingEnabled = false;
    startAttempted = true;
    const snapshot = await requestJson(`/api/v2/meetings/${encodeURIComponent(state.meeting.id)}/start`, { method: "POST" });
    backendStarted = true;
    applySnapshot(snapshot, false);
    const connected = await connectStream(snapshot.id, { initial: true });
    if (connected === false) throw new Error("浼氳杩炴帴宸茶鍙栨秷");
    setNotice("");
  } catch (error) {
    if (startAttempted || backendStarted || state.meeting?.recording_state === "starting" || state.meeting?.recording_state === "recording") {
      await requestJson(`/api/v2/meetings/${encodeURIComponent(meetingId)}/stop`, { method: "POST" }).catch(() => {});
    }
    stopAudioCapture();
    closeStream(true);
    dom.startRecordingButton.disabled = false;
    setNotice(error.message || "无法开始录音", "error");
  }
}

async function stopMeeting() {
  if (!state.meeting || !["starting", "recording"].includes(state.meeting.recording_state)) return;
  dom.recordButton.disabled = true;
  dom.recordingHint.textContent = "正在保存最后一个语音片段，请稍候…";
  try {
    await flushAudioStream();
    await requestJson(`/api/v2/meetings/${encodeURIComponent(state.meeting.id)}/stop`, { method: "POST" });
  }
  catch (error) { setNotice(error.message, "error"); dom.recordButton.disabled = false; }
}

async function retrySummary() {
  if (!state.meeting || state.meeting.summary_state === "running") return;
  dom.retrySummary.disabled = true;
  try {
    await requestJson(`/api/v2/meetings/${encodeURIComponent(state.meeting.id)}/summary`, { method: "POST" });
    state.meeting.summary_state = "running";
    state.meeting.todo_state = "running";
    state.meeting.refined_transcript = [];
    state.meeting.refined_transcript_markdown = "";
    state.meeting.todo_markdown = "";
    renderRefinedTranscript([], "running");
    renderSummary(state.meeting.summary, "running");
    renderTodo(null, "running");
    setNotice("正在生成精修转写、会议纪要和 To-do-list。", "info");
  }
  catch (error) {
    dom.retrySummary.disabled = false;
    setNotice(error.message, "error");
  }
}

function openRenameDialog(meeting) {
  if (!meeting || !dom.renameDialog || !dom.renameDialogTitle) return;
  state.renameMeetingId = meeting.id;
  dom.renameDialogTitle.value = meeting.title || "未命名会议";
  dom.renameDialog.showModal();
  dom.renameDialogTitle.focus();
  dom.renameDialogTitle.select();
}

async function renameMeeting() {
  const meetingId = state.renameMeetingId;
  const title = dom.renameDialogTitle?.value.trim() || "";
  if (!meetingId || !title) {
    setNotice("请输入会议名称。", "warning");
    return;
  }
  dom.renameMeetingSubmit.disabled = true;
  try {
    const snapshot = await requestJson(`/api/v2/meetings/${encodeURIComponent(meetingId)}`, {
      method: "PATCH",
      body: JSON.stringify({ title }),
    });
    if (state.meeting?.id === snapshot.id) applySnapshot(snapshot, false);
    else {
      const meetingIndex = state.meetings.findIndex((meeting) => meeting.id === snapshot.id);
      if (meetingIndex >= 0) state.meetings[meetingIndex] = { ...state.meetings[meetingIndex], ...snapshot };
    }
    renderMeetings();
    dom.renameDialog.close();
    setNotice("会议已重命名。", "success");
  } catch (error) {
    setNotice(error.message || "会议重命名失败。", "error");
  } finally {
    dom.renameMeetingSubmit.disabled = false;
  }
}

async function deleteMeeting(meetingId = state.meeting?.id, meetingTitle = state.meeting?.title) {
  if (!meetingId || !await requestConfirmation("确认删除会议", `确定删除“${meetingTitle || "未命名会议"}”及其录音和文件吗？`, "确认删除")) return;
  try {
    await requestJson(`/api/v2/meetings/${encodeURIComponent(meetingId)}`, { method: "DELETE" });
    state.meetings = state.meetings.filter((meeting) => meeting.id !== meetingId);
    if (state.meeting?.id === meetingId) {
      closeStream(true);
      state.meeting = null;
      clearTranscript();
      dom.welcome.hidden = false;
      dom.meetingPanel.hidden = true;
      dom.pageTitle.textContent = "";
      dom.pageSubtitle.textContent = "";
      renderRefinedTranscript([], "idle");
      renderSummary("", "idle");
      renderTodo(null, "waiting_summary");
      dom.filesCard.hidden = true;
    }
    renderMeetings();
    setNotice("会议已删除。", "success");
  } catch (error) { setNotice(error.message, "error"); }
}

function downloadFile(name) {
  if (!state.meeting) return;
  const url = `/api/v2/meetings/${encodeURIComponent(state.meeting.id)}/files/${name}`;
  const link = document.createElement("a");
  link.href = url;
  link.download = name.split("/").pop() || name;
  link.rel = "noopener";
  document.body.append(link);
  link.click();
  link.remove();
}

function bindEvents() {
  $("#startMeeting").addEventListener("click", () => createMeeting(dom.meetingTitle.value));
  $("#newMeeting").addEventListener("click", () => { dom.dialogTitle.value = ""; dom.dialog.showModal(); });
  dom.dialogForm.addEventListener("submit", (event) => {
    if (event.submitter?.value !== "default") return;
    event.preventDefault();
    dom.dialog.close();
    createMeeting(dom.dialogTitle.value);
  });
  dom.renameDialogForm.addEventListener("submit", (event) => {
    if (event.submitter?.value !== "default") return;
    event.preventDefault();
    renameMeeting();
  });
  dom.renameDialog.addEventListener("close", () => {
    state.renameMeetingId = null;
    dom.renameMeetingSubmit.disabled = false;
  });
  dom.confirmDialogForm.addEventListener("submit", (event) => {
    if (event.submitter?.value !== "default") return;
    event.preventDefault();
    dom.confirmDialog.close("confirm");
  });
  dom.confirmDialog.addEventListener("close", () => {
    const resolver = state.confirmResolver;
    state.confirmResolver = null;
    resolver?.(dom.confirmDialog.returnValue === "confirm");
  });
  dom.authDialogForm.addEventListener("submit", (event) => {
    if (event.submitter?.value !== "default") return;
    event.preventDefault();
    if (!dom.authToken.value.trim()) {
      dom.authDialog.close("cancel");
      return;
    }
    dom.authDialog.close("submit");
  });
  dom.authDialog.addEventListener("close", () => {
    const resolver = state.authResolver;
    state.authResolver = null;
    state.authPromise = null;
    resolver?.(dom.authDialog.returnValue === "submit" ? dom.authToken.value.trim() : null);
  });
  $("#refreshMeetings").addEventListener("click", () => loadMeetings().catch((error) => setNotice(error.message, "error")));
  dom.search.addEventListener("input", renderMeetings);
  dom.startRecordingButton.addEventListener("click", startRecording);
  $("#recordButton").addEventListener("click", stopMeeting);
  dom.themeToggle.addEventListener("click", () => {
    const current = document.documentElement.dataset.theme === "dark" ? "dark" : "light";
    applyTheme(current === "dark" ? "light" : "dark");
  });
  dom.retrySummary.addEventListener("click", retrySummary);
  dom.downloadSummary.addEventListener("click", () => downloadFile("meeting_minutes.md"));
  dom.downloadTodo.addEventListener("click", () => downloadFile("todo_list.json"));
  dom.openAsrSettings.addEventListener("click", openAsrSettings);
  dom.asrSettingsDialog.addEventListener("click", (event) => {
    if (event.target === dom.asrSettingsDialog && dom.asrSettingsDialog.open) {
      dom.asrSettingsDialog.close("cancel");
    }
  });
  dom.asrSettingsForm.addEventListener("submit", (event) => {
    if (event.submitter?.value !== "default") return;
    event.preventDefault();
    saveAsrSettings();
  });
  dom.resetAsrSettings.addEventListener("click", resetMeetingSettings);
  for (const field of MEETING_NUMBER_FIELDS) {
    const slider = $(`#${field.slider}`);
    const input = $(`#${field.input}`);
    slider?.addEventListener("input", (event) => {
      setMeetingField(field, event.target.value, { updateState: false });
    });
    input?.addEventListener("input", (event) => {
      if (event.target.value !== "" && Number.isFinite(Number(event.target.value))) {
        setMeetingField(field, event.target.value, { updateState: false });
      }
    });
    input?.addEventListener("change", (event) => {
      setMeetingField(field, event.target.value, { updateState: false });
    });
  }
  $("#refreshDevices").addEventListener("click", () => refreshDevices().catch(() => {}));
  dom.volumeThreshold.addEventListener("input", () => {
    setVolumeThreshold(dom.volumeThreshold.value);
  });
  dom.volumeThresholdValue.addEventListener("input", () => {
    if (dom.volumeThresholdValue.value !== "" && Number.isFinite(Number(dom.volumeThresholdValue.value))) {
      setVolumeThreshold(dom.volumeThresholdValue.value);
    }
  });
  dom.volumeThresholdValue.addEventListener("change", () => {
    setVolumeThreshold(dom.volumeThresholdValue.value);
  });
  dom.inputDeviceTrigger.addEventListener("click", () => {
    if (state.inputDevicePickerOpen) closeInputDevicePicker({ focusTrigger: true });
    else openInputDevicePicker();
  });
  dom.inputDeviceTrigger.addEventListener("keydown", handleInputDeviceTriggerKeydown);
  dom.inputDeviceOptions.addEventListener("click", (event) => {
    const option = event.target.closest?.('[role="option"]');
    if (option) chooseInputDevice(option.dataset.value || "");
  });
  dom.inputDeviceOptions.addEventListener("keydown", handleInputDeviceOptionsKeydown);
  document.addEventListener("pointerdown", (event) => {
    if (state.inputDevicePickerOpen && !dom.inputDevicePicker.contains(event.target)) closeInputDevicePicker();
  });
  dom.inputDevice.addEventListener("change", async () => {
    flushPendingInputDevices();
    updateInputDevicePicker();
    if (dom.inputDeviceSummary) dom.inputDeviceSummary.textContent = dom.inputDevice.selectedOptions[0]?.textContent || "系统默认设备";
    if (state.meeting?.recording_state === "recording") {
      setNotice("设备选择会在下一场会议生效。", "info");
    } else if (state.meeting?.recording_state === "created") {
      try {
        await restartMicrophonePreview();
        setNotice("已切换麦克风，可根据实时音量重新调整阈值。", "info");
      } catch (error) {
        setNotice(error.message || "无法预览所选麦克风", "error");
      }
    }
  });
  dom.transcriptList.addEventListener("scroll", () => {
    const distance = dom.transcriptList.scrollHeight - dom.transcriptList.scrollTop - dom.transcriptList.clientHeight;
    state.transcriptNearBottom = distance < 80;
    dom.jumpLatest.hidden = state.transcriptNearBottom || !state.transcript.size;
  });
  dom.jumpLatest.addEventListener("click", () => { state.transcriptNearBottom = true; renderTranscript(true); });
  dom.asrSettingsDialog.addEventListener("close", () => closeInputDevicePicker());
}

async function init() {
  applyTheme(storedTheme(), { persist: false });
  moveAudioSettingsIntoDialog();
  updateInputDevicePicker();
  bindEvents();
  state.timer = window.setInterval(updateTimer, 1000);
  window.setInterval(refreshCurrentMeeting, 3000);
  state.pingTimer = window.setInterval(() => {
    if (state.ws?.readyState === WebSocket.OPEN) {
      try { state.ws.send(JSON.stringify({ type: "ping" })); } catch {}
    }
  }, 10000);
  try {
    await checkHealth();
    await loadMeetings();
    await refreshDevices();
  } catch (error) { setNotice(error.message || "无法连接本机服务", "error"); }
  renderSummary("", "idle");
  renderRefinedTranscript([], "idle");
  renderTodo(null, "waiting_summary");
}

window.addEventListener("beforeunload", () => closeStream(true));
init();
