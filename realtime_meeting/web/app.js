const $ = (selector) => document.querySelector(selector);

const RECOMMENDED_ASR_SETTINGS = Object.freeze({
  realtime_beam_size: 5,
  refine_beam_size: 6,
  best_of: 5,
  silence_ms: 700,
  vad_minimum_speech_ms: 450,
});

const RECOMMENDED_MEETING_SETTINGS = Object.freeze({
  ...RECOMMENDED_ASR_SETTINGS,
  volume_threshold_percent: 2.2,
  speech_start_ms: 80,
  audio_pre_roll_ms: 240,
  vad_minimum_speech_ratio: 0.12,
  max_utterance_seconds: 8,
  partial_interval_ms: 900,
  audio_segment_minutes: 30,
  retry_temperature: 0.2,
  log_prob_threshold: -1,
  no_speech_threshold: 0.6,
  compression_ratio_threshold: 2.4,
  translation_beam_size: 2,
  translation_max_decoding_length: 384,
  translation_repetition_penalty: 1.05,
  speaker_cluster_threshold: 0.68,
  speaker_min_speech_seconds: 0.35,
  speaker_max_silence_gap_seconds: 0.25,
  speaker_overlap_include_threshold: 0.15,
  enable_refinement: true,
  enable_postprocess: true,
  diarization_required: true,
  keep_audio: true,
});

const SETTINGS_TEMPLATE_DEFINITIONS = Object.freeze({
  balanced: Object.freeze({
    label: "均衡实时",
    description: "适合日常中文/英文/德文会议，兼顾出字速度和识别质量。",
    values: Object.freeze({ ...RECOMMENDED_MEETING_SETTINGS }),
  }),
  low_latency: Object.freeze({
    label: "低延迟速记",
    description: "更快刷新逐句稿，适合讨论密集、希望尽快看到草稿的会议。",
    values: Object.freeze({
      ...RECOMMENDED_MEETING_SETTINGS,
      speech_start_ms: 60,
      audio_pre_roll_ms: 160,
      silence_ms: 400,
      vad_minimum_speech_ms: 250,
      vad_minimum_speech_ratio: 0.08,
      max_utterance_seconds: 6,
      partial_interval_ms: 400,
      realtime_beam_size: 3,
      refine_beam_size: 4,
      best_of: 3,
      retry_temperature: 0.15,
      log_prob_threshold: -1.2,
      no_speech_threshold: 0.65,
      translation_beam_size: 1,
      translation_max_decoding_length: 256,
      translation_repetition_penalty: 1.02,
      speaker_cluster_threshold: 0.62,
      speaker_min_speech_seconds: 0.25,
      speaker_max_silence_gap_seconds: 0.2,
      speaker_overlap_include_threshold: 0.1,
    }),
  }),
  quality: Object.freeze({
    label: "高质量会议",
    description: "增加识别和翻译搜索宽度，适合正式会议、口音较重或需要复核的录音。",
    values: Object.freeze({
      ...RECOMMENDED_MEETING_SETTINGS,
      volume_threshold_percent: 2.5,
      speech_start_ms: 120,
      audio_pre_roll_ms: 320,
      silence_ms: 900,
      vad_minimum_speech_ms: 600,
      vad_minimum_speech_ratio: 0.16,
      max_utterance_seconds: 10,
      partial_interval_ms: 1200,
      audio_segment_minutes: 60,
      realtime_beam_size: 8,
      refine_beam_size: 10,
      best_of: 9,
      retry_temperature: 0.25,
      log_prob_threshold: -0.8,
      no_speech_threshold: 0.55,
      compression_ratio_threshold: 2.2,
      translation_beam_size: 3,
      translation_max_decoding_length: 512,
      translation_repetition_penalty: 1.1,
      speaker_cluster_threshold: 0.74,
      speaker_min_speech_seconds: 0.5,
      speaker_max_silence_gap_seconds: 0.3,
      speaker_overlap_include_threshold: 0.2,
    }),
  }),
});

const SETTINGS_BOOLEAN_KEYS = ["enable_refinement", "enable_postprocess", "diarization_required", "keep_audio"];

const MEETING_NUMBER_FIELDS = [
  { key: "volume_threshold_percent", slider: "volumeThreshold", input: "volumeThresholdValue", min: 0, max: 30, step: 0.1 },
  { key: "speech_start_ms", slider: "asrSpeechStartMs", input: "asrSpeechStartMsValue", min: 40, max: 1000, step: 10 },
  { key: "audio_pre_roll_ms", slider: "asrAudioPreRollMs", input: "asrAudioPreRollMsValue", min: 40, max: 1000, step: 10 },
  { key: "silence_ms", slider: "asrSilenceMs", input: "asrSilenceMsValue", min: 160, max: 2000, step: 10 },
  { key: "vad_minimum_speech_ms", slider: "asrMinimumSpeechMs", input: "asrMinimumSpeechMsValue", min: 0, max: 2000, step: 10 },
  { key: "vad_minimum_speech_ratio", slider: "asrSpeechRatio", input: "asrSpeechRatioValue", min: 0, max: 1, step: 0.01 },
  { key: "max_utterance_seconds", slider: "asrMaxUtteranceSeconds", input: "asrMaxUtteranceSecondsValue", min: 2, max: 12, step: 0.5 },
  { key: "partial_interval_ms", slider: "asrPartialIntervalMs", input: "asrPartialIntervalMsValue", min: 100, max: 5000, step: 50 },
  { key: "realtime_beam_size", slider: "asrRealtimeBeamSize", input: "asrRealtimeBeamSizeValue", min: 1, max: 10, step: 1 },
  { key: "refine_beam_size", slider: "asrRefineBeamSize", input: "asrRefineBeamSizeValue", min: 1, max: 12, step: 1 },
  { key: "best_of", slider: "asrBestOf", input: "asrBestOfValue", min: 1, max: 12, step: 1 },
  { key: "retry_temperature", slider: "asrRetryTemperature", input: "asrRetryTemperatureValue", min: 0, max: 1, step: 0.05 },
  { key: "log_prob_threshold", slider: "asrLogProbThreshold", input: "asrLogProbThresholdValue", min: -10, max: 0, step: 0.1 },
  { key: "no_speech_threshold", slider: "asrNoSpeechThreshold", input: "asrNoSpeechThresholdValue", min: 0, max: 1, step: 0.01 },
  { key: "compression_ratio_threshold", slider: "asrCompressionRatioThreshold", input: "asrCompressionRatioThresholdValue", min: 1, max: 10, step: 0.1 },
  { key: "translation_beam_size", slider: "translationBeamSize", input: "translationBeamSizeValue", min: 1, max: 8, step: 1 },
  { key: "translation_max_decoding_length", slider: "translationMaxLength", input: "translationMaxLengthValue", min: 64, max: 1024, step: 16 },
  { key: "translation_repetition_penalty", slider: "translationRepetitionPenalty", input: "translationRepetitionPenaltyValue", min: 1, max: 2, step: 0.01 },
  { key: "audio_segment_minutes", slider: "audioSegmentMinutes", input: "audioSegmentMinutesValue", min: 1, max: 120, step: 1 },
  { key: "speaker_cluster_threshold", slider: "speakerClusterThreshold", input: "speakerClusterThresholdValue", min: 0.4, max: 0.95, step: 0.01 },
  { key: "speaker_min_speech_seconds", slider: "speakerMinSpeech", input: "speakerMinSpeechValue", min: 0.2, max: 2, step: 0.05 },
  { key: "speaker_max_silence_gap_seconds", slider: "speakerMaxGap", input: "speakerMaxGapValue", min: 0.05, max: 1, step: 0.05 },
  { key: "speaker_overlap_include_threshold", slider: "speakerOverlap", input: "speakerOverlapValue", min: 0, max: 1, step: 0.01 },
];

// Keep the old name as a small compatibility surface for extensions and
// static checks from the first settings panel version.
const ASR_SETTING_FIELDS = MEETING_NUMBER_FIELDS.filter((field) => [
  "realtime_beam_size", "refine_beam_size", "best_of", "silence_ms", "vad_minimum_speech_ms",
].includes(field.key));

// The live microphone level is normalized to 0-100%.  Keep this separate
// from the 0-30% background-noise filter threshold range.
const MICROPHONE_METER_MAX_PERCENT = 100;

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
  transcript: new Map(),
  transcriptNodes: new Map(),
  draft: null,
  draftNode: null,
  transcriptNearBottom: true,
  summaryStreaming: false,
  summaryRenderFrame: null,
  timer: null,
  volumeThresholdPercent: 2.2,
  asrSettings: { ...RECOMMENDED_ASR_SETTINGS },
  meetingSettings: { ...RECOMMENDED_MEETING_SETTINGS },
  settingsTemplate: "balanced",
  microphoneLevelPercent: 0,
  pendingMicrophoneLevel: 0,
  microphoneLevelFrame: null,
  audioStreamingEnabled: false,
  lockedLanguages: new Set(),
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
  deleteMeeting: $("#deleteMeeting"),
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
  languageLockGroup: $("#languageLockGroup"),
  notice: $("#notice"),
  postprocessPanel: $("#postprocessPanel"),
  postprocessMessage: $("#postprocessMessage"),
  postprocessBadge: $("#postprocessBadge"),
  postprocessProgressBar: $("#postprocessProgressBar"),
  postprocessStages: $("#postprocessStages"),
  retryPostprocess: $("#retryPostprocess"),
  summaryBadge: $("#summaryBadge"),
  summaryProgress: $("#summaryProgress"),
  summaryProgressBar: $("#summaryProgressBar"),
  summaryText: $("#summaryText"),
  retrySummary: $("#retrySummary"),
  downloadSummary: $("#downloadSummary"),
  todoBadge: $("#todoBadge"),
  todoText: $("#todoText"),
  retryTodo: $("#retryTodo"),
  downloadTodo: $("#downloadTodo"),
  filesCard: $("#filesCard"),
  fileLinks: $("#fileLinks"),
  statusDot: $("#statusDot"),
  systemStatus: $("#systemStatus"),
  dialog: $("#newMeetingDialog"),
  dialogForm: $("#newMeetingForm"),
  dialogTitle: $("#dialogTitle"),
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
  settingsTabs: document.querySelectorAll("[data-settings-tab]"),
  settingsPanels: document.querySelectorAll("[data-settings-panel]"),
  settingsTemplateSelect: $("#settingsTemplateSelect"),
  settingsTemplateHint: $("#settingsTemplateHint"),
  resetAsrSettings: $("#resetAsrSettings"),
  saveAsrSettings: $("#saveAsrSettings"),
};

const recordingLabels = {
  created: "待开始",
  starting: "准备录音",
  recording: "正在录音",
  finalizing: "正在保存",
  complete: "录音完成",
  error: "录音异常",
};
const summaryLabels = { idle: "等待会议结束", queued: "排队中", running: "生成中", complete: "已完成", error: "生成失败" };
const todoLabels = { waiting_summary: "等待会议纪要", queued: "排队中", running: "提取中", complete: "已完成", stale: "纪要已更新", error: "生成失败" };

const postprocessStageLabels = { asr_refine: "ASR精修", diarization: "说话人重排", translation: "翻译", summary: "会议纪要", todo: "To-do-list" };
const postprocessStateLabels = { idle: "未开始", queued: "排队中", running: "处理中", ready_for_summary: "精修完成", complete: "已完成", partial: "部分完成", error: "处理失败" };

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

async function requestJson(path, options = {}) {
  const { authRetried = false, ...fetchOptions } = options;
  const response = await fetch(path, {
    ...fetchOptions,
    credentials: "same-origin",
    headers: { ...(options.body ? { "Content-Type": "application/json" } : {}), ...(options.headers || {}) },
  });
  if (response.status === 401 && path !== "/api/v2/auth/session" && !authRetried) {
    const token = window.prompt("此会议服务需要访问令牌：");
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
    state.ws.send(JSON.stringify({ type: "audio_threshold", percent: state.volumeThresholdPercent }));
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

function normalizeComparableSettings(settings = {}) {
  const normalized = {};
  for (const field of MEETING_NUMBER_FIELDS) normalized[field.key] = clampMeetingValue(field, settings[field.key]);
  for (const key of SETTINGS_BOOLEAN_KEYS) normalized[key] = Boolean(settings[key]);
  return normalized;
}

function settingsEqual(left, right) {
  const a = normalizeComparableSettings(left);
  const b = normalizeComparableSettings(right);
  return MEETING_NUMBER_FIELDS.every((field) => a[field.key] === b[field.key])
    && SETTINGS_BOOLEAN_KEYS.every((key) => a[key] === b[key]);
}

function findSettingsTemplate(settings) {
  for (const [id, template] of Object.entries(SETTINGS_TEMPLATE_DEFINITIONS)) {
    if (settingsEqual(settings, template.values)) return id;
  }
  return "custom";
}

function updateTemplatePicker(templateId = state.settingsTemplate) {
  const select = dom.settingsTemplateSelect;
  if (!select) return;
  const isKnownTemplate = Object.prototype.hasOwnProperty.call(SETTINGS_TEMPLATE_DEFINITIONS, templateId);
  const selected = isKnownTemplate || templateId === "custom" ? templateId : "custom";
  select.value = selected;
  const custom = selected === "custom";
  if (dom.settingsTemplateHint) {
    dom.settingsTemplateHint.textContent = custom ? "自定义模板" : "预设模板";
    dom.settingsTemplateHint.className = `settings-template-status${custom ? " custom" : ""}`;
    dom.settingsTemplateHint.title = custom
      ? "当前参数包含手动调整"
      : SETTINGS_TEMPLATE_DEFINITIONS[selected]?.description || "";
  }
  select.title = custom
    ? "当前参数包含手动调整"
    : SETTINGS_TEMPLATE_DEFINITIONS[selected]?.description || "";
}

function markSettingsCustom() {
  if (state.settingsTemplate === "custom") return;
  state.settingsTemplate = "custom";
  updateTemplatePicker("custom");
}

function applySettingsTemplate(templateId) {
  const template = SETTINGS_TEMPLATE_DEFINITIONS[templateId];
  if (!template) return;
  state.settingsTemplate = templateId;
  renderMeetingSettings(template.values);
  updateTemplatePicker(templateId);
  if (dom.asrSettingsNotice) dom.asrSettingsNotice.textContent = `${template.label}：${template.description}`;
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
    state.settingsTemplate = findSettingsTemplate(state.meetingSettings);
    updateTemplatePicker(state.settingsTemplate);
  }
  for (const field of MEETING_NUMBER_FIELDS) setMeetingField(field, values[field.key], { updateState });
  if (updateState) setVolumeThreshold(values.volume_threshold_percent, false);
  const toggles = {
    enable_refinement: "settingEnableRefinement",
    enable_postprocess: "settingEnablePostprocess",
    diarization_required: "settingDiarizationRequired",
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
  for (const id of ["settingEnableRefinement", "settingEnablePostprocess", "settingDiarizationRequired", "settingKeepAudio"]) {
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
  const values = {};
  for (const field of MEETING_NUMBER_FIELDS) values[field.key] = clampMeetingValue(field, $(`#${field.input}`)?.value);
  for (const [key, id] of Object.entries({
    enable_refinement: "settingEnableRefinement",
    enable_postprocess: "settingEnablePostprocess",
    diarization_required: "settingDiarizationRequired",
    keep_audio: "settingKeepAudio",
  })) values[key] = Boolean($(`#${id}`)?.checked);
  state.settingsTemplate = findSettingsTemplate(values);
  updateTemplatePicker(state.settingsTemplate);
  return values;
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
  state.microphoneLevelFrame = window.requestAnimationFrame(() => {
    state.microphoneLevelFrame = null;
    const value = state.pendingMicrophoneLevel;
    renderMicrophoneLevel(value * 100, true);
    dom.levelBar.style.width = `${Math.round(value * 100)}%`;
    dom.levelText.textContent = value > 0.03 ? "正在采集声音" : "等待说话";
    if (state.meeting) state.meeting.audio_level = value;
  });
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
    const button = document.createElement("button");
    button.type = "button";
    button.className = `meeting-entry ${state.meeting?.id === meeting.id ? "active" : ""}`;
    button.dataset.meetingId = meeting.id;
    button.innerHTML = `<span class="meeting-entry-title"></span><span class="meeting-entry-meta"></span><span class="meeting-entry-state"></span>`;
    button.querySelector(".meeting-entry-title").textContent = meeting.title || "未命名会议";
    button.querySelector(".meeting-entry-meta").textContent = `${formatDate(meeting.started_at)} · ${formatTime(meeting.duration_seconds)}`;
    const stateNode = button.querySelector(".meeting-entry-state");
    stateNode.textContent = recordingLabels[meeting.recording_state] || stateText(meeting.recording_state);
    stateNode.classList.toggle("live", meeting.recording_state === "recording");
    button.addEventListener("click", () => selectMeeting(meeting.id));
    dom.meetingList.append(button);
  }
}

function clearTranscript() {
  clearDraft();
  state.transcript.clear();
  state.transcriptNodes.clear();
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
    if (!current || Number(utterance.revision || 1) >= Number(current.revision || 1)) {
      transcript.delete(utterance.segment_id);
      return Boolean(current);
    }
    return false;
  }
  const previous = transcript.get(utterance.segment_id);
  if (!previous || Number(utterance.revision || 1) >= Number(previous.revision || 1)) {
    transcript.set(utterance.segment_id, utterance);
    return !previous || JSON.stringify(previous) !== JSON.stringify(utterance);
  }
  return false;
}

function createTranscriptNode() {
  const article = document.createElement("article");
  article.className = "transcript-item transcript-message";
  article.innerHTML = `<div class="transcript-meta"><span class="speaker-avatar" aria-hidden="true"></span><span class="speaker-tag"></span><time></time><span class="language-tag"></span></div><div class="transcript-bubbles"><p class="transcript-bubble original-bubble"></p><div class="transcript-bubble translation-bubble" hidden><span class="translation-label">中文翻译</span><b></b><span class="translation-pending" hidden><i></i><i></i><i></i></span></div></div>`;
  return article;
}

function updateTranscriptNode(article, item) {
  const lang = item.language || "unknown";
  article.dataset.segmentId = item.segment_id;
  const speaker = Number(item.speaker_id);
  article.dataset.speaker = Number.isFinite(speaker) ? String(speaker) : "unknown";
  article.querySelector("time").textContent = `${formatTime(item.start).slice(3)} – ${formatTime(item.end).slice(3)}`;
  article.querySelector(".speaker-tag").textContent = `演讲人 ${item.speaker_id ?? "?"}`;
  article.querySelector(".language-tag").textContent = lang.toUpperCase();
  article.querySelector(".original-bubble").textContent = item.text || "";
  const translation = article.querySelector(".translation-bubble");
  const translated = Boolean(item.translation_zh && lang !== "zh");
  const pending = lang !== "zh" && !translated && item.translation_status !== "unsupported";
  const unsupported = lang !== "zh" && !translated && item.translation_status === "unsupported";
  translation.hidden = lang === "zh" || unsupported;
  translation.classList.toggle("is-pending", pending);
  translation.querySelector("b").textContent = translated ? item.translation_zh : "";
  translation.querySelector(".translation-label").textContent = pending ? "中文翻译" : "中文翻译";
  translation.querySelector(".translation-pending").hidden = !pending;
}

function createDraftNode() {
  const article = document.createElement("article");
  article.className = "transcript-item transcript-message transcript-draft";
  article.innerHTML = `<div class="transcript-meta"><span class="speaker-avatar draft-avatar" aria-hidden="true">✦</span><span class="speaker-tag">实时识别</span><time></time><span class="language-tag"></span></div><div class="transcript-bubbles"><p class="transcript-bubble original-bubble"><span class="draft-text"></span><span class="streaming-cursor" aria-hidden="true"></span></p><div class="draft-status"><span class="status-pulse"></span><span>正在识别</span></div></div>`;
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
  dom.utteranceCount.textContent = `${itemCount} 条记录`;
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

function renderSummary(summary, summaryState) {
  const value = String(summary || "").trim();
  const stages = state.meeting?.postprocess?.stages || {};
  const preprocessingReady = ["asr_refine", "diarization", "translation"].every((key) => stages[key]?.state === "complete");
  dom.summaryBadge.textContent = summaryLabels[summaryState] || stateText(summaryState);
  dom.summaryBadge.className = `badge ${summaryState === "complete" ? "success" : summaryState === "error" ? "danger" : summaryState === "running" ? "live" : "neutral"}`;
  dom.summaryText.classList.toggle("empty-result", !value);
  dom.summaryText.classList.toggle("is-streaming", state.summaryStreaming && summaryState === "running");
  dom.summaryText.innerHTML = value
    ? markdownToHtml(value)
    : preprocessingReady
      ? "ASR 精修、说话人重排和翻译已经完成，可以生成会议纪要和 To-do-list。"
      : "停止会议后会先自动完成 ASR 精修、说话人重排和翻译。";
  if (state.summaryStreaming && summaryState === "running" && value) {
    const cursor = document.createElement("span");
    cursor.className = "streaming-cursor summary-cursor";
    cursor.setAttribute("aria-label", "正在生成");
    dom.summaryText.append(cursor);
  }
  dom.retrySummary.hidden = !(preprocessingReady && ["idle", "error", "complete"].includes(summaryState));
  dom.retrySummary.disabled = !preprocessingReady || summaryState === "running";
  dom.retrySummary.textContent = summaryState === "complete" ? "重新生成纪要和 To-do-list" : "生成纪要和 To-do-list";
  dom.downloadSummary.hidden = !value || !state.meeting?.files?.includes("meeting_minutes.md");
  dom.summaryProgress.hidden = !["queued", "running"].includes(summaryState);
}

function scheduleSummaryRender(summaryState = "running") {
  if (state.summaryRenderFrame != null) return;
  state.summaryRenderFrame = window.requestAnimationFrame(() => {
    state.summaryRenderFrame = null;
    renderSummary(state.meeting?.summary, summaryState);
  });
}

function cancelSummaryRender() {
  if (state.summaryRenderFrame != null) {
    window.cancelAnimationFrame(state.summaryRenderFrame);
    state.summaryRenderFrame = null;
  }
}

function renderTodo(todo, todoState) {
  dom.todoBadge.textContent = todoLabels[todoState] || stateText(todoState);
  dom.todoBadge.className = `badge ${todoState === "complete" ? "success" : todoState === "error" ? "danger" : "neutral"}`;
  const items = todo?.items || [];
  dom.todoText.replaceChildren();
  dom.todoText.classList.toggle("empty-result", !todo);
  if (!todo) {
    dom.todoText.textContent = "会议纪要完成后，系统会从中提取明确的行动项。";
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
  dom.retryTodo.hidden = todoState !== "error";
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

function renderPostprocess(postprocess) {
  const value = postprocess || { state: "idle", overall_percent: 0, stages: {} };
  const stages = value.stages || {};
  dom.postprocessPanel.hidden = !value.state || value.state === "idle";
  dom.postprocessBadge.textContent = postprocessStateLabels[value.state] || stateText(value.state);
  dom.postprocessBadge.className = `badge ${value.state === "complete" ? "success" : value.state === "error" ? "danger" : "neutral"}`;
  dom.postprocessProgressBar.style.width = `${Math.max(0, Math.min(100, Number(value.overall_percent) || 0))}%`;
  dom.postprocessMessage.textContent = value.error || (value.state === "ready_for_summary" ? "自动精修已完成，可以生成纪要和 To-do-list" : value.current_stage ? `${postprocessStageLabels[value.current_stage] || value.current_stage}处理中` : "等待处理");
  dom.postprocessStages.replaceChildren();
  for (const [key, stage] of Object.entries(stages)) {
    const article = document.createElement("article");
    article.className = `postprocess-stage ${stage.state || "idle"}`;
    const current = Number(stage.current || 0);
    const total = Number(stage.total || 0);
    article.innerHTML = `<strong></strong><small></small>`;
    article.querySelector("strong").textContent = postprocessStageLabels[key] || key;
    article.querySelector("small").textContent = stage.error || (total ? `${current}/${total}` : postprocessStateLabels[stage.state] || stage.state || "等待");
    dom.postprocessStages.append(article);
  }
  dom.retryPostprocess.hidden = !["error", "partial"].includes(value.state);
}

function applySnapshot(snapshot, replace = false) {
  if (!snapshot) return;
  if (state.meeting?.id === snapshot.id && Number(snapshot.snapshot_revision || 0) < Number(state.meeting.snapshot_revision || 0)) return;
  const changed = state.meeting?.id !== snapshot.id;
  state.meeting = { ...(state.meeting || {}), ...snapshot };
  if (changed || replace || state.meeting.recording_state !== "recording") clearDraft();
  if (state.meeting.summary_state !== "running") {
    state.summaryStreaming = false;
    cancelSummaryRender();
  }
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
    state.settingsTemplate = findSettingsTemplate(state.meetingSettings);
    renderMeetingSettings(state.meetingSettings);
  }
  const meetingIndex = state.meetings.findIndex((meeting) => meeting.id === snapshot.id);
  if (meetingIndex >= 0) state.meetings[meetingIndex] = { ...state.meetings[meetingIndex], ...snapshot };
  else state.meetings.unshift(snapshot);
  let transcriptChanged = changed || replace;
  if (transcriptChanged) {
    clearTranscript();
  }
  for (const item of snapshot.recent_utterances || []) transcriptChanged = upsertUtterance(item) || transcriptChanged;
  dom.pageTitle.textContent = state.meeting.title || "未命名会议";
  dom.pageSubtitle.textContent = state.meeting.recording_state === "recording"
    ? "实时保留原文；英文和德文句子会异步补充中文翻译。"
    : isCreated ? "会议已创建，可以先调整输入设备和背景声过滤，再手动开始录音。" : "这场会议的录音、原文、纪要和行动项已保存到本机。";
  dom.welcome.hidden = true;
  dom.meetingPanel.hidden = false;
  dom.deleteMeeting.hidden = false;
  dom.recordingState.textContent = recordingLabels[state.meeting.recording_state] || stateText(state.meeting.recording_state);
  dom.recordingIndicator.classList.toggle("active", state.meeting.recording_state === "recording");
  dom.startRecordingButton.hidden = !isCreated;
  dom.startRecordingButton.disabled = !isCreated;
  dom.recordButton.hidden = !canStop;
  dom.recordButton.disabled = !canStop;
  dom.recordButton.setAttribute("aria-label", "停止会议");
  dom.recordingHint.textContent = state.meeting.error || (
    isCreated ? "请先确认输入设备和背景声过滤设置，再点击“开始录音”。"
      : state.meeting.recording_state === "recording" ? "正在接收麦克风音频。结束后会自动精修，精修完成后可生成纪要和 To-do-list。"
        : "录音已经结束，可以查看或重试后处理任务。"
  );
  dom.levelBar.style.width = `${Math.round((state.meeting.audio_level || 0) * 100)}%`;
  if (state.meeting.locked_languages) syncLanguageLockUI(state.meeting.locked_languages);
  renderLanguageLockStatus(state.meeting.current_language);
  if (transcriptChanged) renderTranscript();
  renderSummary(state.meeting.summary, state.meeting.summary_state);
  renderTodo(state.meeting.todo, state.meeting.todo_state);
  renderPostprocess(state.meeting.postprocess);
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

async function loadFullTranscript(id) {
  let offset = 0;
  const limit = 1000;
  const transcript = new Map();
  while (state.meeting?.id === id) {
    const page = await requestJson(`/api/v2/meetings/${encodeURIComponent(id)}/transcript?offset=${offset}&limit=${limit}`);
    for (const item of page.items || []) upsertUtterance(item, transcript);
    offset += (page.items || []).length;
    if (!page.has_more || !(page.items || []).length) break;
  }
  if (state.meeting?.id === id) {
    state.transcript = transcript;
    renderTranscript();
  }
}

async function refreshCurrentMeeting() {
  if (!state.meeting || ["recording", "finalizing"].includes(state.meeting.recording_state)) return;
  try {
    const previousRevision = Number(state.meeting.snapshot_revision || 0);
    const snapshot = await requestJson(`/api/v2/meetings/${encodeURIComponent(state.meeting.id)}`);
    if (state.meeting?.id === snapshot.id) {
      applySnapshot(snapshot, false);
      if (
        snapshot.recording_state === "complete"
        && Number(snapshot.snapshot_revision || 0) > previousRevision
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
      if (state.meeting?.recording_state === "created") stopAudioCapture();
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

async function refreshDevices() {
  if (!navigator.mediaDevices?.enumerateDevices) return;
  const devices = await navigator.mediaDevices.enumerateDevices();
  const microphones = devices.filter((device) => device.kind === "audioinput");
  const selected = dom.inputDevice.value;
  dom.inputDevice.replaceChildren(new Option("系统默认设备", ""));
  for (const device of microphones) dom.inputDevice.append(new Option(device.label || `麦克风 ${dom.inputDevice.length}`, device.deviceId));
  if ([...dom.inputDevice.options].some((option) => option.value === selected)) dom.inputDevice.value = selected;
  if (dom.inputDeviceSummary) dom.inputDeviceSummary.textContent = dom.inputDevice.selectedOptions[0]?.textContent || "系统默认设备";
}

async function prepareMicrophone() {
  if (state.stream) return;
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
}

async function startAudioCapture() {
  if (!state.stream || state.audioContext) return;
  state.audioContext = new AudioContext();
  await state.audioContext.audioWorklet.addModule("/static/audio-worklet.js?v=4");
  state.audioSource = state.audioContext.createMediaStreamSource(state.stream);
  state.audioNode = new AudioWorkletNode(state.audioContext, "meeting-capture-processor", { processorOptions: { targetRate: 16000, packetSamples: 640, thresholdPercent: state.volumeThresholdPercent } });
  state.audioNode.port.onmessage = (event) => {
    if (event.data?.type === "level") {
      const level = Math.min(1, Number(event.data.value) || 0);
      queueMicrophoneLevel(level);
    } else if (event.data?.type === "audio" && state.audioStreamingEnabled && state.ws?.readyState === WebSocket.OPEN) {
      state.ws.send(event.data.buffer);
    }
  };
  state.audioSource.connect(state.audioNode);
  const silentGain = state.audioContext.createGain();
  silentGain.gain.value = 0;
  state.audioNode.connect(silentGain);
  silentGain.connect(state.audioContext.destination);
  await state.audioContext.resume();
  state.audioReady = true;
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
  if (state.stream) state.stream.getTracks().forEach((track) => track.stop());
  state.stream = null;
  if (state.microphoneLevelFrame != null) window.cancelAnimationFrame(state.microphoneLevelFrame);
  state.microphoneLevelFrame = null;
  renderMicrophoneLevel(0, false);
  dom.levelBar.style.width = "0%";
  dom.levelText.textContent = "麦克风已停止";
}

async function startMicrophonePreview() {
  if (state.meeting?.recording_state !== "created") return;
  state.audioStreamingEnabled = false;
  await prepareMicrophone();
  await startAudioCapture();
  dom.levelText.textContent = "麦克风预览中";
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

async function connectStream(id) {
  if (state.reconnectTimer) { clearTimeout(state.reconnectTimer); state.reconnectTimer = null; }
  const ticket = await requestJson(`/api/v2/meetings/${encodeURIComponent(id)}/stream-ticket`, { method: "POST" });
  state.intentionalClose = false;
  const socket = new WebSocket(websocketUrl(id));
  socket.binaryType = "arraybuffer";
  state.ws = socket;
  setConnection("正在连接会议", "warning");
  socket.onopen = () => {
    state.reconnectAttempt = 0;
    socket.send(JSON.stringify({ type: "auth", ticket: ticket.ticket }));
    setConnection("会议连接中", "warning");
    // Sync language lock state on reconnect
    if (state.lockedLanguages.size > 0) {
      socket.send(JSON.stringify({ type: "language_lock", languages: getLockedLanguages() }));
    }
  };
  socket.onmessage = async (event) => {
    let payload;
    try { payload = JSON.parse(event.data); } catch { return; }
    await handleEvent(payload);
  };
  socket.onerror = () => setConnection("连接异常", "danger");
  socket.onclose = () => {
    if (state.ws === socket) state.ws = null;
    if (!state.intentionalClose && state.meeting?.id === id && ["starting", "recording", "finalizing"].includes(state.meeting.recording_state)) {
      setConnection("连接断开，正在重连", "warning");
      const delay = Math.min(5000, 400 * 2 ** state.reconnectAttempt++);
      state.reconnectTimer = setTimeout(() => connectStream(id).catch(() => {}), delay);
    } else if (state.meeting?.recording_state !== "recording") {
      setConnection("本地已保存", "neutral");
    }
  };
}

function closeStream(intentional = true) {
  if (state.reconnectTimer) { clearTimeout(state.reconnectTimer); state.reconnectTimer = null; }
  state.intentionalClose = intentional;
  if (state.ws) {
    try { state.ws.close(); } catch {}
    state.ws = null;
  }
  if (intentional) stopAudioCapture();
}

async function handleEvent(payload) {
  const type = payload.type;
  if (type === "auth_ok") {
    if (["starting", "recording"].includes(state.meeting?.recording_state)) {
      if (state.ws?.readyState === WebSocket.OPEN) state.ws.send(JSON.stringify({ type: "audio_config", sample_rate: 16000, channels: 1, encoding: "pcm_s16le", packet_ms: 40, sequence_header: true, volume_threshold_percent: state.volumeThresholdPercent }));
      setConnection("实时连接", "success");
    } else {
      setConnection("处理任务连接", "success");
    }
  } else if (type === "audio_config_ack") {
    state.audioStreamingEnabled = true;
    await startAudioCapture();
  } else if (type === "audio_threshold_ack") {
    setVolumeThreshold(payload.percent, false);
  } else if (type === "audio_threshold_error") {
    setNotice(payload.message || "音量阈值设置无效", "error");
  } else if (type === "snapshot") {
    applySnapshot(payload.meeting || payload, false);
    if (state.meeting?.recording_state === "recording") setConnection("实时连接", "success");
  } else if (type === "meeting_state") {
    if (payload.meeting) applySnapshot(payload.meeting, false);
    if (payload.message) dom.recordingHint.textContent = payload.message;
  } else if (type === "utterance") {
    if (draftMatchesUtterance(payload.utterance)) clearDraft();
    const changed = upsertUtterance(payload.utterance);
    if (state.meeting) state.meeting.utterance_count = Math.max(state.meeting.utterance_count || 0, state.transcript.size);
    if (changed) renderTranscriptItem(payload.utterance.segment_id, true);
  } else if (type === "utterance_deleted") {
    if (payload.segment_id) {
      state.transcript.delete(payload.segment_id);
      removeTranscriptItem(payload.segment_id);
    }
  } else if (type === "translation_update") {
    const item = state.transcript.get(payload.segment_id);
    if (item && Number(payload.revision || 1) >= Number(item.revision || 1)) {
      item.translation_zh = payload.translation_zh || "";
      item.translation_status = payload.translation_status || "ready";
      item.revision = payload.revision || item.revision;
      renderTranscriptItem(payload.segment_id);
    }
  } else if (type === "draft") {
    renderDraft(payload);
    dom.recordingHint.textContent = payload.text ? `正在识别：${payload.text}` : "正在接收麦克风音频。";
    if (payload.language) renderLanguageLockStatus(payload.language);
  } else if (type === "audio_input") {
    if (state.meeting) Object.assign(state.meeting, payload);
  } else if (type === "language_lock_ack") {
    syncLanguageLockUI(payload.languages || []);
    renderLanguageLockStatus(state.meeting?.current_language);
  } else if (type === "recording_complete") {
    clearDraft();
    stopAudioCapture();
    applySnapshot(payload.meeting, false);
    // ASR refinement, diarization and translation are server-side
    // jobs. Close the audio websocket deliberately so its normal shutdown is
    // never presented as an endless reconnect loop.
    closeStream(true);
    setConnection("后台处理中", "warning");
  } else if (type === "postprocess_update") {
    if (payload.meeting) applySnapshot(payload.meeting, false);
  } else if (type === "summary_progress") {
    dom.summaryProgress.hidden = false;
    if (payload.total) dom.summaryProgressBar.style.width = `${Math.round((payload.current / payload.total) * 100)}%`;
  } else if (type === "summary_delta") {
    if (state.meeting) state.meeting.summary = `${state.meeting.summary || ""}${payload.content || ""}`;
    state.summaryStreaming = true;
    scheduleSummaryRender("running");
  } else if (type === "summary_reset") {
    cancelSummaryRender();
    if (state.meeting) state.meeting.summary = "";
    state.summaryStreaming = true;
    renderSummary("", "running");
  } else if (type === "summary_complete") {
    cancelSummaryRender();
    if (state.meeting) Object.assign(state.meeting, { summary: payload.content, summary_revision: payload.summary_revision, summary_state: "complete", files: payload.files || state.meeting.files, todo_state: "queued" });
    state.summaryStreaming = false;
    renderSummary(payload.content, "complete");
    renderTodo(null, "queued");
    renderFiles(state.meeting?.files || []);
  } else if (type === "todo_progress") {
    if (state.meeting) state.meeting.todo_state = "running";
    renderTodo(null, "running");
  } else if (type === "todo_complete") {
    if (state.meeting) Object.assign(state.meeting, { todo: payload.todo, todo_state: "complete", files: payload.files || state.meeting.files });
    renderTodo(payload.todo, "complete");
    renderFiles(state.meeting?.files || []);
  } else if (type === "warning") {
    setNotice(payload.message || "处理出现警告", "warning");
  } else if (type === "error") {
    setNotice(payload.message || "处理失败", "error");
    if (state.meeting && payload.code === "summary_failed") Object.assign(state.meeting, {
      summary_state: "error",
      summary: payload.summary ?? state.meeting.summary,
      summary_revision: payload.summary_revision ?? state.meeting.summary_revision,
    });
    if (state.meeting && payload.code === "todo_failed") state.meeting.todo_state = "error";
    if (payload.code === "summary_failed") {
      state.summaryStreaming = false;
      cancelSummaryRender();
    }
    renderSummary(state.meeting?.summary, state.meeting?.summary_state || "error");
    renderTodo(state.meeting?.todo, state.meeting?.todo_state || "error");
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
  setNotice("正在申请麦克风权限…", "info");
  try {
    await checkHealth();
    await prepareMicrophone();
    state.audioStreamingEnabled = false;
    const snapshot = await requestJson(`/api/v2/meetings/${encodeURIComponent(state.meeting.id)}/start`, { method: "POST" });
    applySnapshot(snapshot, false);
    await connectStream(snapshot.id);
    setNotice("");
  } catch (error) {
    stopAudioCapture();
    dom.startRecordingButton.disabled = false;
    setNotice(error.message || "无法开始录音", "error");
  }
}

async function stopMeeting() {
  if (!state.meeting || !["starting", "recording"].includes(state.meeting.recording_state)) return;
  dom.recordButton.disabled = true;
  dom.recordingHint.textContent = "正在保存最后一个语音片段，请稍候…";
  try { await requestJson(`/api/v2/meetings/${encodeURIComponent(state.meeting.id)}/stop`, { method: "POST" }); }
  catch (error) { setNotice(error.message, "error"); dom.recordButton.disabled = false; }
}

async function retrySummary() {
  if (!state.meeting) return;
  try { await requestJson(`/api/v2/meetings/${encodeURIComponent(state.meeting.id)}/summary`, { method: "POST" }); state.meeting.summary_state = "running"; state.summaryStreaming = true; renderSummary(state.meeting.summary, "running"); setNotice("正在生成会议纪要，完成后会自动生成 To-do-list。", "info"); }
  catch (error) { setNotice(error.message, "error"); }
}

async function retryTodo() {
  if (!state.meeting) return;
  try { await requestJson(`/api/v2/meetings/${encodeURIComponent(state.meeting.id)}/todo`, { method: "POST" }); state.meeting.todo_state = "running"; renderTodo(state.meeting.todo, "running"); }
  catch (error) { setNotice(error.message, "error"); }
}

async function retryPostprocess() {
  if (!state.meeting) return;
  try {
    await requestJson(`/api/v2/meetings/${encodeURIComponent(state.meeting.id)}/postprocess`, { method: "POST" });
    setNotice("后台处理已重新排队", "info");
  } catch (error) { setNotice(error.message, "error"); }
}

async function deleteMeeting() {
  if (!state.meeting || !window.confirm(`确定删除“${state.meeting.title}”及其录音和文件吗？`)) return;
  try {
    await requestJson(`/api/v2/meetings/${encodeURIComponent(state.meeting.id)}`, { method: "DELETE" });
    closeStream(true);
    state.meeting = null;
    clearTranscript();
    dom.welcome.hidden = false;
    dom.meetingPanel.hidden = true;
    dom.deleteMeeting.hidden = true;
    dom.pageTitle.textContent = "";
    dom.pageSubtitle.textContent = "";
    renderSummary("", "idle");
    renderTodo(null, "waiting_summary");
    dom.filesCard.hidden = true;
    await loadMeetings();
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

function getLockedLanguages() {
  return [...dom.languageLockGroup.querySelectorAll(".language-lock-btn.active")].map((el) => el.dataset.lang);
}

function syncLanguageLockUI(languages) {
  const target = new Set(languages || []);
  for (const btn of dom.languageLockGroup.querySelectorAll(".language-lock-btn")) {
    btn.classList.toggle("active", target.has(btn.dataset.lang));
  }
  state.lockedLanguages = target;
}

function sendLanguageLock() {
  const languages = getLockedLanguages();
  state.lockedLanguages = new Set(languages);
  if (state.ws?.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify({ type: "language_lock", languages }));
  }
}

function renderLanguageLockStatus(detectedLang) {
  // Visual feedback: when locked, show detected language alongside locks
  const count = state.lockedLanguages.size;
  if (count === 0 && detectedLang) {
    dom.languageLockGroup.title = `自动检测：${detectedLang.toUpperCase()}`;
  } else if (count > 0 && detectedLang) {
    dom.languageLockGroup.title = `锁定：${[...state.lockedLanguages].join(", ").toUpperCase()} | 当前识别：${detectedLang.toUpperCase()}`;
  } else {
    dom.languageLockGroup.title = count > 0 ? `已锁定：${[...state.lockedLanguages].join(", ").toUpperCase()}` : "自动检测语言（点击按钮锁定）";
  }
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
  $("#refreshMeetings").addEventListener("click", () => loadMeetings().catch((error) => setNotice(error.message, "error")));
  dom.search.addEventListener("input", renderMeetings);
  dom.startRecordingButton.addEventListener("click", startRecording);
  $("#recordButton").addEventListener("click", stopMeeting);
  dom.deleteMeeting.addEventListener("click", deleteMeeting);
  dom.retrySummary.addEventListener("click", retrySummary);
  dom.retryTodo.addEventListener("click", retryTodo);
  dom.retryPostprocess.addEventListener("click", retryPostprocess);
  dom.downloadSummary.addEventListener("click", () => downloadFile("meeting_minutes.md"));
  dom.downloadTodo.addEventListener("click", () => downloadFile("todo_list.json"));
  dom.openAsrSettings.addEventListener("click", openAsrSettings);
  dom.asrSettingsForm.addEventListener("submit", (event) => {
    if (event.submitter?.value !== "default") return;
    event.preventDefault();
    saveAsrSettings();
  });
  dom.resetAsrSettings.addEventListener("click", () => applySettingsTemplate("balanced"));
  dom.settingsTemplateSelect?.addEventListener("change", () => {
    const selected = dom.settingsTemplateSelect.value;
    if (selected === "custom") {
      updateTemplatePicker("custom");
      return;
    }
    applySettingsTemplate(selected);
  });
  for (const tab of dom.settingsTabs) {
    tab.addEventListener("click", () => {
      const selected = tab.dataset.settingsTab;
      for (const item of dom.settingsTabs) {
        const active = item.dataset.settingsTab === selected;
        item.classList.toggle("active", active);
        item.setAttribute("aria-selected", String(active));
      }
      for (const panel of dom.settingsPanels) panel.hidden = panel.dataset.settingsPanel !== selected;
    });
  }
  for (const field of MEETING_NUMBER_FIELDS) {
    const slider = $(`#${field.slider}`);
    const input = $(`#${field.input}`);
    slider?.addEventListener("input", (event) => {
      setMeetingField(field, event.target.value, { updateState: false });
      markSettingsCustom();
    });
    input?.addEventListener("input", (event) => {
      if (event.target.value !== "" && Number.isFinite(Number(event.target.value))) {
        setMeetingField(field, event.target.value, { updateState: false });
        markSettingsCustom();
      }
    });
    input?.addEventListener("change", (event) => {
      setMeetingField(field, event.target.value, { updateState: false });
      markSettingsCustom();
    });
  }
  for (const [key, id] of Object.entries({
    enable_refinement: "settingEnableRefinement",
    enable_postprocess: "settingEnablePostprocess",
    diarization_required: "settingDiarizationRequired",
    keep_audio: "settingKeepAudio",
  })) {
    $(`#${id}`)?.addEventListener("change", markSettingsCustom);
  }
  $("#refreshDevices").addEventListener("click", () => refreshDevices().catch(() => {}));
  dom.volumeThreshold.addEventListener("input", () => {
    setVolumeThreshold(dom.volumeThreshold.value);
    markSettingsCustom();
  });
  dom.volumeThresholdValue.addEventListener("input", () => {
    if (dom.volumeThresholdValue.value !== "" && Number.isFinite(Number(dom.volumeThresholdValue.value))) {
      setVolumeThreshold(dom.volumeThresholdValue.value);
      markSettingsCustom();
    }
  });
  dom.volumeThresholdValue.addEventListener("change", () => {
    setVolumeThreshold(dom.volumeThresholdValue.value);
    markSettingsCustom();
  });
  dom.inputDevice.addEventListener("change", async () => {
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
  // Language lock buttons — toggle active state and send to backend
  dom.languageLockGroup.addEventListener("click", (e) => {
    const btn = e.target.closest(".language-lock-btn");
    if (!btn) return;
    btn.classList.toggle("active");
    sendLanguageLock();
    renderLanguageLockStatus(state.meeting?.current_language);
  });
  dom.transcriptList.addEventListener("scroll", () => {
    const distance = dom.transcriptList.scrollHeight - dom.transcriptList.scrollTop - dom.transcriptList.clientHeight;
    state.transcriptNearBottom = distance < 80;
    dom.jumpLatest.hidden = state.transcriptNearBottom || !state.transcript.size;
  });
  dom.jumpLatest.addEventListener("click", () => { state.transcriptNearBottom = true; renderTranscript(true); });
}

async function init() {
  moveAudioSettingsIntoDialog();
  bindEvents();
  state.timer = window.setInterval(updateTimer, 1000);
  window.setInterval(refreshCurrentMeeting, 3000);
  try {
    await checkHealth();
    await loadMeetings();
    await refreshDevices();
  } catch (error) { setNotice(error.message || "无法连接本机服务", "error"); }
  renderSummary("", "idle");
  renderTodo(null, "waiting_summary");
}

window.addEventListener("beforeunload", () => closeStream(true));
init();
