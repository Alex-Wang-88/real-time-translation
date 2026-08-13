const $ = (selector) => document.querySelector(selector);

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
  transcriptNearBottom: true,
  timer: null,
  volumeThresholdPercent: 2.2,
  microphoneLevelPercent: 0,
  pendingMicrophoneLevel: 0,
  microphoneLevelFrame: null,
  audioStreamingEnabled: false,
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
  languageIndicator: $("#languageIndicator"),
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
  if (dom.volumeThreshold) dom.volumeThreshold.value = String(state.volumeThresholdPercent);
  if (dom.volumeThresholdValue) dom.volumeThresholdValue.textContent = `${state.volumeThresholdPercent.toFixed(1)}%`;
  renderMicrophoneLevel(state.microphoneLevelPercent, Boolean(state.stream));
  if (!propagate) return;
  state.audioNode?.port.postMessage({ type: "volume_threshold", percent: state.volumeThresholdPercent });
  if (state.meeting) state.meeting.volume_threshold_percent = state.volumeThresholdPercent;
  if (state.ws?.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify({ type: "audio_threshold", percent: state.volumeThresholdPercent }));
  }
}

function renderMicrophoneLevel(levelPercent, live = true) {
  const level = Math.max(0, Math.min(100, Number(levelPercent) || 0));
  const meterMax = Number(dom.volumeThreshold?.max) || 30;
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
  state.transcript.clear();
  dom.transcriptList.replaceChildren();
  dom.transcriptList.append(dom.transcriptEmpty);
  dom.transcriptEmpty.hidden = false;
}

function upsertUtterance(utterance, transcript = state.transcript) {
  if (!utterance || !utterance.segment_id) return;
  if (utterance.deleted) {
    const current = transcript.get(utterance.segment_id);
    if (!current || Number(utterance.revision || 1) >= Number(current.revision || 1)) transcript.delete(utterance.segment_id);
    return;
  }
  const previous = transcript.get(utterance.segment_id);
  if (!previous || Number(utterance.revision || 1) >= Number(previous.revision || 1)) {
    transcript.set(utterance.segment_id, utterance);
  }
}

function renderTranscript(autoScroll = false) {
  const items = [...state.transcript.values()].sort((a, b) => (a.start || 0) - (b.start || 0));
  dom.transcriptList.replaceChildren();
  if (!items.length) {
    dom.transcriptList.append(dom.transcriptEmpty);
    dom.transcriptEmpty.hidden = false;
  } else {
    dom.transcriptEmpty.hidden = true;
    for (const item of items) {
      const article = document.createElement("article");
      article.className = "transcript-item";
      const lang = item.language || "unknown";
      const translation = item.translation_zh && lang !== "zh" ? `<div class="translation-line"><span>中译</span><b></b></div>` : "";
      article.innerHTML = `<div class="transcript-meta"><time></time><span class="speaker-tag"></span><span class="language-tag"></span></div><p class="original-line"></p>${translation}`;
      article.querySelector("time").textContent = `${formatTime(item.start).slice(3)} – ${formatTime(item.end).slice(3)}`;
      article.querySelector(".speaker-tag").textContent = `演讲人 ${item.speaker_id ?? "?"}`;
      article.querySelector(".language-tag").textContent = lang.toUpperCase();
      article.querySelector(".original-line").textContent = item.text || "";
      if (translation) article.querySelector(".translation-line b").textContent = item.translation_zh;
      dom.transcriptList.append(article);
    }
  }
  dom.utteranceCount.textContent = `${state.meeting?.utterance_count || items.length} 条记录`;
  const shouldScroll = autoScroll || state.transcriptNearBottom;
  if (shouldScroll) requestAnimationFrame(() => { dom.transcriptList.scrollTop = dom.transcriptList.scrollHeight; });
  dom.jumpLatest.hidden = shouldScroll || !items.length;
}

function renderSummary(summary, summaryState) {
  const value = String(summary || "").trim();
  const stages = state.meeting?.postprocess?.stages || {};
  const preprocessingReady = ["asr_refine", "diarization", "translation"].every((key) => stages[key]?.state === "complete");
  dom.summaryBadge.textContent = summaryLabels[summaryState] || stateText(summaryState);
  dom.summaryBadge.className = `badge ${summaryState === "complete" ? "success" : summaryState === "error" ? "danger" : "neutral"}`;
  dom.summaryText.classList.toggle("empty-result", !value);
  dom.summaryText.innerHTML = value
    ? markdownToHtml(value)
    : preprocessingReady
      ? "ASR 精修、说话人重排和翻译已经完成，可以生成会议纪要和 To-do-list。"
      : "停止会议后会先自动完成 ASR 精修、说话人重排和翻译。";
  dom.retrySummary.hidden = !(preprocessingReady && ["idle", "error", "complete"].includes(summaryState));
  dom.retrySummary.disabled = !preprocessingReady || summaryState === "running";
  dom.retrySummary.textContent = summaryState === "complete" ? "重新生成纪要和 To-do-list" : "生成纪要和 To-do-list";
  dom.downloadSummary.hidden = !value || !state.meeting?.files?.includes("meeting_minutes.md");
  dom.summaryProgress.hidden = !["queued", "running"].includes(summaryState);
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
  if (changed && snapshot.volume_threshold_percent != null) setVolumeThreshold(snapshot.volume_threshold_percent, false);
  const isCreated = state.meeting.recording_state === "created";
  const canAdjustAudio = ["created", "starting", "recording"].includes(state.meeting.recording_state);
  const canStop = ["starting", "recording"].includes(state.meeting.recording_state);
  dom.volumeThreshold.disabled = !canAdjustAudio;
  const meetingIndex = state.meetings.findIndex((meeting) => meeting.id === snapshot.id);
  if (meetingIndex >= 0) state.meetings[meetingIndex] = { ...state.meetings[meetingIndex], ...snapshot };
  else state.meetings.unshift(snapshot);
  if (changed || replace) {
    clearTranscript();
  }
  for (const item of snapshot.recent_utterances || []) upsertUtterance(item);
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
  dom.languageIndicator.textContent = state.meeting.current_language ? `最近语言：${String(state.meeting.current_language).toUpperCase()}` : "等待语音";
  renderTranscript();
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
    upsertUtterance(payload.utterance);
    if (state.meeting) state.meeting.utterance_count = Math.max(state.meeting.utterance_count || 0, state.transcript.size);
    renderTranscript(true);
  } else if (type === "utterance_deleted") {
    if (payload.segment_id) state.transcript.delete(payload.segment_id);
    renderTranscript();
  } else if (type === "translation_update") {
    const item = state.transcript.get(payload.segment_id);
    if (item && Number(payload.revision || 1) >= Number(item.revision || 1)) {
      item.translation_zh = payload.translation_zh || "";
      item.translation_status = payload.translation_status || "ready";
      item.revision = payload.revision || item.revision;
      renderTranscript();
    }
  } else if (type === "draft") {
    dom.recordingHint.textContent = payload.text ? `正在识别：${payload.text}` : "正在接收麦克风音频。";
    if (payload.language) dom.languageIndicator.textContent = `当前语言：${String(payload.language).toUpperCase()}`;
  } else if (type === "audio_input") {
    if (state.meeting) Object.assign(state.meeting, payload);
  } else if (type === "recording_complete") {
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
    renderSummary(state.meeting?.summary, "running");
  } else if (type === "summary_reset") {
    if (state.meeting) state.meeting.summary = "";
    renderSummary("", "running");
  } else if (type === "summary_complete") {
    if (state.meeting) Object.assign(state.meeting, { summary: payload.content, summary_revision: payload.summary_revision, summary_state: "complete", files: payload.files || state.meeting.files, todo_state: "queued" });
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
  try { await requestJson(`/api/v2/meetings/${encodeURIComponent(state.meeting.id)}/summary`, { method: "POST" }); state.meeting.summary_state = "running"; renderSummary(state.meeting.summary, "running"); setNotice("正在生成会议纪要，完成后会自动生成 To-do-list。", "info"); }
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
  $("#refreshDevices").addEventListener("click", () => refreshDevices().catch(() => {}));
  dom.volumeThreshold.addEventListener("input", () => setVolumeThreshold(dom.volumeThreshold.value));
  dom.inputDevice.addEventListener("change", async () => {
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
}

async function init() {
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
