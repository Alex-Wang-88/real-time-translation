const ui = {
  backendPill: document.querySelector("#backendPill"),
  gpuPill: document.querySelector("#gpuPill"),
  jimoPill: document.querySelector("#jimoPill"),
  mainButton: document.querySelector("#mainButton"),
  mainButtonText: document.querySelector("#mainButtonText"),
  timer: document.querySelector("#timer"),
  languageBadge: document.querySelector("#languageBadge"),
  levelBar: document.querySelector("#levelBar"),
  levelTrack: document.querySelector(".level-track"),
  levelValue: document.querySelector("#levelValue"),
  micStatus: document.querySelector("#micStatus"),
  audioDebug: document.querySelector("#audioDebug"),
  hotwords: document.querySelector("#hotwords"),
  deviceSelect: document.querySelector("#deviceSelect"),
  deviceHint: document.querySelector("#deviceHint"),
  statusMessage: document.querySelector("#statusMessage"),
  steps: [...document.querySelectorAll("#steps li")],
  notice: document.querySelector("#notice"),
  transcriptList: document.querySelector("#transcriptList"),
  transcriptEmpty: document.querySelector("#transcriptEmpty"),
  utteranceCount: document.querySelector("#utteranceCount"),
  jumpLatest: document.querySelector("#jumpLatest"),
  summaryText: document.querySelector("#summaryText"),
  retrySummary: document.querySelector("#retrySummary"),
  downloads: document.querySelector("#downloads"),
  downloadLinks: document.querySelector("#downloadLinks"),
  diskStatus: document.querySelector("#diskStatus"),
};

const state = {
  health: null,
  sessionId: null,
  meetingState: "idle",
  ws: null,
  stream: null,
  audioContext: null,
  worklet: null,
  source: null,
  timerHandle: null,
  startedAt: 0,
  utterances: 0,
  following: true,
  stopping: false,
  restored: false,
  audioWatchdog: null,
  captureStartedAt: 0,
  audioFramesProduced: 0,
  audioFramesSent: 0,
  audioBytesSent: 0,
  backendAudioPackets: 0,
  lastAudioLevel: 0,
  lastAudioLevelAt: 0,
  lastNonZeroLevelAt: 0,
  audioWarningShown: false,
  deviceRequest: (() => { try { return localStorage.getItem("meeting_device") || "auto"; } catch { return "auto"; } })(),
  switchingDevice: false,
};

const languageNames = {
  zh: "中文", en: "英文", de: "德文", ru: "俄文", es: "西班牙文",
  pt: "葡萄牙文", fr: "法文", it: "意大利文", ja: "日文", ko: "韩文",
  ar: "阿拉伯文", uk: "乌克兰文", pl: "波兰文", nl: "荷兰文", tr: "土耳其文",
  vi: "越南文",
};
const stageIndex = {
  checking: 0, loading: 0, microphone: 1, recording: 2, listening: 2,
  transcribing: 3, translating: 4, finalizing: 5, saving: 5,
  summarizing: 6, summarizing_chunks: 6, summarizing_final: 6, complete: 7,
};

function setPill(element, status, detail) {
  element.classList.toggle("ready", status === "ready");
  element.classList.toggle("error", status === "error");
  element.querySelector("em").textContent = detail;
}

function formatBytes(value) {
  if (!Number.isFinite(value)) return "—";
  if (value >= 1024 ** 3) return `${(value / 1024 ** 3).toFixed(1)} GB 可用`;
  return `${(value / 1024 ** 2).toFixed(0)} MB 可用`;
}

function setMicLevel(value) {
  const normalized = Number.isFinite(value) ? Math.max(0, Math.min(1, value)) : 0;
  const percent = Math.min(100, Math.max(0, Math.round(normalized * 420)));
  ui.levelBar.style.width = `${Math.max(2, percent)}%`;
  ui.levelValue.textContent = `${percent}%`;
  ui.levelTrack?.setAttribute("aria-valuenow", String(percent));
  state.lastAudioLevel = normalized;
  state.lastAudioLevelAt = performance.now();
  if (normalized >= 0.003) state.lastNonZeroLevelAt = state.lastAudioLevelAt;
  if (percent >= 4) ui.micStatus.textContent = "正在接收声音";
  else if (state.meetingState === "recording") ui.micStatus.textContent = "已连接，等待发言";
}

function resetMicFeedback(message = "等待麦克风") {
  if (state.audioWatchdog) window.clearInterval(state.audioWatchdog);
  state.audioWatchdog = null;
  state.captureStartedAt = 0;
  state.audioFramesProduced = 0;
  state.audioFramesSent = 0;
  state.audioBytesSent = 0;
  state.backendAudioPackets = 0;
  state.lastAudioLevel = 0;
  state.lastAudioLevelAt = 0;
  state.lastNonZeroLevelAt = 0;
  state.audioWarningShown = false;
  ui.levelBar.style.width = "2%";
  ui.levelValue.textContent = "0%";
  ui.levelTrack?.setAttribute("aria-valuenow", "0");
  ui.micStatus.textContent = message;
  ui.audioDebug.textContent = "开始会议后会显示音频帧和后端接收状态。";
}

function updateAudioDebug() {
  const backend = state.backendAudioPackets
    ? ` · 后端已收 ${state.backendAudioPackets} 包`
    : " · 后端暂未确认音频";
  ui.audioDebug.textContent = `前端 ${state.audioFramesSent} 帧 / ${formatAudioBytes(state.audioBytesSent)}${backend}`;
}

function formatAudioBytes(value) {
  if (!Number.isFinite(value) || value <= 0) return "0 B";
  if (value < 1024) return `${Math.round(value)} B`;
  if (value < 1024 ** 2) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / 1024 ** 2).toFixed(1)} MB`;
}

function formatClock(seconds) {
  const value = Math.max(0, Math.floor(seconds));
  const h = String(Math.floor(value / 3600)).padStart(2, "0");
  const m = String(Math.floor((value % 3600) / 60)).padStart(2, "0");
  const s = String(value % 60).padStart(2, "0");
  return `${h}:${m}:${s}`;
}

function formatTimestamp(seconds) {
  const ms = Math.max(0, Math.round(seconds * 1000));
  const h = String(Math.floor(ms / 3600000)).padStart(2, "0");
  const m = String(Math.floor((ms % 3600000) / 60000)).padStart(2, "0");
  const s = String(Math.floor((ms % 60000) / 1000)).padStart(2, "0");
  const millis = String(ms % 1000).padStart(3, "0");
  return `${h}:${m}:${s}.${millis}`;
}

function setStage(stage, message) {
  const index = stageIndex[stage] ?? 0;
  ui.steps.forEach((item, position) => {
    item.classList.toggle("done", position < index || stage === "complete");
    item.classList.toggle("current", position === index && stage !== "complete");
  });
  ui.statusMessage.textContent = message;
}

function showNotice(message, kind = "error") {
  ui.notice.textContent = message;
  ui.notice.className = `notice ${kind === "warning" ? "warning" : ""}`;
  ui.notice.hidden = false;
}

function clearNotice() {
  ui.notice.hidden = true;
  ui.notice.textContent = "";
}

function updateMainButton() {
  const ready = state.health?.status === "ready";
  const recording = state.meetingState === "recording";
  const busy = ["finalizing", "summarizing"].includes(state.meetingState) || state.stopping;
  ui.mainButton.classList.toggle("recording", recording);
  ui.mainButton.disabled = busy || (!recording && !ready);
  ui.hotwords.disabled = recording || busy;
  if (recording) {
    ui.mainButtonText.textContent = "停止会议";
    ui.mainButton.setAttribute("aria-label", "停止会议");
  } else if (busy) {
    ui.mainButtonText.textContent = "正在完成";
  } else if (ready) {
    ui.mainButtonText.textContent = "开始会议";
    ui.mainButton.setAttribute("aria-label", "开始会议");
  } else {
    ui.mainButtonText.textContent = "模型加载中";
  }
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  let payload = null;
  try { payload = await response.json(); } catch { payload = {}; }
  if (!response.ok) throw new Error(payload.detail || `请求失败 (${response.status})`);
  return payload;
}

async function pollHealth() {
  try {
    const health = await requestJson("/api/health");
    state.health = health;
    setPill(ui.backendPill, health.status, health.status === "ready" ? "已就绪" : health.message);
    setPill(ui.gpuPill, health.status === "error" ? "error" : "ready", health.device?.toUpperCase() || "加载中");
    setPill(ui.jimoPill, health.jimo_configured ? "ready" : "error", health.jimo_configured ? "已配置" : "未配置");
    ui.diskStatus.textContent = formatBytes(health.disk_free_bytes);
    if (health.status === "loading" && state.meetingState === "idle") setStage("loading", health.message || "正在加载模型…");
    if (health.status === "ready" && state.meetingState === "idle") setStage("checking", "服务和模型已就绪，可以开始会议");
    if (health.status === "error") showNotice(health.message || "模型加载失败");
    if (health.active_session && !state.restored && !state.sessionId) {
      state.restored = true;
      restoreMeeting(health.active_session);
    }
    // Reflect an in-flight or failed device switch driven by the backend.
    if (state.switchingDevice) {
      if (health.switch_error && !health.switching) {
        state.switchingDevice = false;
        const fallback = health.device === "cuda" ? "cuda" : health.device === "cpu" ? "cpu" : "auto";
        state.deviceRequest = fallback;
        ui.deviceSelect.value = fallback;
        showNotice(`切换失败：${health.switch_error}（继续使用原设备）`, "warning");
      } else if (!health.switching && !health.switch_error) {
        state.switchingDevice = false;
      }
    }
  } catch (error) {
    state.health = { status: "error" };
    setPill(ui.backendPill, "error", "无法连接");
    showNotice(`无法连接本机服务：${error.message}`);
  }
  updateMainButton();
  updateDeviceControls();
}

function deviceLabel(value) {
  if (value === "cuda") return "GPU（CUDA）";
  if (value === "cpu") return "CPU";
  return "自动（推荐）";
}

function updateDeviceControls() {
  const health = state.health || {};
  const recording = state.meetingState === "recording";
  const busy = ["finalizing", "summarizing"].includes(state.meetingState) || state.stopping;
  if (state.switchingDevice) {
    ui.deviceSelect.disabled = true;
    ui.deviceHint.textContent = "正在切换推理设备，请稍候（约 10–60 秒）…";
    return;
  }
  if (ui.deviceSelect.value !== state.deviceRequest) ui.deviceSelect.value = state.deviceRequest;
  ui.deviceSelect.disabled = recording || busy;
  if (health.switch_error) {
    ui.deviceHint.textContent = `切换失败：${health.switch_error}（继续使用原设备）`;
  } else if (health.device) {
    ui.deviceHint.textContent = `当前运行：${deviceLabel(health.device)}`;
  } else {
    ui.deviceHint.textContent = "有独显时自动启用 GPU 加速";
  }
}

async function onDeviceChange() {
  const requested = ui.deviceSelect.value;
  state.deviceRequest = requested;
  try { localStorage.setItem("meeting_device", requested); } catch { /* ignore */ }
  state.switchingDevice = true;
  updateDeviceControls();
  try {
    await requestJson("/api/device", { method: "POST", body: JSON.stringify({ device: requested }) });
  } catch (error) {
    state.switchingDevice = false;
    const cur = state.health?.device === "cuda" ? "cuda" : state.health?.device === "cpu" ? "cpu" : "auto";
    state.deviceRequest = cur;
    ui.deviceSelect.value = cur;
    showNotice(error.message, "warning");
    updateDeviceControls();
  }
}

function startTimer(offset = 0) {
  stopTimer();
  state.startedAt = performance.now() - offset * 1000;
  state.timerHandle = window.setInterval(() => {
    ui.timer.textContent = formatClock((performance.now() - state.startedAt) / 1000);
  }, 250);
}

function stopTimer() {
  if (state.timerHandle) window.clearInterval(state.timerHandle);
  state.timerHandle = null;
}

async function connectWebSocket(sessionId) {
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${scheme}://${location.host}/api/meetings/${sessionId}/stream`);
  ws.binaryType = "arraybuffer";
  await new Promise((resolve, reject) => {
    const timer = window.setTimeout(() => reject(new Error("实时连接超时")), 6000);
    ws.addEventListener("open", () => { window.clearTimeout(timer); resolve(); }, { once: true });
    ws.addEventListener("error", () => { window.clearTimeout(timer); reject(new Error("无法建立实时连接")); }, { once: true });
  });
  state.ws = ws;
  ws.addEventListener("message", (event) => {
    try { handleServerEvent(JSON.parse(event.data)); } catch { /* Ignore malformed status frames. */ }
  });
  ws.addEventListener("close", () => {
    if (state.meetingState === "recording") {
      stopCapture();
      state.meetingState = "finalizing";
      state.stopping = true;
      showNotice("实时连接已断开，麦克风已停止；后端会保存已收到的记录。", "warning");
      setStage("finalizing", "连接中断，正在保存已有记录");
      updateMainButton();
    }
  });
}

async function startCapture(stream) {
  const context = new AudioContext({ latencyHint: "interactive" });
  await context.audioWorklet.addModule("/static/audio-worklet.js");
  if (context.state === "suspended") await context.resume();
  if (context.state !== "running") throw new Error(`音频引擎未运行（${context.state}）`);
  const track = stream.getAudioTracks()[0];
  if (!track || track.readyState !== "live") throw new Error("没有可用的麦克风音轨");
  const source = context.createMediaStreamSource(stream);
  const worklet = new AudioWorkletNode(context, "pcm-capture-processor");
  const silent = context.createGain();
  silent.gain.value = 0;
  source.connect(worklet);
  worklet.connect(silent);
  silent.connect(context.destination);
  resetMicFeedback("麦克风已连接，等待发言");
  state.captureStartedAt = performance.now();
  worklet.port.onmessage = (event) => {
    if (event.data?.type === "audio") {
      state.audioFramesProduced += 1;
      const buffer = event.data.buffer;
      if (state.ws?.readyState === WebSocket.OPEN) {
        try {
          state.ws.send(buffer);
          state.audioFramesSent += 1;
          state.audioBytesSent += buffer.byteLength || 0;
        } catch (error) {
          ui.audioDebug.textContent = `音频发送失败：${error.message}`;
        }
      }
      updateAudioDebug();
    } else if (event.data?.type === "level") {
      setMicLevel(event.data.value);
    }
  };
  state.audioContext = context;
  state.source = source;
  state.worklet = worklet;
  state.stream = stream;
  const settings = track.getSettings ? track.getSettings() : {};
  const device = track.label || (settings.deviceId ? "已选择输入设备" : "默认输入设备");
  ui.audioDebug.textContent = `${device} · ${settings.sampleRate || context.sampleRate} Hz · 等待后端确认`;
  if (track.muted) ui.micStatus.textContent = "输入设备当前静音";
  track.addEventListener("mute", () => {
    ui.micStatus.textContent = "输入设备已静音";
    ui.audioDebug.textContent = "浏览器报告麦克风音轨 muted，请检查 Windows 输入设备和隐私权限。";
  });
  track.addEventListener("unmute", () => {
    ui.micStatus.textContent = "麦克风已恢复，等待发言";
    updateAudioDebug();
  });
  track.addEventListener("ended", () => {
    ui.micStatus.textContent = "麦克风设备已断开";
    ui.audioDebug.textContent = "麦克风音轨已结束，请重新连接设备后重试。";
  });
  state.audioWatchdog = window.setInterval(() => {
    if (state.meetingState !== "recording") return;
    const elapsed = performance.now() - state.captureStartedAt;
    const silentFor = state.lastNonZeroLevelAt ? performance.now() - state.lastNonZeroLevelAt : elapsed;
    if (!state.audioFramesProduced && elapsed > 1500) {
      ui.micStatus.textContent = "未产生音频帧";
      ui.audioDebug.textContent = "浏览器未输出音频帧，请检查麦克风权限和输入设备。";
      if (elapsed > 3500 && !state.audioWarningShown) {
        state.audioWarningShown = true;
        showNotice("麦克风已授权，但没有产生音频帧。请检查 Windows 输入设备、浏览器麦克风权限，并重新开始。", "warning");
      }
    } else if (silentFor > 1800 && state.lastAudioLevel < 0.004) {
      ui.micStatus.textContent = "已连接，当前音量很低";
      ui.audioDebug.textContent = `已发送 ${state.audioFramesSent} 帧，但输入电平接近 0${state.backendAudioPackets ? " · 后端已收到" : ""}；请检查系统麦克风音量。`;
    } else {
      updateAudioDebug();
    }
  }, 500);
}

async function stopCapture() {
  if (state.worklet) state.worklet.port.postMessage({ type: "flush" });
  await new Promise((resolve) => window.setTimeout(resolve, 80));
  state.stream?.getTracks().forEach((track) => track.stop());
  try { state.source?.disconnect(); } catch { /* already disconnected */ }
  try { state.worklet?.disconnect(); } catch { /* already disconnected */ }
  if (state.audioContext && state.audioContext.state !== "closed") await state.audioContext.close();
  state.stream = null;
  state.source = null;
  state.worklet = null;
  state.audioContext = null;
  resetMicFeedback("录音已停止");
}

async function startMeeting() {
  clearNotice();
  setStage("microphone", "正在请求麦克风权限");
  let requestedStream = null;
  let createdMeetingId = null;
  try {
    requestedStream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      video: false,
    });
    const meeting = await requestJson("/api/meetings", {
      method: "POST",
      body: JSON.stringify({ hotwords: ui.hotwords.value.trim() || null }),
    });
    createdMeetingId = meeting.id;
    state.sessionId = meeting.id;
    state.meetingState = "recording";
    state.stopping = false;
    resetMeetingUi();
    await connectWebSocket(meeting.id);
    await startCapture(requestedStream);
    startTimer(meeting.elapsed_seconds || 0);
    setStage("recording", "麦克风已开启，正在监听语音");
    updateMainButton();
  } catch (error) {
    requestedStream?.getTracks().forEach((track) => track.stop());
    state.stream?.getTracks().forEach((track) => track.stop());
    if (createdMeetingId) {
      try {
        await requestJson(`/api/meetings/${createdMeetingId}/stop`, { method: "POST", body: "{}" });
      } catch {
        // The server also finalizes an orphaned meeting when its socket disconnects.
      }
    }
    state.sessionId = null;
    state.meetingState = "idle";
    showNotice(`无法开始会议：${error.message}`);
    setStage("microphone", "请允许麦克风权限后重试");
    updateMainButton();
  }
}

async function stopMeeting() {
  if (!state.sessionId || state.stopping) return;
  state.stopping = true;
  state.meetingState = "finalizing";
  updateMainButton();
  setStage("finalizing", "正在停止麦克风并处理最后一段语音");
  try {
    await stopCapture();
    await requestJson(`/api/meetings/${state.sessionId}/stop`, { method: "POST", body: "{}" });
  } catch (error) {
    showNotice(`停止会议时出现问题：${error.message}`);
  }
}

function resetMeetingUi() {
  state.utterances = 0;
  ui.utteranceCount.textContent = "0 条稳定记录";
  ui.transcriptList.replaceChildren(ui.transcriptEmpty);
  ui.transcriptEmpty.hidden = false;
  ui.summaryText.textContent = "停止录音后，完整逐句稿会分块发送给积墨 AI。纪要将在此处流式生成。";
  ui.summaryText.classList.remove("streaming");
  ui.downloads.hidden = true;
  ui.retrySummary.hidden = true;
  ui.languageBadge.textContent = "等待发言";
}

function sourceLine(item) {
  return `[${formatTimestamp(item.start)} - ${formatTimestamp(item.end)}] 演讲人${item.speaker_id}（${languageNames[item.language] || item.language}）：“${item.text}”`;
}

function translationLine(item) {
  return `[${formatTimestamp(item.start)} - ${formatTimestamp(item.end)}] 演讲人${item.speaker_id}（中文翻译）：“${item.translation_zh ?? item.translation_en ?? ""}”`;
}

function appendUtterance(item) {
  ui.transcriptEmpty.hidden = true;
  document.querySelectorAll(".utterance.latest").forEach((node) => node.classList.remove("latest"));
  document.querySelector("#partialUtterance")?.remove();
  const card = document.createElement("div");
  card.className = "utterance latest";
  const original = document.createElement("p");
  const translation = document.createElement("p");
  original.textContent = sourceLine(item);
  translation.textContent = translationLine(item);
  card.append(original, translation);
  ui.transcriptList.append(card);
  state.utterances += 1;
  ui.utteranceCount.textContent = `${state.utterances} 条稳定记录`;
  ui.languageBadge.textContent = languageNames[item.language] || "识别完成";
  while (ui.transcriptList.querySelectorAll(".utterance:not(.partial)").length > 500) {
    ui.transcriptList.querySelector(".utterance:not(.partial)")?.remove();
  }
  if (state.following) ui.transcriptList.scrollTop = ui.transcriptList.scrollHeight;
}

function showPartial(event) {
  ui.transcriptEmpty.hidden = true;
  let card = document.querySelector("#partialUtterance");
  if (!card) {
    card = document.createElement("div");
    card.id = "partialUtterance";
    card.className = "utterance partial";
    card.append(document.createElement("p"));
    ui.transcriptList.append(card);
  }
  const language = event.language ? ` · ${languageNames[event.language] || event.language}` : "";
  card.querySelector("p").textContent = `${formatTimestamp(event.start)}${language}  ${event.text}`;
  if (state.following) ui.transcriptList.scrollTop = ui.transcriptList.scrollHeight;
}

function renderDownloads(files = []) {
  ui.downloadLinks.replaceChildren();
  const preferred = ["meeting_minutes.md", "meeting_transcript.md", "translated_zh.md", "transcript.json", "audio_manifest.json"];
  preferred.filter((name) => files.includes(name)).forEach((name) => {
    const link = document.createElement("a");
    link.href = `/api/meetings/${state.sessionId}/files/${encodeURIComponent(name)}`;
    link.textContent = name;
    link.download = name;
    ui.downloadLinks.append(link);
  });
  ui.downloads.hidden = ui.downloadLinks.childElementCount === 0;
}

function applySnapshot(meeting) {
  state.sessionId = meeting.id;
  state.meetingState = meeting.state;
  state.utterances = 0;
  ui.transcriptList.replaceChildren(ui.transcriptEmpty);
  for (const item of meeting.recent_utterances || []) appendUtterance(item);
  if (meeting.summary) ui.summaryText.textContent = meeting.summary;
  if (meeting.current_language) ui.languageBadge.textContent = languageNames[meeting.current_language] || meeting.current_language;
  if (meeting.state === "recording") startTimer(meeting.elapsed_seconds || 0);
  else ui.timer.textContent = formatClock(meeting.elapsed_seconds || 0);
  if (["finalizing", "summarizing"].includes(meeting.state)) {
    state.stopping = true;
    setStage(meeting.state === "summarizing" ? "summarizing" : "finalizing", "正在恢复会议处理状态");
  } else if (meeting.state === "complete") {
    setStage("complete", "会议已完成");
    renderDownloads(meeting.files);
  } else if (meeting.state === "summary_error") {
    showNotice(meeting.error || "会议纪要生成失败，原稿已经保存");
    ui.retrySummary.hidden = false;
  }
  updateMainButton();
}

async function restoreMeeting(meeting) {
  applySnapshot(meeting);
  if (!["complete", "error"].includes(meeting.state)) {
    try { await connectWebSocket(meeting.id); } catch (error) { showNotice(`无法恢复实时状态：${error.message}`); }
  }
}

function handleServerEvent(event) {
  if (event.type === "snapshot") return applySnapshot(event.meeting);
  if (event.type === "status") {
    setStage(event.stage, event.message);
    if (event.state) state.meetingState = event.state;
    if (event.stage.startsWith("summarizing")) {
      state.meetingState = "summarizing";
      ui.summaryText.classList.add("streaming");
    }
    updateMainButton();
  } else if (event.type === "partial") {
    showPartial(event);
  } else if (event.type === "partial_clear") {
    document.querySelector("#partialUtterance")?.remove();
  } else if (event.type === "utterance") {
    appendUtterance(event.utterance);
  } else if (event.type === "audio_input") {
    state.backendAudioPackets = event.packets_received || 0;
    if (Number.isFinite(event.level) && !state.lastAudioLevelAt) setMicLevel(event.level);
    updateAudioDebug();
    if (event.vad_active) ui.micStatus.textContent = "检测到语音，正在转写";
  } else if (event.type === "summary_delta") {
    if (!ui.summaryText.classList.contains("streaming")) {
      ui.summaryText.textContent = "";
      ui.summaryText.classList.add("streaming");
    }
    ui.summaryText.textContent += event.content;
    ui.summaryText.scrollTop = ui.summaryText.scrollHeight;
  } else if (event.type === "summary_reset") {
    ui.summaryText.textContent = "";
  } else if (event.type === "summary_complete") {
    state.meetingState = "complete";
    state.stopping = false;
    stopTimer();
    ui.summaryText.textContent = event.content;
    ui.summaryText.classList.remove("streaming");
    renderDownloads(event.files);
    setStage("complete", "会议逐句稿和纪要已保存");
    updateMainButton();
  } else if (event.type === "warning") {
    showNotice(event.message, "warning");
  } else if (event.type === "error") {
    showNotice(event.message);
    if (event.code === "summary_failed") {
      state.meetingState = "summary_error";
      state.stopping = false;
      ui.retrySummary.hidden = !event.retryable;
      ui.summaryText.classList.remove("streaming");
    }
    updateMainButton();
  }
}

async function retrySummary() {
  if (!state.sessionId) return;
  clearNotice();
  ui.retrySummary.hidden = true;
  state.meetingState = "summarizing";
  state.stopping = true;
  ui.summaryText.textContent = "";
  ui.summaryText.classList.add("streaming");
  setStage("summarizing", "正在重试生成会议纪要");
  updateMainButton();
  try {
    await requestJson(`/api/meetings/${state.sessionId}/retry-summary`, { method: "POST", body: "{}" });
  } catch (error) {
    state.stopping = false;
    showNotice(error.message);
    ui.retrySummary.hidden = false;
    updateMainButton();
  }
}

ui.mainButton.addEventListener("click", () => {
  if (state.meetingState === "recording") stopMeeting();
  else startMeeting();
});
ui.deviceSelect.addEventListener("change", onDeviceChange);
ui.retrySummary.addEventListener("click", retrySummary);
ui.transcriptList.addEventListener("scroll", () => {
  const distance = ui.transcriptList.scrollHeight - ui.transcriptList.scrollTop - ui.transcriptList.clientHeight;
  state.following = distance < 80;
  ui.jumpLatest.hidden = state.following;
});
ui.jumpLatest.addEventListener("click", () => {
  state.following = true;
  ui.transcriptList.scrollTop = ui.transcriptList.scrollHeight;
  ui.jumpLatest.hidden = true;
});

setStage("checking", "正在检查本机服务…");
pollHealth();
window.setInterval(pollHealth, 2500);
