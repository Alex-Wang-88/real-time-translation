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
  inputWarning: document.querySelector("#inputWarning"),
  inputDeviceSelect: document.querySelector("#inputDeviceSelect"),
  refreshInputDevices: document.querySelector("#refreshInputDevices"),
  inputDeviceHint: document.querySelector("#inputDeviceHint"),
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
  audioSequence: 0,
  audioConfigReady: false,
  backendAudioPackets: 0,
  lastAudioLevel: 0,
  lastAudioLevelAt: 0,
  lastNonZeroLevelAt: 0,
  audioWarningShown: false,
  apiToken: new URLSearchParams(window.location.search).get("token") || "",
  streamTicket: "",
  utteranceNodes: new Map(),
  latestUtteranceNode: null,
  transcriptScrollFrame: 0,
  scrollStateFrame: 0,
  inputDeviceId: (() => { try { return localStorage.getItem("meeting_input_device") || ""; } catch { return ""; } })(),
  deviceRequest: (() => { try { return localStorage.getItem("meeting_device") || "auto"; } catch { return "auto"; } })(),
  switchingDevice: false,
};

const languageNames = {
  zh: "中文", en: "英文", de: "德文",
};
const supportedLanguages = new Set(["zh", "en", "de"]);
const MAX_RENDERED_UTTERANCES = 240;
const SCROLL_BOTTOM_THRESHOLD = 80;
const stageIndex = {
  checking: 0, loading: 0, microphone: 1, recording: 2, listening: 2,
  transcribing: 3, translating: 4, finalizing: 5, saving: 5,
  summarizing: 6, summarizing_chunks: 6, summarizing_final: 6,
  summary_pending: 6, summary_error: 6, complete: 7,
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
  const db = normalized > 0 ? 20 * Math.log10(normalized) : -60;
  const percent = Math.min(100, Math.max(0, Math.round((Math.max(-60, db) + 60) * 100 / 60)));
  ui.levelBar.style.width = `${Math.max(2, percent)}%`;
  ui.levelValue.textContent = `${percent}%`;
  ui.levelTrack?.setAttribute("aria-valuenow", String(percent));
  state.lastAudioLevel = normalized;
  state.lastAudioLevelAt = performance.now();
  if (normalized >= 0.003) {
    state.lastNonZeroLevelAt = state.lastAudioLevelAt;
    clearInputWarning();
  }
  if (percent >= 4) ui.micStatus.textContent = "正在接收声音";
  else if (state.meetingState === "recording") ui.micStatus.textContent = "已连接，等待发言";
}

function setInputWarning(message = "") {
  // Keep the diagnostic node in the DOM for compatibility, but surface a
  // warning through the fixed notice slot so showing/hiding it cannot move
  // the page vertically.
  ui.inputWarning.textContent = "";
  ui.inputWarning.hidden = true;
  if (message) showNotice(message, "warning");
  else if (ui.notice?.classList.contains("warning")) clearNotice();
}

function clearInputWarning() {
  setInputWarning("");
}

function resetMicFeedback(message = "等待麦克风") {
  if (state.audioWatchdog) window.clearInterval(state.audioWatchdog);
  state.audioWatchdog = null;
  state.captureStartedAt = 0;
  state.audioFramesProduced = 0;
  state.audioFramesSent = 0;
  state.audioBytesSent = 0;
  state.audioSequence = 0;
  state.audioConfigReady = false;
  state.backendAudioPackets = 0;
  state.lastAudioLevel = 0;
  state.lastAudioLevelAt = 0;
  state.lastNonZeroLevelAt = 0;
  state.audioWarningShown = false;
  clearInputWarning();
  ui.levelBar.style.width = "2%";
  ui.levelValue.textContent = "0%";
  ui.levelTrack?.setAttribute("aria-valuenow", "0");
  ui.micStatus.textContent = message;
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
  updateDeviceControls();
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(state.apiToken ? { Authorization: `Bearer ${state.apiToken}` } : {}),
      ...(options.headers || {}),
    },
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
    if (health.language_labels) {
      for (const code of supportedLanguages) {
        if (health.language_labels[code]) languageNames[code] = health.language_labels[code];
      }
    }
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

async function refreshInputDevices() {
  if (!navigator.mediaDevices?.enumerateDevices) {
    ui.inputDeviceHint.textContent = "当前浏览器不支持设备枚举，将使用系统默认输入设备。";
    return;
  }
  try {
    const devices = (await navigator.mediaDevices.enumerateDevices())
      .filter((device) => device.kind === "audioinput" && device.deviceId && device.deviceId !== "default");
    const options = [new Option("系统默认（浏览器当前输入设备）", "")];
    const seen = new Set();
    devices.forEach((device, index) => {
      if (seen.has(device.deviceId)) return;
      seen.add(device.deviceId);
      options.push(new Option(device.label || `麦克风 ${index + 1}`, device.deviceId));
    });
    ui.inputDeviceSelect.replaceChildren(...options);
    const selected = options.some((option) => option.value === state.inputDeviceId);
    if (!selected) state.inputDeviceId = "";
    ui.inputDeviceSelect.value = state.inputDeviceId;
    try {
      if (state.inputDeviceId) localStorage.setItem("meeting_input_device", state.inputDeviceId);
      else localStorage.removeItem("meeting_input_device");
    } catch { /* ignore unavailable storage */ }
    if (devices.length) {
      ui.inputDeviceHint.textContent = `${devices.length} 个输入设备可用；选择后点击开始会议。`;
    } else {
      ui.inputDeviceHint.textContent = "尚未获得麦克风设备名称，点击开始会议后会请求权限。";
    }
  } catch (error) {
    ui.inputDeviceHint.textContent = `无法读取输入设备：${error.message || "浏览器拒绝访问"}`;
  }
}

function onInputDeviceChange() {
  state.inputDeviceId = ui.inputDeviceSelect.value || "";
  try {
    if (state.inputDeviceId) localStorage.setItem("meeting_input_device", state.inputDeviceId);
    else localStorage.removeItem("meeting_input_device");
  } catch { /* ignore unavailable storage */ }
  ui.inputDeviceHint.textContent = state.inputDeviceId
    ? "已选择输入设备，点击开始会议后生效。"
    : "将使用浏览器当前默认输入设备。";
}

function updateDeviceControls() {
  const health = state.health || {};
  const recording = state.meetingState === "recording";
  const busy = ["finalizing", "summarizing"].includes(state.meetingState) || state.stopping;
  ui.inputDeviceSelect.disabled = recording || busy;
  ui.refreshInputDevices.disabled = recording || busy;
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
  const ticket = await requestJson(`/api/meetings/${sessionId}/stream-ticket`, {
    method: "POST",
    body: "{}",
  });
  state.streamTicket = ticket.ticket || "";
  const ws = new WebSocket(`${scheme}://${location.host}/api/meetings/${sessionId}/stream`);
  ws.binaryType = "arraybuffer";
  await new Promise((resolve, reject) => {
    const timer = window.setTimeout(() => reject(new Error("实时连接超时")), 6000);
    ws.addEventListener("open", () => {
      window.clearTimeout(timer);
      try {
        ws.send(JSON.stringify({ type: "auth", ticket: state.streamTicket }));
      } catch (error) {
        reject(error);
        return;
      }
      resolve();
    }, { once: true });
    ws.addEventListener("error", () => { window.clearTimeout(timer); reject(new Error("无法建立实时连接")); }, { once: true });
  });
  state.ws = ws;
  ws.addEventListener("message", (event) => {
    try { handleServerEvent(JSON.parse(event.data)); } catch { /* Ignore malformed status frames. */ }
  });
  ws.addEventListener("close", () => {
    if (state.ws === ws) state.ws = null;
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
  let context;
  try {
    context = new AudioContext({ sampleRate: 16000, latencyHint: "interactive" });
  } catch {
    context = new AudioContext({ latencyHint: "interactive" });
  }
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
  if (state.ws?.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify({
      type: "audio_config",
      sample_rate: 16000,
      channels: 1,
      encoding: "pcm_s16le",
      packet_ms: 40,
      sequence_header: true,
    }));
    state.audioConfigReady = true;
  }
  resetMicFeedback("麦克风已连接，等待发言");
  state.captureStartedAt = performance.now();
  worklet.port.onmessage = (event) => {
    if (event.data?.type === "audio") {
      state.audioFramesProduced += 1;
      const buffer = event.data.buffer;
      if (state.ws?.readyState === WebSocket.OPEN) {
        try {
          const sourceBytes = new Uint8Array(buffer);
          const packet = new Uint8Array(sourceBytes.byteLength + 4);
          new DataView(packet.buffer).setUint32(0, state.audioSequence, true);
          packet.set(sourceBytes, 4);
          state.audioSequence += 1;
          state.ws.send(packet.buffer);
          state.audioFramesSent += 1;
          state.audioBytesSent += sourceBytes.byteLength || 0;
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
  if (settings.deviceId && settings.deviceId !== "default") {
    state.inputDeviceId = settings.deviceId;
    try { localStorage.setItem("meeting_input_device", state.inputDeviceId); } catch { /* ignore unavailable storage */ }
    refreshInputDevices();
  }
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
        const message = "麦克风已授权，但没有产生音频帧。请检查输入设备、浏览器麦克风权限，并重新开始。";
        setInputWarning(message);
        showNotice(message, "warning");
      }
    } else if (silentFor > 1800 && state.lastAudioLevel < 0.004) {
      ui.micStatus.textContent = "已连接，当前音量很低";
      ui.audioDebug.textContent = `已发送 ${state.audioFramesSent} 帧，但输入电平接近 0${state.backendAudioPackets ? " · 后端已收到" : ""}；请检查系统麦克风音量。`;
      if (state.backendAudioPackets > 20) {
        setInputWarning("未检测到有效声音，请检查输入设备、麦克风权限和系统输入音量。");
      }
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

function microphoneErrorMessage(error) {
  if (error?.name === "NotAllowedError" || error?.name === "SecurityError") {
    return "浏览器没有麦克风权限，请在地址栏或系统隐私设置中允许访问。";
  }
  if (error?.name === "NotFoundError") return "没有找到可用的麦克风输入设备。";
  if (error?.name === "NotReadableError") return "麦克风已被其他程序占用，或当前输入设备无法读取。";
  if (error?.name === "OverconstrainedError") return "所选输入设备当前不可用，请刷新设备后重试。";
  return error?.message || "无法访问麦克风。";
}

async function startMeeting() {
  clearNotice();
  setStage("microphone", "正在请求麦克风权限");
  let requestedStream = null;
  let createdMeetingId = null;
  try {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error("当前页面无法访问麦克风，请使用 localhost 或 HTTPS 打开 Web 页面。");
    }
    const audioConstraints = {
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    };
    if (state.inputDeviceId) audioConstraints.deviceId = { exact: state.inputDeviceId };
    requestedStream = await navigator.mediaDevices.getUserMedia({
      audio: audioConstraints,
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
    showNotice(`无法开始会议：${microphoneErrorMessage(error)}`);
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
  cancelTranscriptFrames();
  state.utteranceNodes.clear();
  state.latestUtteranceNode = null;
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
  const language = item.language === "yue" ? "zh" : item.language;
  return `[${formatTimestamp(item.start)} - ${formatTimestamp(item.end)}] 演讲人${item.speaker_id}（${languageNames[language] || language}）：“${item.text}”`;
}

function translationLine(item) {
  const status = item.translation_status || (item.translation_zh ? "ready" : "pending");
  let text = item.translation_zh ?? item.translation_en ?? "";
  if (status === "pending") text = text || "翻译中…";
  if (status === "unsupported") text = text || "未配置该语种的本地翻译模型";
  if (status === "failed") text = text || "翻译失败，保留原文";
  return `[${formatTimestamp(item.start)} - ${formatTimestamp(item.end)}] 演讲人${item.speaker_id}（中文翻译）：“${text}”`;
}

function itemKey(item) {
  return item.segment_id || `${state.sessionId || "meeting"}:${item.segment_revision || 0}:${item.id || 0}`;
}

function cancelTranscriptFrames() {
  if (state.transcriptScrollFrame) cancelAnimationFrame(state.transcriptScrollFrame);
  if (state.scrollStateFrame) cancelAnimationFrame(state.scrollStateFrame);
  state.transcriptScrollFrame = 0;
  state.scrollStateFrame = 0;
}

function scheduleTranscriptScroll() {
  if (!state.following || state.transcriptScrollFrame) return;
  state.transcriptScrollFrame = requestAnimationFrame(() => {
    state.transcriptScrollFrame = 0;
    if (state.following) ui.transcriptList.scrollTop = ui.transcriptList.scrollHeight;
  });
}

function removeTranscriptNode(node) {
  if (!node) return;
  if (state.latestUtteranceNode === node) state.latestUtteranceNode = null;
  const key = node.dataset.segmentId;
  if (key && state.utteranceNodes.get(key) === node) state.utteranceNodes.delete(key);
  node.remove();
}

function pruneTranscript() {
  while (state.utteranceNodes.size > MAX_RENDERED_UTTERANCES) {
    const oldest = state.utteranceNodes.values().next().value;
    if (!oldest) break;
    removeTranscriptNode(oldest);
  }
}

function markLatestUtterance(card) {
  if (state.latestUtteranceNode && state.latestUtteranceNode !== card) {
    state.latestUtteranceNode.classList.remove("latest");
  }
  state.latestUtteranceNode = card;
  card.classList.add("latest");
}

function appendUtterance(item) {
  const language = item.language === "yue" ? "zh" : item.language;
  if (!supportedLanguages.has(language)) return;
  item = { ...item, language };
  ui.transcriptEmpty.hidden = true;
  ui.transcriptList.querySelector("#partialUtterance")?.remove();
  if (Number.isFinite(item.segment_revision)) {
    ui.transcriptList.querySelector(`[data-draft-revision="${item.segment_revision}"]`)?.remove();
  }
  const key = itemKey(item);
  let card = state.utteranceNodes.get(key);
  const isNew = !card;
  if (!card) {
    card = document.createElement("div");
    card.className = "utterance";
    card.dataset.segmentId = key;
    card.append(document.createElement("p"), document.createElement("p"));
    ui.transcriptList.append(card);
    state.utteranceNodes.set(key, card);
  }
  markLatestUtterance(card);
  card.classList.toggle("draft", item.recognition_stage === "fast" || item.stage === "draft");
  card.dataset.start = String(item.start ?? 0);
  card.dataset.end = String(item.end ?? 0);
  card.dataset.speakerId = String(item.speaker_id ?? "?");
  card.dataset.language = String(item.language ?? "");
  card.firstElementChild.textContent = sourceLine(item);
  card.lastElementChild.textContent = translationLine(item);
  if (isNew) state.utterances += 1;
  ui.utteranceCount.textContent = `${state.utterances} 条稳定记录`;
  ui.languageBadge.textContent = languageNames[item.language] || "识别完成";
  pruneTranscript();
  scheduleTranscriptScroll();
}

function showPartial(event) {
  const language = event.language === "yue" ? "zh" : event.language;
  if (language && !supportedLanguages.has(language)) {
    ui.transcriptList.querySelector("#partialUtterance")?.remove();
    return;
  }
  ui.transcriptEmpty.hidden = true;
  let card = ui.transcriptList.querySelector("#partialUtterance");
  if (!card) {
    card = document.createElement("div");
    card.id = "partialUtterance";
    card.className = "utterance partial";
    card.append(document.createElement("p"));
    ui.transcriptList.append(card);
  }
  const label = language ? ` · ${languageNames[language] || language}` : "";
  const text = `${formatTimestamp(event.start)}${label}  ${event.text}`;
  if (card.firstElementChild.textContent !== text) card.firstElementChild.textContent = text;
  scheduleTranscriptScroll();
}

function renderDownloads(files = []) {
  ui.downloadLinks.replaceChildren();
  const preferred = ["meeting_minutes.md", "meeting_transcript.md", "translated_zh.md", "transcript.json", "audio_manifest.json"];
  const token = state.apiToken ? `?token=${encodeURIComponent(state.apiToken)}` : "";
  preferred.filter((name) => files.includes(name)).forEach((name) => {
    const link = document.createElement("a");
    link.href = `/api/meetings/${state.sessionId}/files/${encodeURIComponent(name)}${token}`;
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
  cancelTranscriptFrames();
  state.utteranceNodes.clear();
  state.latestUtteranceNode = null;
  ui.transcriptList.replaceChildren(ui.transcriptEmpty);
  for (const item of meeting.recent_utterances || []) appendUtterance(item);
  if (meeting.summary) ui.summaryText.textContent = meeting.summary;
  if (meeting.current_language) {
    const language = meeting.current_language === "yue" ? "zh" : meeting.current_language;
    ui.languageBadge.textContent = languageNames[language] || "等待发言";
  }
  if (meeting.state === "recording") startTimer(meeting.elapsed_seconds || 0);
  else ui.timer.textContent = formatClock(meeting.elapsed_seconds || 0);
  if (["finalizing", "summarizing"].includes(meeting.state)) {
    state.stopping = true;
    setStage(meeting.state === "summarizing" ? "summarizing" : "finalizing", "正在恢复会议处理状态");
  } else if (meeting.state === "summary_pending") {
    state.stopping = false;
    ui.retrySummary.hidden = false;
    ui.retrySummary.textContent = "生成会议纪要";
    ui.summaryText.textContent = "会议已保存。点击“生成会议纪要”后，才会请求 AI。";
    renderDownloads(meeting.files);
    setStage("summary_pending", "完整逐句稿已保存，可以手动生成会议纪要");
  } else if (meeting.state === "complete") {
    ui.retrySummary.hidden = true;
    setStage("complete", "会议已完成");
    renderDownloads(meeting.files);
  } else if (meeting.state === "summary_error") {
    showNotice(meeting.error || "会议纪要生成失败，原稿已经保存");
    ui.retrySummary.hidden = false;
    ui.retrySummary.textContent = "重试生成";
    renderDownloads(meeting.files);
    setStage("summary_error", "会议纪要生成失败，可以重试");
  } else if (meeting.state === "error") {
    showNotice(meeting.error || "会议处理未完整结束，已有记录仍可下载");
    renderDownloads(meeting.files);
    setStage("finalizing", "会议处理未完整结束，已有记录仍可下载");
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
  if (event.type === "auth_ok" || event.type === "audio_config_ack") return;
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
  } else if (event.type === "draft") {
    showDraft(event);
  } else if (event.type === "partial_clear") {
    ui.transcriptList.querySelector("#partialUtterance")?.remove();
  } else if (event.type === "utterance") {
    appendUtterance(event.utterance);
  } else if (event.type === "utterance_update") {
    appendUtterance(event.utterance);
  } else if (event.type === "translation_update") {
    updateTranslation(event);
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
  } else if (event.type === "summary_pending") {
    state.sessionId = event.session_id || state.sessionId;
    state.meetingState = "summary_pending";
    state.stopping = false;
    ui.retrySummary.hidden = false;
    ui.retrySummary.textContent = "生成会议纪要";
    ui.summaryText.classList.remove("streaming");
    ui.summaryText.textContent = "会议已保存。点击“生成会议纪要”后，才会请求 AI。";
    if (event.error) showNotice(event.error, "warning");
    renderDownloads(event.files || []);
    setStage("summary_pending", "完整逐句稿已保存，可以手动生成会议纪要");
    updateMainButton();
  } else if (event.type === "summary_complete") {
    state.meetingState = "complete";
    state.stopping = false;
    stopTimer();
    ui.retrySummary.hidden = true;
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
      ui.retrySummary.textContent = "重试生成";
      ui.summaryText.classList.remove("streaming");
    }
    updateMainButton();
  }
}

async function retrySummary() {
  if (!state.sessionId) return;
  const retryState = state.meetingState === "summary_error" ? "summary_error" : "summary_pending";
  clearNotice();
  ui.retrySummary.hidden = true;
  state.meetingState = "summarizing";
  state.stopping = true;
  ui.summaryText.textContent = "";
  ui.summaryText.classList.add("streaming");
  setStage("summarizing", "正在重试生成会议纪要");
  updateMainButton();
  try {
    if (!state.ws || state.ws.readyState !== WebSocket.OPEN) {
      await connectWebSocket(state.sessionId);
    }
    await requestJson(`/api/meetings/${state.sessionId}/retry-summary`, { method: "POST", body: "{}" });
  } catch (error) {
    state.meetingState = retryState;
    state.stopping = false;
    showNotice(error.message);
    ui.retrySummary.hidden = false;
    ui.retrySummary.textContent = retryState === "summary_error" ? "重试生成" : "生成会议纪要";
    setStage(retryState, retryState === "summary_error" ? "会议纪要生成失败，可以重试" : "会议已保存，可以手动生成会议纪要");
    updateMainButton();
  }
}

function showDraft(event) {
  const language = event.language === "yue" ? "zh" : event.language;
  if (language && !supportedLanguages.has(language)) return;
  ui.transcriptEmpty.hidden = true;
  ui.transcriptList.querySelector("#partialUtterance")?.remove();
  let card = ui.transcriptList.querySelector(`[data-draft-revision="${event.revision}"]`);
  if (!card) {
    card = document.createElement("div");
    card.className = "utterance partial draft";
    card.dataset.draftRevision = String(event.revision);
    card.append(document.createElement("p"));
    ui.transcriptList.append(card);
  }
  const label = language ? ` · ${languageNames[language] || language}` : "";
  const text = `${formatTimestamp(event.start)}${label}  ${event.text}（精修中）`;
  if (card.firstElementChild.textContent !== text) card.firstElementChild.textContent = text;
  scheduleTranscriptScroll();
}

function updateTranslation(event) {
  const card = state.utteranceNodes.get(event.segment_id);
  if (!card) return;
  const item = {
    start: Number(card.dataset.start || 0),
    end: Number(card.dataset.end || 0),
    speaker_id: card.dataset.speakerId || "?",
    language: card.dataset.language || "",
    translation_zh: event.translation_zh || "",
    translation_status: event.translation_status || "ready",
  };
  card.lastElementChild.textContent = translationLine(item);
}

ui.mainButton.addEventListener("click", () => {
  if (state.meetingState === "recording") stopMeeting();
  else startMeeting();
});
ui.inputDeviceSelect.addEventListener("change", onInputDeviceChange);
ui.refreshInputDevices.addEventListener("click", () => refreshInputDevices());
ui.deviceSelect.addEventListener("change", onDeviceChange);
ui.retrySummary.addEventListener("click", retrySummary);
ui.transcriptList.addEventListener("scroll", () => {
  if (state.scrollStateFrame) return;
  state.scrollStateFrame = requestAnimationFrame(() => {
    state.scrollStateFrame = 0;
    const distance = ui.transcriptList.scrollHeight - ui.transcriptList.scrollTop - ui.transcriptList.clientHeight;
    const following = distance < SCROLL_BOTTOM_THRESHOLD;
    if (following !== state.following) {
      state.following = following;
      ui.jumpLatest.hidden = following;
    }
  });
}, { passive: true });
ui.jumpLatest.addEventListener("click", () => {
  state.following = true;
  ui.transcriptList.scrollTop = ui.transcriptList.scrollHeight;
  ui.jumpLatest.hidden = true;
});

setStage("checking", "正在检查本机服务…");
refreshInputDevices();
if (navigator.mediaDevices?.addEventListener) {
  navigator.mediaDevices.addEventListener("devicechange", () => refreshInputDevices());
}
pollHealth();
window.setInterval(pollHealth, 2500);
