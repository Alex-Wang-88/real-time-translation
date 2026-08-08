"""Minimal PyQt6 desktop client for the local real-time meeting service."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import queue
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import websockets
from PyQt6.QtCore import QObject, QSettings, QThread, QTimer, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QCloseEvent, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .text_normalize import simplify_chinese
from .models import language_label


BASE_URL = "http://127.0.0.1:8765"
LANGUAGE_NAMES = {
    "zh": "中文", "en": "英文", "de": "德文", "ru": "俄文",
    "es": "西班牙文", "pt": "葡萄牙文", "fr": "法文", "it": "意大利文",
    "ja": "日文", "ko": "韩文", "ar": "阿拉伯文", "uk": "乌克兰文",
    "pl": "波兰文", "nl": "荷兰文", "tr": "土耳其文", "vi": "越南文",
}


def list_input_devices() -> list[tuple[int, str]]:
    import sounddevice as sd

    devices: list[tuple[int, str]] = []
    hostapis = sd.query_hostapis()
    for index, device in enumerate(sd.query_devices()):
        if int(device.get("max_input_channels", 0)) <= 0:
            continue
        name = str(device.get("name", f"麦克风 {index + 1}")).replace("\r", " ").replace("\n", " ").strip()
        lower_name = name.casefold()
        if "sound mapper" in lower_name or "声音映射器" in name or "主声音捕获驱动程序" in name:
            continue
        if (
            ("output" in lower_name or "speaker" in lower_name or "扬声器" in name or "输出" in name)
            and "input" not in lower_name
            and "输入" not in name
        ):
            continue
        host_index = int(device.get("hostapi", 0))
        host_name = str(hostapis[host_index].get("name", "音频接口"))
        sample_rate = int(round(float(device.get("default_samplerate", 16_000))))
        # Keep every host-api endpoint. Windows often exposes the same
        # physical microphone through MME, DirectSound, WASAPI and WDM-KS;
        # one endpoint can be silent while another carries the real signal.
        devices.append((index, f"{name}  ·  {host_name}  ·  {sample_rate} Hz"))
    return devices


class MeetingWorker(QObject):
    status = pyqtSignal(str)
    audio_level = pyqtSignal(float)
    audio_stats = pyqtSignal(int, int, int)
    partial = pyqtSignal(dict)
    partial_clear = pyqtSignal()
    utterance = pyqtSignal(dict)
    summary_delta = pyqtSignal(str)
    summary_reset = pyqtSignal()
    summary_complete = pyqtSignal(str, list)
    summary_ready = pyqtSignal(str, list)
    error = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, base_url: str, device_index: int | None) -> None:
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.device_index = device_index
        self.session_id: str | None = None
        self.stop_event = threading.Event()
        self.audio_queue: queue.Queue[bytes] = queue.Queue(maxsize=300)
        self.audio_stream: Any = None
        self.audio_frames_sent = 0
        self.audio_bytes_sent = 0
        self.backend_packets_received = 0
        self.last_level_emit = 0.0
        self.capture_rate = 16_000
        self.capture_device_name = ""
        self.mute_note = ""
        self._endpoint_volume: Any = None
        self._previous_mute: bool | None = None
        self._com_initialized = False

    @pyqtSlot()
    def request_stop(self) -> None:
        self.stop_event.set()

    def _audio_callback(self, indata: np.ndarray, _frames: int, _time_info: Any, status: Any) -> None:
        if status:
            self.status.emit(f"音频设备提示：{status}")
        if indata is None or not len(indata):
            return
        samples = np.asarray(indata, dtype=np.float32)
        if samples.ndim == 2:
            samples = np.mean(samples, axis=1, dtype=np.float32)
        # Some Windows/PortAudio drivers can occasionally hand the callback
        # non-finite values (NaN/Inf), especially while an endpoint is being
        # opened or resumed.  Do this before resampling as np.interp would
        # otherwise propagate those values into the whole block.
        samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
        if self.capture_rate != 16_000 and len(samples) > 1:
            # Capture at the endpoint's native rate, then send the exact
            # 16 kHz mono format expected by the backend. This covers devices
            # that reject a direct 16 kHz PortAudio stream (common with WASAPI).
            target_count = max(1, round(len(samples) * 16_000 / self.capture_rate))
            source_x = np.linspace(0.0, 1.0, len(samples), endpoint=False)
            target_x = np.linspace(0.0, 1.0, target_count, endpoint=False)
            samples = np.interp(target_x, source_x, samples)
        samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
        samples = np.clip(np.rint(samples), -32768, 32767).astype(np.int16)
        pcm = samples.tobytes()
        try:
            self.audio_queue.put_nowait(pcm)
        except queue.Full:
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.put_nowait(pcm)
            except queue.Empty:
                pass
        now = time.monotonic()
        if now - self.last_level_emit >= 0.1:
            # Cast back to float before squaring so int16 values cannot
            # overflow.  Keep the signal finite even if a future audio
            # backend changes the sample representation again.
            float_samples = samples.astype(np.float32, copy=False)
            rms = float(np.sqrt(np.mean(float_samples * float_samples)) / 32768.0)
            self.audio_level.emit(rms if math.isfinite(rms) else 0.0)
            self.last_level_emit = now

    def _prepare_input_endpoint(self, raw_name: str) -> None:
        """Temporarily clear a Windows endpoint mute set on the input device.

        Windows can grant Python microphone permission while the capture
        endpoint itself remains muted. PortAudio then reports valid packet
        timing but every sample is zero. On meeting start we clear that mute
        and restore it when the meeting ends.
        """
        self.mute_note = ""
        try:
            import comtypes
            from pycaw.pycaw import AudioUtilities

            comtypes.CoInitialize()
            self._com_initialized = True
            target = raw_name.casefold().strip()
            best: tuple[int, Any] | None = None
            for device in AudioUtilities.GetAllDevices():
                try:
                    friendly = str(getattr(device, "FriendlyName", "")).strip()
                    endpoint = getattr(device, "EndpointVolume", None)
                except Exception:
                    continue
                if not friendly or endpoint is None:
                    continue
                candidate = friendly.casefold()
                score = 0
                if candidate == target:
                    score = 3
                elif target in candidate or candidate in target:
                    score = 2
                elif "麦克风" in raw_name and "麦克风" in friendly:
                    score = 1
                if score and (best is None or score > best[0]):
                    best = (score, endpoint)
            if best is None:
                return
            endpoint = best[1]
            previous = bool(endpoint.GetMute())
            self._endpoint_volume = endpoint
            self._previous_mute = previous
            if previous:
                endpoint.SetMute(False, None)
                self.mute_note = "检测到输入端点静音，已临时解除（会议结束后恢复）"
        except Exception:
            # Mute control is a diagnostic enhancement; failure must not stop
            # recording on systems without pycaw/COM endpoint access.
            self._endpoint_volume = None
            self._previous_mute = None

    def _restore_input_mute(self) -> None:
        try:
            if self._endpoint_volume is not None and self._previous_mute is True:
                self._endpoint_volume.SetMute(True, None)
        except Exception:
            pass
        finally:
            self._endpoint_volume = None
            self._previous_mute = None
            if self._com_initialized:
                try:
                    import comtypes

                    comtypes.CoUninitialize()
                except Exception:
                    pass
                self._com_initialized = False

    def _open_audio(self) -> None:
        import sounddevice as sd

        selected_device = self.device_index
        if selected_device is None:
            default_device = int(sd.default.device[0])
            selected_device = default_device if default_device >= 0 else None
        info = sd.query_devices(selected_device, "input")
        hostapis = sd.query_hostapis()
        host_index = int(info.get("hostapi", 0))
        host_name = str(hostapis[host_index].get("name", "音频接口"))
        native_rate = int(round(float(info.get("default_samplerate", 16_000))))
        try:
            sd.check_input_settings(
                device=selected_device,
                channels=1,
                dtype="int16",
                samplerate=16_000,
            )
            self.capture_rate = 16_000
        except Exception:
            self.capture_rate = max(8_000, native_rate)
        self.capture_device_name = (
            str(info.get("name", "麦克风")).replace("\r", " ").replace("\n", " ").strip()
            + f" · {host_name}"
        )
        raw_name = str(info.get("name", "麦克风")).replace("\r", " ").replace("\n", " ").strip()
        self._prepare_input_endpoint(raw_name)
        kwargs: dict[str, Any] = {
            "samplerate": self.capture_rate,
            "blocksize": max(1, round(self.capture_rate * 0.02)),
            "channels": 1,
            "dtype": "int16",
            "callback": self._audio_callback,
        }
        if selected_device is not None:
            kwargs["device"] = selected_device
        self.audio_stream = sd.InputStream(**kwargs)
        self.audio_stream.start()

    def _close_audio(self) -> None:
        try:
            if self.audio_stream is not None:
                self.audio_stream.stop()
                self.audio_stream.close()
        finally:
            self.audio_stream = None
            self._restore_input_mute()

    async def _send_audio(self, socket: Any) -> None:
        while True:
            try:
                pcm = self.audio_queue.get_nowait()
            except queue.Empty:
                if self.stop_event.is_set():
                    return
                # Poll without creating a blocking executor thread for every
                # empty queue interval. This keeps long silent meetings stable.
                await asyncio.sleep(0.02)
                continue
            await socket.send(pcm)
            self.audio_frames_sent += 1
            self.audio_bytes_sent += len(pcm)
            self.audio_stats.emit(
                self.audio_frames_sent,
                self.audio_bytes_sent,
                self.backend_packets_received,
            )

    async def _receive_events(self, socket: Any) -> None:
        async for raw in socket:
            event = json.loads(raw)
            kind = event.get("type")
            if kind == "status":
                self.status.emit(str(event.get("message", "处理中")))
            elif kind == "partial":
                self.partial.emit(event)
            elif kind == "utterance":
                self.utterance.emit(event.get("utterance", {}))
            elif kind == "partial_clear":
                self.partial_clear.emit()
            elif kind == "audio_input":
                self.backend_packets_received = int(event.get("packets_received", 0))
                self.audio_stats.emit(
                    self.audio_frames_sent,
                    self.audio_bytes_sent,
                    self.backend_packets_received,
                )
            elif kind == "summary_delta":
                self.summary_delta.emit(str(event.get("content", "")))
            elif kind == "summary_reset":
                self.summary_reset.emit()
            elif kind == "summary_complete":
                self.summary_complete.emit(str(event.get("content", "")), event.get("files", []))
                return
            elif kind == "summary_pending":
                self.summary_ready.emit(
                    str(event.get("session_id", self.session_id or "")),
                    event.get("files", []),
                )
                return
            elif kind == "error":
                self.error.emit(str(event.get("message", "处理失败")))
                if event.get("code") in {"summary_failed", "disk_critical", "inference_failed"}:
                    return

    async def _run_async(self) -> None:
        timeout = httpx.Timeout(30.0, connect=10.0)
        async with httpx.AsyncClient(base_url=self.base_url, timeout=timeout) as client:
            response = await client.post("/api/meetings", json={})
            response.raise_for_status()
            meeting = response.json()
            self.session_id = meeting["id"]
            ws_url = self.base_url.replace("http://", "ws://").replace("https://", "wss://")
            async with websockets.connect(f"{ws_url}/api/meetings/{self.session_id}/stream") as socket:
                self._open_audio()
                mute_note = f"；{self.mute_note}" if self.mute_note else ""
                self.status.emit(
                    f"已打开：{self.capture_device_name}（{self.capture_rate} Hz → 16 kHz）{mute_note}"
                )
                receiver = asyncio.create_task(self._receive_events(socket))
                sender = asyncio.create_task(self._send_audio(socket))
                try:
                    await asyncio.to_thread(self.stop_event.wait)
                    self.status.emit("正在保存最后一段语音")
                finally:
                    self._close_audio()
                    try:
                        await asyncio.wait_for(sender, 2.0)
                    except (asyncio.TimeoutError, asyncio.CancelledError):
                        sender.cancel()
                    await client.post(f"/api/meetings/{self.session_id}/stop", json={})
                try:
                    await asyncio.wait_for(receiver, 180.0)
                except asyncio.TimeoutError:
                    receiver.cancel()
                    raise RuntimeError("等待会议纪要超时；原稿仍已保存")

    def run(self) -> None:
        try:
            asyncio.run(self._run_async())
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            self._close_audio()
            self.finished.emit()


class SummaryWorker(QObject):
    """Request and stream a summary only after the user clicks the button."""

    status = pyqtSignal(str)
    summary_delta = pyqtSignal(str)
    summary_reset = pyqtSignal()
    summary_complete = pyqtSignal(str, list)
    error = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, base_url: str, session_id: str) -> None:
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.session_id = session_id

    async def _run_async(self) -> None:
        ws_url = self.base_url.replace("http://", "ws://").replace("https://", "wss://")
        timeout = httpx.Timeout(30.0, connect=10.0)
        async with httpx.AsyncClient(base_url=self.base_url, timeout=timeout) as client:
            # Connect before scheduling the server task so no streamed delta is
            # lost between the POST and the websocket subscription.
            async with websockets.connect(
                f"{ws_url}/api/meetings/{self.session_id}/stream"
            ) as socket:
                response = await client.post(
                    f"/api/meetings/{self.session_id}/retry-summary", json={}
                )
                response.raise_for_status()
                async for raw in socket:
                    event = json.loads(raw)
                    kind = event.get("type")
                    if kind == "status":
                        self.status.emit(str(event.get("message", "正在生成会议纪要")))
                    elif kind == "summary_delta":
                        self.summary_delta.emit(str(event.get("content", "")))
                    elif kind == "summary_reset":
                        self.summary_reset.emit()
                    elif kind == "summary_complete":
                        self.summary_complete.emit(
                            str(event.get("content", "")), event.get("files", [])
                        )
                        return
                    elif kind == "error":
                        self.error.emit(str(event.get("message", "会议纪要生成失败")))
                        return

    def run(self) -> None:
        try:
            asyncio.run(self._run_async())
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            self.finished.emit()


class DesktopWindow(QMainWindow):
    def __init__(self, base_url: str, backend_process: subprocess.Popen[bytes] | None = None) -> None:
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.backend_process = backend_process
        self.worker: MeetingWorker | None = None
        self.worker_thread: QThread | None = None
        self.summary_worker: SummaryWorker | None = None
        self.summary_thread: QThread | None = None
        self.last_meeting_id: str | None = None
        self.health: dict[str, Any] = {}
        self.meeting_started_at = 0.0
        # Inference-device selector state.
        self.device_initialized = False  # sync combo with backend device once
        self.switching_device = False  # a device switch is in flight
        self.settings = QSettings("real-time-translation", "desktop")
        self.device_select_combo.currentIndexChanged.connect(self._on_device_selected)
        self.last_audio_level = 0.0
        self.last_backend_packets = 0
        self.setWindowTitle("本机实时会议转译")
        self.resize(1120, 800)
        self.setMinimumSize(900, 650)
        # Keep the entire window on one Chinese-capable UI font even when the
        # window is created by an embedding client instead of ``main()``.
        self.setFont(QFont("Microsoft YaHei", 10))
        self._build_ui()
        self._load_devices()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._poll_health)
        self.timer.start(1500)
        self._poll_health()

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(22, 18, 22, 18)
        layout.setSpacing(10)

        header = QFrame()
        header.setObjectName("header")
        header.setMaximumHeight(150)
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(22, 16, 22, 16)
        header_layout.setSpacing(4)
        header_top = QHBoxLayout()
        header_top.setContentsMargins(0, 0, 0, 0)
        eyebrow = QLabel("LOCAL  /  LIVE TRANSLATION")
        eyebrow.setObjectName("eyebrow")
        header_top.addWidget(eyebrow)
        header_top.addStretch(1)
        header_badge = QLabel("本机模式")
        header_badge.setObjectName("headerBadge")
        header_top.addWidget(header_badge)
        header_layout.addLayout(header_top)
        title = QLabel("实时会议转译")
        title.setObjectName("title")
        subtitle = QLabel("本机模式  ·  多语言原文  →  简体中文")
        subtitle.setObjectName("subtitle")
        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        layout.addWidget(header)

        status_card = QFrame()
        status_card.setObjectName("card")
        status_card.setMaximumHeight(64)
        status_layout = QHBoxLayout(status_card)
        status_layout.setContentsMargins(16, 8, 16, 8)
        status_layout.setSpacing(10)
        status_title = QLabel("系统状态")
        status_title.setObjectName("cardTitle")
        status_layout.addWidget(status_title)
        status_layout.addSpacing(8)
        self.backend_label = QLabel("后端：检查中")
        self.gpu_label = QLabel("GPU：检查中")
        self.jimo_label = QLabel("积墨：检查中")
        for label in (self.backend_label, self.gpu_label, self.jimo_label):
            label.setObjectName("statusChip")
            status_layout.addWidget(label)
        status_layout.addStretch(1)
        self.status_hint = QLabel("等待本机模型就绪")
        self.status_hint.setObjectName("sectionHint")
        status_layout.addWidget(self.status_hint)
        layout.addWidget(status_card)

        control_card = QFrame()
        control_card.setObjectName("card")
        control_card.setMaximumHeight(220)
        self.control_card = control_card
        control_layout = QVBoxLayout(control_card)
        control_layout.setContentsMargins(18, 8, 18, 8)
        control_layout.setSpacing(5)
        control_header = QHBoxLayout()
        control_header.setContentsMargins(0, 0, 0, 0)
        control_title = QLabel("会议控制")
        control_title.setObjectName("cardTitle")
        control_header.addWidget(control_title)
        control_header.addStretch(1)
        control_hint = QLabel("选择输入设备后开始录音")
        control_hint.setObjectName("sectionHint")
        control_header.addWidget(control_hint)
        control_layout.addLayout(control_header)

        device_row = QHBoxLayout()
        device_row.setContentsMargins(0, 0, 0, 0)
        device_row.setSpacing(8)
        device_label = QLabel("输入设备")
        device_label.setObjectName("fieldLabel")
        device_label.setMinimumWidth(72)
        device_row.addWidget(device_label)
        self.device_combo = QComboBox()
        self.device_combo.setMinimumHeight(40)
        self.device_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.device_combo.setMinimumContentsLength(8)
        self.device_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        device_row.addWidget(self.device_combo, 1)
        self.refresh_button = QPushButton("刷新设备")
        self.refresh_button.setObjectName("refreshButton")
        self.refresh_button.setFixedHeight(40)
        self.refresh_button.setMinimumWidth(84)
        self.refresh_button.clicked.connect(self._load_devices)
        device_row.addWidget(self.refresh_button)
        control_layout.addLayout(device_row)

        # Inference device selector: lets users without a CUDA GPU fall back to
        # CPU, or force GPU, without editing .env or restarting the backend.
        device_select_row = QHBoxLayout()
        device_select_row.setContentsMargins(0, 0, 0, 0)
        device_select_row.setSpacing(8)
        device_select_label = QLabel("推理设备")
        device_select_label.setObjectName("fieldLabel")
        device_select_label.setMinimumWidth(72)
        device_select_row.addWidget(device_select_label)
        self.device_select_combo = QComboBox()
        self.device_select_combo.setMinimumHeight(40)
        self.device_select_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.device_select_combo.setMinimumContentsLength(10)
        self.device_select_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        # Data role carries the value passed to the backend (auto/cpu/cuda).
        self.device_select_combo.addItem("自动（推荐）", "auto")
        self.device_select_combo.addItem("CPU（无显卡可用）", "cpu")
        self.device_select_combo.addItem("GPU（需 CUDA）", "cuda")
        device_select_row.addWidget(self.device_select_combo, 1)
        control_layout.addLayout(device_select_row)

        action_row = QHBoxLayout()
        action_row.setContentsMargins(0, 0, 0, 0)
        action_row.setSpacing(8)
        self.start_button = QPushButton("开始会议")
        self.start_button.setObjectName("primaryButton")
        self.start_button.setFixedHeight(40)
        self.start_button.setMinimumWidth(108)
        self.start_button.clicked.connect(self._toggle_meeting)
        action_row.addWidget(self.start_button)
        level_label = QLabel("麦克风电平")
        level_label.setObjectName("metricLabel")
        action_row.addWidget(level_label)
        self.volume_bar = QProgressBar()
        self.volume_bar.setRange(0, 100)
        self.volume_bar.setValue(0)
        self.volume_bar.setFormat("%p%")
        self.volume_bar.setMinimumHeight(24)
        self.volume_bar.setMinimumWidth(60)
        action_row.addWidget(self.volume_bar, 1)
        self.audio_stats_label = QLabel("尚未采集音频")
        self.audio_stats_label.setObjectName("mutedLabel")
        action_row.addWidget(self.audio_stats_label)
        control_layout.addLayout(action_row)

        info_row = QHBoxLayout()
        info_row.setContentsMargins(0, 0, 0, 0)
        info_row.setSpacing(8)
        self.language_label = QLabel("当前语言：—")
        self.language_label.setObjectName("infoPill")
        info_row.addWidget(self.language_label)
        self.status_label = QLabel("正在检查本机服务…")
        self.status_label.setObjectName("infoLabel")
        self.status_label.setWordWrap(True)
        info_row.addWidget(self.status_label, 1)
        control_layout.addLayout(info_row)
        self.input_warning = QLabel()
        self.input_warning.setObjectName("warning")
        self.input_warning.setWordWrap(True)
        # Keep a permanent, fixed-height status slot.  Toggling QLabel
        # visibility changes the parent layout's size hint and makes the
        # transcript area jump whenever audio crosses the warning threshold.
        self.input_warning.setFixedHeight(56)
        self.input_warning.setProperty("active", False)
        self.input_warning.setToolTip(
            "请在下拉框选择实际麦克风，并检查 Windows 麦克风权限、静音键和输入音量。"
            "如果设备支持，可优先尝试带 WASAPI 的输入设备。"
        )
        self.input_warning.setVisible(True)
        control_layout.addWidget(self.input_warning)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(10)
        transcript_card = QFrame()
        transcript_card.setObjectName("card")
        transcript_layout = QVBoxLayout(transcript_card)
        transcript_layout.setContentsMargins(18, 16, 18, 18)
        transcript_layout.setSpacing(10)
        transcript_header = QHBoxLayout()
        transcript_header.setContentsMargins(0, 0, 0, 0)
        transcript_title = QLabel("实时逐句稿")
        transcript_title.setObjectName("sectionTitle")
        transcript_header.addWidget(transcript_title)
        transcript_header.addStretch(1)
        transcript_hint = QLabel("原文 + 中文翻译")
        transcript_hint.setObjectName("sectionHint")
        transcript_header.addWidget(transcript_hint)
        transcript_layout.addLayout(transcript_header)
        self.partial_label = QLabel("等待发言")
        self.partial_label.setObjectName("partial")
        self.partial_label.setWordWrap(True)
        self.transcript = QPlainTextEdit()
        self.transcript.setReadOnly(True)
        self.transcript.setPlaceholderText("开始会议后，识别到的原文和中文翻译会显示在这里")
        transcript_layout.addWidget(self.partial_label)
        transcript_layout.addWidget(self.transcript, 1)

        summary_card = QFrame()
        summary_card.setObjectName("card")
        self.summary_card = summary_card
        summary_layout = QVBoxLayout(summary_card)
        summary_layout.setContentsMargins(18, 16, 18, 18)
        summary_layout.setSpacing(10)
        summary_header = QHBoxLayout()
        summary_header.setContentsMargins(0, 0, 0, 0)
        summary_title = QLabel("中文会议纪要")
        summary_title.setObjectName("sectionTitle")
        summary_header.addWidget(summary_title)
        summary_header.addStretch(1)
        summary_hint = QLabel("停止后手动生成")
        summary_hint.setObjectName("sectionHint")
        summary_header.addWidget(summary_hint)
        self.summary_button = QPushButton("生成会议纪要")
        self.summary_button.setObjectName("summaryButton")
        self.summary_button.setMinimumHeight(34)
        self.summary_button.setEnabled(False)
        self.summary_button.clicked.connect(self._generate_summary)
        summary_header.addWidget(self.summary_button)
        summary_layout.addLayout(summary_header)
        self.summary = QPlainTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setPlaceholderText("停止会议后，点击“生成会议纪要”开始请求 AI")
        summary_layout.addWidget(self.summary, 1)

        # The transcript owns the tall left pane.  The compact controls and
        # summary share one right column, so their widths stay identical while
        # the splitter remains user-resizable.
        right_column = QWidget()
        right_layout = QVBoxLayout(right_column)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)
        control_card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        right_layout.addWidget(control_card)
        right_layout.addWidget(summary_card, 1)

        splitter.addWidget(transcript_card)
        splitter.addWidget(right_column)
        self.splitter = splitter
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([760, 380])
        layout.addWidget(splitter, 1)

        text_font = QFont("Microsoft YaHei", 11)
        self.transcript.setFont(text_font)
        self.summary.setFont(text_font)
        self.setStyleSheet(
            """
            QWidget#root { background: #F4F7FB; color: #17233B; font-size: 10pt; }
            QFrame#header { background: #111B31; border: 1px solid #1F2D49; border-radius: 18px; }
            QLabel#eyebrow { color: #8EA6C9; font-size: 9pt; font-weight: 700; letter-spacing: 1px; }
            QLabel#headerBadge { color: #C7D8F2; background: #1D2A45; border: 1px solid #33476A; border-radius: 10px; padding: 5px 11px; font-size: 9pt; font-weight: 700; }
            QLabel#title { color: #F8FAFF; font-size: 29px; font-weight: 800; }
            QLabel#subtitle { color: #B8C8E1; font-size: 11pt; }
            QFrame#card { background: #FFFFFF; border: 1px solid #D7E1EE; border-radius: 16px; }
            QLabel#cardTitle, QLabel#sectionTitle { color: #17233B; font-size: 16px; font-weight: 800; }
            QLabel#sectionHint { color: #7A8BA5; font-size: 9pt; }
            QLabel#statusChip { color: #355070; background: #F1F5FA; border: 1px solid #D7E1EE; border-radius: 10px; padding: 6px 10px; font-size: 9pt; font-weight: 700; }
            QLabel#fieldLabel, QLabel#metricLabel { color: #344765; font-size: 10pt; font-weight: 700; }
            QLabel#infoLabel { color: #526782; font-size: 10pt; }
            QLabel#infoPill { color: #315FAD; background: #EEF4FF; border: 1px solid #CFE0FF; border-radius: 10px; padding: 7px 11px; font-size: 9pt; font-weight: 700; }
            QLabel#mutedLabel { color: #8291A8; font-size: 9pt; }
            QLabel#partial { color: #2E65C2; background: #F0F5FF; border: 1px solid #D8E6FF; border-radius: 10px; padding: 8px 11px; font-size: 10pt; }
            QLabel#warning { color: #9A4B2A; background: #FFF7ED; border: 1px solid #FED7AA; border-radius: 10px; padding: 8px 11px; font-size: 9pt; }
            QLabel#warning[active="false"] { color: transparent; background: transparent; border-color: transparent; }
            QComboBox { color: #17233B; background: #FFFFFF; border: 1px solid #C8D4E3; border-radius: 10px; padding: 7px 11px; selection-background-color: #DCEAFF; font-size: 10pt; }
            QComboBox:hover, QComboBox:focus { border-color: #4B7BE5; }
            QComboBox::drop-down { width: 30px; border: 0; }
            QComboBox QAbstractItemView { color: #17233B; background: #FFFFFF; border: 1px solid #C8D4E3; selection-background-color: #DCEAFF; selection-color: #17233B; padding: 4px; }
            QPlainTextEdit { color: #1E2D46; background: #FBFCFE; border: 1px solid #D7E1EE; border-radius: 12px; padding: 11px; selection-background-color: #CFE0FF; selection-color: #17233B; font-size: 10.5pt; }
            QPlainTextEdit:focus { border-color: #8EAEEC; }
            QProgressBar { color: #31507A; background: #EAF0F7; border: 0; border-radius: 8px; text-align: center; font-size: 9pt; font-weight: 800; }
            QProgressBar::chunk { background: #4778E5; border-radius: 8px; }
            QSplitter::handle:horizontal { background: #D7E1EE; width: 8px; margin: 10px 0; border-radius: 4px; }
            QPushButton { min-height: 30px; border-radius: 10px; padding: 3px 10px; font-size: 10pt; font-weight: 800; }
            QPushButton#primaryButton { color: #FFFFFF; background: #356CE6; border: 1px solid #356CE6; }
            QPushButton#primaryButton:hover { background: #285BD0; border-color: #285BD0; }
            QPushButton#stopButton { color: #FFFFFF; background: #D95555; border: 1px solid #D95555; }
            QPushButton#stopButton:hover { background: #BE3F3F; border-color: #BE3F3F; }
            QPushButton#refreshButton { color: #3564B8; background: #EEF4FF; border: 1px solid #CFE0FF; }
            QPushButton#refreshButton:hover { color: #264E98; background: #DDEAFF; border-color: #AFC9F5; }
            QPushButton#summaryButton { color: #2F62B7; background: #EEF4FF; border: 1px solid #CFE0FF; padding: 5px 12px; }
            QPushButton#summaryButton:hover { color: #244F98; background: #DDEAFF; border-color: #AFC9F5; }
            QPushButton#summaryButton:disabled { color: #97A7BC; background: #F1F4F8; border-color: #E0E6EE; }
            QPushButton:disabled { color: #9AA8BA; background: #E7EDF5; border-color: #E7EDF5; }
            """
        )

    def _load_devices(self) -> None:
        previous = self.device_combo.currentData() if self.device_combo.count() else None
        self.device_combo.clear()
        self.device_combo.addItem("系统默认（按 Windows 当前输入设备）", None)
        try:
            for index, name in list_input_devices():
                self.device_combo.addItem(name, index)
            if previous is not None:
                position = self.device_combo.findData(previous)
                if position >= 0:
                    self.device_combo.setCurrentIndex(position)
            self._set_input_warning("")
        except Exception as exc:
            self.device_combo.addItem("无法读取音频设备", None)
            self.status_label.setText(f"无法读取麦克风设备：{exc}")

    def _on_device_selected(self, _index: int) -> None:
        """User picked a different inference device from the dropdown."""
        # Skip programmatic selection while we sync with the backend on first
        # poll, and ignore changes while a meeting or switch is running.
        if not self.device_initialized or self.switching_device or self.worker is not None:
            return
        device = self.device_select_combo.currentData()
        if device in (None, ""):
            return
        if device == str(self.health.get("device", "")).lower():
            return
        self._switch_device(device)

    def _switch_device(self, device: str) -> None:
        try:
            payload = json.dumps({"device": device}).encode("utf-8")
            req = urllib.request.Request(
                f"{self.base_url}/api/device",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=3.0) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "ignore")
            self.status_label.setText(f"切换设备被拒绝：{detail}")
            self._sync_device_combo()
            return
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            self.status_label.setText(f"无法连接后端以切换设备：{exc}")
            self._sync_device_combo()
            return
        if result.get("status") != "switching":
            self.status_label.setText("后端未确认设备切换")
            return
        # Persist so the choice survives a backend restart.
        self._write_env_device(device)
        self.settings.setValue("device", device)
        self.switching_device = True
        self.device_select_combo.setEnabled(False)
        self.start_button.setEnabled(False)
        self.status_label.setText("正在切换推理设备，模型重新加载中，请稍候（约 10–60 秒）…")
        self.status_hint.setText("切换推理设备")

    @staticmethod
    def _write_env_device(device: str) -> None:
        """Persist MEETING_DEVICE into .env so the next backend launch keeps it.

        Defensive: never touches other lines, and bails out if the file is
        missing instead of creating a fresh .env that would erase credentials.
        """
        env_path = Path(__file__).resolve().parent.parent / ".env"
        if not env_path.is_file():
            return
        try:
            lines = env_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return
        key = "MEETING_DEVICE"
        new_line = f"{key}={device}"
        replaced = False
        for index, line in enumerate(lines):
            stripped = line.strip()
            if stripped == "" or stripped.startswith("#"):
                continue
            if stripped.split("=", 1)[0].strip().upper() == key:
                lines[index] = new_line
                replaced = True
                break
        if not replaced:
            lines.append(new_line)
        try:
            env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except OSError:
            return

    def _sync_device_combo(self) -> None:
        """Reflect the backend's current device in the dropdown (no re-trigger)."""
        backend_device = str(self.health.get("device", "")).lower()
        position = self.device_select_combo.findData(backend_device)
        if position >= 0 and position != self.device_select_combo.currentIndex():
            self.device_select_combo.blockSignals(True)
            self.device_select_combo.setCurrentIndex(position)
            self.device_select_combo.blockSignals(False)

    def _poll_health(self) -> None:
        try:
            with urllib.request.urlopen(f"{self.base_url}/api/health", timeout=1.0) as response:
                self.health = json.loads(response.read().decode("utf-8"))
            ready = self.health.get("status") == "ready"
            self.backend_label.setText(f"后端：{'已就绪' if ready else self.health.get('status', '未知')}")
            self.gpu_label.setText(f"GPU：{str(self.health.get('device', '—')).upper()}")
            self.jimo_label.setText(f"积墨：{'已配置' if self.health.get('jimo_configured') else '未配置'}")
            switching = bool(self.health.get("switching"))
            switch_error = self.health.get("switch_error")

            # First successful poll: align the dropdown with the backend device.
            if not self.device_initialized:
                self._sync_device_combo()
                self.device_initialized = True

            if switching:
                # A switch is in flight: keep controls locked and show progress.
                self.switching_device = True
                self.device_select_combo.setEnabled(False)
                self.refresh_button.setEnabled(False)
                self.start_button.setEnabled(False)
                self.status_hint.setText("切换推理设备")
                self.status_label.setText("正在切换推理设备，模型重新加载中，请稍候…")
                return

            # Switch finished (successfully or not): release the lock.
            if self.switching_device:
                self.switching_device = False
                self.device_select_combo.setEnabled(True)
                self.refresh_button.setEnabled(True)
                self._sync_device_combo()

            if self.worker is None:
                self.start_button.setEnabled(ready and self.summary_worker is None)
                if ready:
                    self.status_hint.setText("服务在线 · 可开始")
                    if switch_error:
                        self.status_label.setText(f"切换失败：{switch_error}（仍使用原设备）")
                    elif self.summary_worker is None and self.last_meeting_id is None:
                        self.status_label.setText("服务和模型已就绪，可以开始会议")
                else:
                    self.status_hint.setText("模型加载中")
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            self.backend_label.setText("后端：未连接")
            self.start_button.setEnabled(False)
            self.status_hint.setText("等待本机后端")
            if self.worker is None:
                self.status_label.setText("等待本地后端启动…")

    def _toggle_meeting(self) -> None:
        if self.worker is not None:
            self.worker.request_stop()
            self.start_button.setEnabled(False)
            self.status_label.setText("正在停止并保存完整逐句稿…")
            return
        if self.summary_worker is not None:
            return
        self.transcript.clear()
        self.summary.clear()
        self.last_meeting_id = None
        self.summary_button.setText("生成会议纪要")
        self.summary_button.setEnabled(False)
        self.partial_label.setText("等待发言")
        self.language_label.setText("当前语言：—")
        self.volume_bar.setValue(0)
        self.audio_stats_label.setText("正在打开麦克风…")
        self._set_input_warning("")
        self.meeting_started_at = time.monotonic()
        self.last_audio_level = 0.0
        self.last_backend_packets = 0
        self.start_button.setText("停止会议")
        self.start_button.setObjectName("stopButton")
        self.start_button.style().unpolish(self.start_button)
        self.start_button.style().polish(self.start_button)
        self.device_combo.setEnabled(False)
        self.refresh_button.setEnabled(False)
        self.device_select_combo.setEnabled(False)
        self.worker_thread = QThread(self)
        self.worker = MeetingWorker(self.base_url, self.device_combo.currentData())
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.status.connect(self.status_label.setText)
        self.worker.audio_level.connect(self._set_audio_level)
        self.worker.audio_stats.connect(self._set_audio_stats)
        self.worker.partial.connect(self._show_partial)
        self.worker.partial_clear.connect(self._clear_partial)
        self.worker.utterance.connect(self._append_utterance)
        self.worker.summary_ready.connect(self._summary_ready)
        self.worker.summary_delta.connect(self.summary.insertPlainText)
        self.worker.summary_reset.connect(self.summary.clear)
        self.worker.summary_complete.connect(self._summary_complete)
        self.worker.error.connect(self._show_error)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self._worker_finished)
        self.worker_thread.start()

    @pyqtSlot(float)
    def _set_audio_level(self, value: float) -> None:
        # UI updates are deliberately defensive: a malformed level must not
        # make round()/QProgressBar throw and terminate the desktop process.
        if not math.isfinite(value) or value < 0.0:
            value = 0.0
        value = min(1.0, value)
        self.last_audio_level = value
        # A dB-like scale keeps quiet but valid speech visible instead of
        # making the bar look permanently stuck at 0%.
        db = 20.0 * math.log10(max(value, 1e-6))
        percent = max(0, min(100, round((db + 60.0) * 100.0 / 60.0)))
        self.volume_bar.setValue(percent)
        elapsed = time.monotonic() - self.meeting_started_at if self.meeting_started_at else 0
        if elapsed > 2.0 and self.last_backend_packets > 20 and value < 0.0015:
            self._set_input_warning(
                "未检测到有效声音，请检查输入设备、麦克风权限和音量；可尝试 WASAPI。"
            )
        elif value >= 0.003:
            self._set_input_warning("")

    def _set_input_warning(self, message: str) -> None:
        """Update the fixed warning slot without changing window geometry."""
        active = bool(message)
        if self.input_warning.text() == message and self.input_warning.property("active") == active:
            return
        self.input_warning.setText(message)
        self.input_warning.setProperty("active", active)
        style = self.input_warning.style()
        style.unpolish(self.input_warning)
        style.polish(self.input_warning)

    @pyqtSlot(int, int, int)
    def _set_audio_stats(self, sent_frames: int, sent_bytes: int, backend_packets: int) -> None:
        self.last_backend_packets = backend_packets
        self.audio_stats_label.setText(
            f"前端 {sent_frames} 帧 / {sent_bytes // 1024} KB · 后端 {backend_packets} 包"
        )

    @pyqtSlot(dict)
    def _show_partial(self, event: dict) -> None:
        language = language_label(event.get("language")) if event.get("language") else ""
        self.language_label.setText(f"当前语言：{language or '识别中'}")
        text = str(event.get("text", ""))
        if event.get("language") == "zh":
            text = simplify_chinese(text)
        self.partial_label.setText(f"临时（{language}）：{text}")

    @pyqtSlot()
    def _clear_partial(self) -> None:
        self.partial_label.setText("正在处理稳定片段…")

    @pyqtSlot(dict)
    def _append_utterance(self, item: dict) -> None:
        language = language_label(item.get("language"))
        self.language_label.setText(f"当前语言：{language or '未知'}")
        start = float(item.get("start", 0))
        end = float(item.get("end", 0))
        def stamp(value: float) -> str:
            ms = max(0, round(value * 1000))
            return f"{ms // 3600000:02d}:{(ms // 60000) % 60:02d}:{(ms // 1000) % 60:02d}.{ms % 1000:03d}"
        text = str(item.get("text", ""))
        if item.get("language") == "zh":
            text = simplify_chinese(text)
        source = f"[{stamp(start)} - {stamp(end)}] 演讲人{item.get('speaker_id', 1)}（{language}）：“{text}”"
        translation = f"[{stamp(start)} - {stamp(end)}] 演讲人{item.get('speaker_id', 1)}（中文翻译）：“{item.get('translation_zh', item.get('translation_en', ''))}”"
        self.transcript.appendPlainText(source + "\n" + translation)
        self.transcript.verticalScrollBar().setValue(self.transcript.verticalScrollBar().maximum())
        self.partial_label.setText("正在监听下一段语音")

    @pyqtSlot(str, list)
    def _summary_ready(self, session_id: str, _files: list) -> None:
        if not session_id:
            return
        self.last_meeting_id = session_id
        self.summary.setPlainText("会议已保存。点击右上角“生成会议纪要”后，才会请求 AI。")
        self.summary_button.setText("生成会议纪要")
        self.summary_button.setEnabled(True)
        self.status_label.setText("完整逐句稿已保存，可以手动生成会议纪要")

    @pyqtSlot()
    def _generate_summary(self) -> None:
        if not self.last_meeting_id or self.summary_worker is not None or self.worker is not None:
            return
        self.summary.clear()
        self.summary_button.setText("生成中…")
        self.summary_button.setEnabled(False)
        self.status_label.setText("正在请求积墨 AI 生成会议纪要…")
        self.start_button.setEnabled(False)
        self.summary_thread = QThread(self)
        self.summary_worker = SummaryWorker(self.base_url, self.last_meeting_id)
        self.summary_worker.moveToThread(self.summary_thread)
        self.summary_thread.started.connect(self.summary_worker.run)
        self.summary_worker.status.connect(self.status_label.setText)
        self.summary_worker.summary_delta.connect(self.summary.insertPlainText)
        self.summary_worker.summary_reset.connect(self.summary.clear)
        self.summary_worker.summary_complete.connect(self._summary_complete)
        self.summary_worker.error.connect(self._summary_error)
        self.summary_worker.finished.connect(self.summary_thread.quit)
        self.summary_thread.finished.connect(self._summary_worker_finished)
        self.summary_thread.start()

    @pyqtSlot(str)
    def _summary_error(self, message: str) -> None:
        self.status_label.setText(f"会议纪要生成失败：{message}")
        self.summary_button.setText("重试生成")
        self.summary_button.setEnabled(bool(self.last_meeting_id))

    def _summary_worker_finished(self) -> None:
        if self.summary_thread:
            self.summary_thread.deleteLater()
        self.summary_worker = None
        self.summary_thread = None
        if self.worker is None and self.health.get("status") == "ready":
            self.start_button.setEnabled(True)

    @pyqtSlot(str, list)
    def _summary_complete(self, content: str, _files: list) -> None:
        self.summary.setPlainText(content)
        self.summary.verticalScrollBar().setValue(self.summary.verticalScrollBar().maximum())
        self.summary_button.setText("纪要已生成")
        self.summary_button.setEnabled(False)
        self.status_label.setText("会议纪要已生成并保存")

    @pyqtSlot(str)
    def _show_error(self, message: str) -> None:
        self.status_label.setText(message)

    def _worker_finished(self) -> None:
        if self.worker_thread:
            self.worker_thread.deleteLater()
        self.worker = None
        self.worker_thread = None
        self.start_button.setText("开始会议")
        self.start_button.setObjectName("primaryButton")
        self.start_button.style().unpolish(self.start_button)
        self.start_button.style().polish(self.start_button)
        self.start_button.setEnabled(
            self.health.get("status") == "ready" and self.summary_worker is None
        )
        self.device_combo.setEnabled(True)
        self.refresh_button.setEnabled(True)
        self.device_select_combo.setEnabled(True)
        self._set_input_warning("")
        self.volume_bar.setValue(0)
        self.audio_stats_label.setText("尚未采集音频")

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.worker is not None:
            self.worker.request_stop()
            if self.worker_thread:
                self.worker_thread.quit()
                self.worker_thread.wait(3000)
        if self.summary_thread:
            self.summary_thread.quit()
            self.summary_thread.wait(3000)
        if self.backend_process and self.backend_process.poll() is None:
            self.backend_process.terminate()
            try:
                self.backend_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.backend_process.kill()
        event.accept()


def _backend_is_reachable(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{base_url}/api/health", timeout=0.5):
            return True
    except OSError:
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="本机 PyQt 实时会议转译")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-backend", action="store_true")
    args = parser.parse_args(argv)
    base_url = f"http://{args.host}:{args.port}"
    backend_process = None
    if not args.no_backend and not _backend_is_reachable(base_url):
        project_root = Path(__file__).resolve().parent.parent
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        backend_process = subprocess.Popen(
            [sys.executable, "-m", "realtime_meeting.cli", "--no-browser", "--host", args.host, "--port", str(args.port)],
            cwd=project_root,
            creationflags=creationflags,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
    app = QApplication(sys.argv)
    # Fusion keeps the visual palette consistent on Windows, even when the
    # system is using a dark theme. All text colors are also explicit in the
    # stylesheet so labels never become white-on-white.
    app.setStyle("Fusion")
    app.setFont(QFont("Microsoft YaHei", 10))
    window = DesktopWindow(base_url, backend_process)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
