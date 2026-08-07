from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QCoreApplication

from realtime_meeting.desktop import MeetingWorker


def test_audio_callback_sanitizes_nonfinite_driver_samples():
    """A transient NaN/Inf from a driver must never reach the UI or PCM queue."""

    app = QCoreApplication.instance() or QCoreApplication([])
    worker = MeetingWorker("http://127.0.0.1:8765", None)
    levels: list[float] = []
    worker.audio_level.connect(levels.append)

    worker._audio_callback(
        np.array([[np.nan], [np.inf], [-np.inf], [100.0]], dtype=np.float32),
        4,
        None,
        None,
    )

    pcm = worker.audio_queue.get_nowait()
    samples = np.frombuffer(pcm, dtype=np.int16)
    assert np.isfinite(samples).all()
    assert levels and np.isfinite(levels[-1])
    assert levels[-1] >= 0.0
    app.processEvents()
