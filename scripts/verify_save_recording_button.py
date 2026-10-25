"""
验证“保存录音”按钮在录音完成态下正确替换小字提示。
依赖：服务已在 http://127.0.0.1:8765 运行。
产物：result/meeting_frontend_render/save_recording_*.png
"""
import os, time, json
from playwright.sync_api import sync_playwright

MEETING_ID = "manufacturing-role-meeting-v3-realtime"
BASE = "http://127.0.0.1:8765"
OUT = r"C:\Users\12992\Desktop\work\code\real-time-translation\result\meeting_frontend_render"
os.makedirs(OUT, exist_ok=True)

CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
REPORT_PATH = os.path.join(OUT, "_save_recording_report.json")


def log(*a):
    print(*a, flush=True)


def dump_panel(pg):
    return pg.evaluate("""() => ({
        recordingState: document.getElementById('recordingState')?.textContent,
        recordingHintVisible: document.getElementById('recordingHint') && !document.getElementById('recordingHint').hidden,
        saveRecordingVisible: document.getElementById('saveRecording') && !document.getElementById('saveRecording').hidden,
        saveRecordingText: document.getElementById('saveRecording')?.textContent,
    })""")


def fetch_meeting_snapshot(meeting_id):
    import urllib.request
    req = urllib.request.Request(
        f"{BASE}/api/v2/meetings/{meeting_id}",
        headers={"X-User-Id": "local", "Authorization": "Bearer local"},
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read().decode("utf-8"))


with sync_playwright() as p:
    log("== 启动 Chrome")
    b = p.chromium.launch(executable_path=CHROME, args=["--no-sandbox", "--disable-gpu"])
    pg = b.new_page(viewport={"width": 1440, "height": 900})
    pg.on("console", lambda m: log("  [console]", m.type, m.text[:120]))
    pg.on("pageerror", lambda e: log("  [pageerror]", str(e)[:200]))

    cached_snapshot = [None]

    def route_meeting(route, request):
        url = request.url
        if cached_snapshot[0] and f"/api/v2/meetings/{MEETING_ID}" in url and "transcript" not in url:
            log("== 拦截轮询请求，返回缓存快照")
            route.fulfill(status=200, content_type="application/json", body=cached_snapshot[0])
            return
        route.continue_()

    pg.route(f"{BASE}/api/v2/meetings/*", route_meeting)

    log("== 打开首页")
    pg.goto(BASE, wait_until="domcontentloaded", timeout=30000)
    pg.wait_for_selector(".meeting-entry-select", timeout=30000)
    log("== 会议列表数量:", pg.evaluate("document.querySelectorAll('.meeting-entry').length"))

    # 点击目标会议
    clicked = pg.evaluate(
        """(mid) => {
            const entries = [...document.querySelectorAll('.meeting-entry')];
            const t = entries.find(e => e.dataset.meetingId === mid) || entries[0];
            if (!t) return false;
            t.querySelector('.meeting-entry-select').click();
            return true;
        }""", MEETING_ID)
    log("== 点击目标会议:", clicked)

    # 等待 saveRecording 按钮可见
    deadline = time.time() + 25
    panel = {}
    snapshot = {}
    while time.time() < deadline:
        panel = dump_panel(pg)
        if panel.get("saveRecordingVisible"):
            # 缓存快照避免被轮询刷掉
            try:
                snapshot = fetch_meeting_snapshot(MEETING_ID)
                cached_snapshot[0] = json.dumps(snapshot).encode("utf-8")
                log("== 已缓存完成快照，audio_segments:", len(snapshot.get("audio_segments", [])))
            except Exception as e:
                log("== 缓存快照失败:", e)
            break
        time.sleep(0.2)

    log("== 面板状态:", panel)
    log("== 后端 snapshot audio_segments:", len(snapshot.get("audio_segments", [])))

    # 截图
    time.sleep(0.3)
    full = os.path.join(OUT, "save_recording_full.png")
    view = os.path.join(OUT, "save_recording_view.png")
    pg.screenshot(path=view)
    pg.screenshot(path=full, full_page=True)
    log("== 截图完成 ->", OUT)

    report = {
        "meeting_id": MEETING_ID,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "panel": panel,
        "view_screenshot": view,
        "full_screenshot": full,
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    log("== 报告:", REPORT_PATH)

    try:
        b.close()
    except Exception:
        pass

log("DONE")
