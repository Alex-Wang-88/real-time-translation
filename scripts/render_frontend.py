"""
打开会记前端，点开指定会议，等待三段结果（精修转写/纪要/To-do）渲染后截图。
依赖：服务已在 http://127.0.0.1:8765 运行。
产物：result/meeting_frontend_render/

行为：
- 若后端已完成 summary/todo，前端轮询约 3 秒后会自动渲染，脚本直接截图，不再触发二次请求。
- 若尚未生成，则点击"生成三段结果"按钮触发一次请求。
- 检测到完成态后，拦截并缓存会议 GET 请求，避免前端轮询把状态刷回 idle。
"""
import os, time
from playwright.sync_api import sync_playwright

MEETING_ID = "manufacturing-role-meeting-v3-realtime"
BASE = "http://127.0.0.1:8765"
OUT = r"C:\Users\12992\Desktop\work\code\real-time-translation\result\meeting_frontend_render"
os.makedirs(OUT, exist_ok=True)

CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

def log(*a):
    print(*a, flush=True)

def dump_panel(pg):
    return pg.evaluate("""() => ({
        summaryBadge: document.getElementById('summaryBadge')?.textContent,
        todoBadge: document.getElementById('todoBadge')?.textContent,
        refinedBadge: document.getElementById('refinedBadge')?.textContent,
        retryBtn: document.getElementById('retrySummary')?.textContent,
        summaryTextLen: document.getElementById('summaryText')?.innerText?.length || 0,
        todoTextLen: document.getElementById('todoText')?.innerText?.length || 0,
        refinedTextLen: document.getElementById('refinedTranscriptText')?.innerText?.length || 0,
    })""")

def is_complete(panel):
    return panel.get("summaryBadge") == "已完成" or panel.get("retryBtn") == "重新生成三段结果"

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

    log("== 打开首页 (domcontentloaded)")
    pg.goto(BASE, wait_until="domcontentloaded", timeout=30000)
    log("== 首页 DOM 已加载，等待会议列表选择器")
    pg.wait_for_selector(".meeting-entry-select", timeout=30000)
    n = pg.evaluate("document.querySelectorAll('.meeting-entry').length")
    log("== 会议列表数量:", n)

    # 点开目标会议
    clicked = pg.evaluate(
        """(mid) => {
            const entries = [...document.querySelectorAll('.meeting-entry')];
            const t = entries.find(e => e.dataset.meetingId === mid) || entries[0];
            if (!t) return false;
            t.querySelector('.meeting-entry-select').click();
            return true;
        }""", MEETING_ID)
    log("== 点击目标会议:", clicked)

    # 轮询等待右侧面板进入完成状态（后端已完成时，前端轮询约 3 秒后自动刷新）
    complete = False
    deadline = time.time() + 25
    while time.time() < deadline:
        panel = dump_panel(pg)
        if is_complete(panel):
            complete = True
            log("== 检测到完成态:", panel)
            # 缓存当前快照，拦截后续轮询
            try:
                import urllib.request
                req = urllib.request.Request(f"{BASE}/api/v2/meetings/{MEETING_ID}", headers={"X-User-Id":"local","Authorization":"Bearer local"})
                cached_snapshot[0] = urllib.request.urlopen(req, timeout=5).read()
                log("== 已缓存完成快照并启用拦截")
            except Exception as e:
                log("== 缓存快照失败:", e)
            time.sleep(0.3)
            break
        time.sleep(0.2)
    if not complete:
        log("== 25 秒内未检测到完成态，当前面板:", dump_panel(pg))

    # 如果尚未完成，才点击"生成三段结果"触发请求
    if not complete:
        try:
            pg.wait_for_selector("button:has-text('生成三段结果'):not(:has-text('重新'))", timeout=10000)
            btn = pg.query_selector("button:has-text('生成三段结果'):not(:has-text('重新'))")
            if btn:
                btn.click()
                log("== 已点击生成三段结果")
                end = time.time() + 120
                while time.time() < end:
                    panel = dump_panel(pg)
                    if is_complete(panel):
                        log("== 点击后检测到完成态:", panel)
                        break
                    time.sleep(0.2)
        except Exception as e:
            log("== 点击/等待生成三段结果失败:", repr(e))

    # 截图
    full = os.path.join(OUT, "huiji_frontend_full.png")
    view = os.path.join(OUT, "huiji_frontend_view.png")
    pg.screenshot(path=view)
    log("== viewport 截图完成，面板:", dump_panel(pg))
    pg.screenshot(path=full, full_page=True)
    log("== 截图完成 ->", OUT)
    log("== 截图后面板:", dump_panel(pg))
    try:
        b.close()
    except Exception:
        pass
log("DONE")
