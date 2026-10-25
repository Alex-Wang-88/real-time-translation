import os, sys, json, time

REPORT = r"C:\Users\12992\Desktop\work\code\real-time-translation\scripts\_verify_report.json"
OUT_DIR = r"C:\Users\12992\Desktop\work\code\real-time-translation\result\meeting_frontend_render"
MEETING_ID = "manufacturing-role-meeting-v3-realtime"
BASE = "http://127.0.0.1:8765"
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

report = {}

def log(*a):
    msg = " ".join(str(x) for x in a)
    report.setdefault("log", []).append(msg)
    print(msg, flush=True)

try:
    from playwright.sync_api import sync_playwright
    log("playwright imported")
except Exception as e:
    report["error"] = f"playwright import failed: {e}"
    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    sys.exit(2)

with sync_playwright() as p:
    b = p.chromium.launch(executable_path=CHROME, args=["--no-sandbox", "--disable-gpu"])
    pg = b.new_page(viewport={"width": 1440, "height": 900})
    pg.on("pageerror", lambda e: log("[pageerror]", str(e)[:200]))

    pg.goto(BASE, wait_until="domcontentloaded", timeout=30000)
    pg.wait_for_selector(".meeting-entry-select", timeout=30000)
    log("meeting list rendered")

    pg.evaluate(
        """(mid) => {
            const entries = [...document.querySelectorAll('.meeting-entry')];
            const t = entries.find(e => e.dataset.meetingId === mid) || entries[0];
            t.querySelector('.meeting-entry-select').click();
        }""",
        MEETING_ID,
    )
    log("clicked meeting:", MEETING_ID)

    # Wait until refined swap is active: right card hidden + bubbles present.
    try:
        pg.wait_for_function(
            """() => {
                const card = document.getElementById('refinedCard');
                const bubbles = document.querySelectorAll('#transcriptList .transcript-bubble.original-bubble');
                return card && card.hidden === true && bubbles.length > 0;
            }""",
            timeout=30000,
        )
        log("refined swap active (card hidden + bubbles present)")
    except Exception as e:
        log("wait for swap timed out:", repr(e))

    def sample():
        return pg.evaluate(
            """() => {
                const card = document.getElementById('refinedCard');
                const tag = document.getElementById('transcriptModeTag');
                const bubbles = [...document.querySelectorAll('#transcriptList .transcript-bubble.original-bubble')];
                const firstText = bubbles[0] ? bubbles[0].textContent : '';
                const summaryBadge = document.getElementById('summaryBadge');
                const todoBadge = document.getElementById('todoBadge');
                const summaryText = document.getElementById('summaryText');
                return {
                    refinedCardHidden: card ? card.hidden : null,
                    modeTagVisible: tag ? !tag.hidden : null,
                    bubbleCount: bubbles.length,
                    firstBubbleText: firstText.slice(0, 80),
                    summaryBadge: summaryBadge ? summaryBadge.textContent : null,
                    todoBadge: todoBadge ? todoBadge.textContent : null,
                    summaryTextLen: summaryText ? summaryText.textContent.length : 0,
                };
            }"""
        )

    time.sleep(6)
    info1 = sample()
    log("sample@6s:", json.dumps(info1, ensure_ascii=False))
    time.sleep(4)
    info = sample()
    log("sample@10s:", json.dumps(info, ensure_ascii=False))
    report["frontend"] = info
    report["frontend_sample_6s"] = info1

    os.makedirs(OUT_DIR, exist_ok=True)
    full = os.path.join(OUT_DIR, "refined_swap_full.png")
    view = os.path.join(OUT_DIR, "refined_swap_view.png")
    pg.screenshot(path=view)
    pg.screenshot(path=full, full_page=True)
    report["screenshots"] = [view, full]
    log("screenshots saved")

    b.close()

report["ok"] = True
with open(REPORT, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
