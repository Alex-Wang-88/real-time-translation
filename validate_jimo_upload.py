"""Validate the real 'upload audio -> Jimo -> three-segment result' path.

Runs standalone (no server, no ASR) to de-risk the 'pass the file to the
platform API' requirement before committing to the 13-minute full run.
"""
import asyncio
import os

os.environ.setdefault(
    "JIMO_API_URL",
    "https://jimoai-bot-api.xiaohuodui.cn/v2/chat/completions/share?shareId=bDujbPw00CUHbFpQ",
)
os.environ.setdefault(
    "JIMO_AUTHORIZATION",
    "amlheWl3ZW5odWE6MTJiZTA4OGFkOWUwYzdmNWQ1YjVjY2RlNWIzMzFhZWI=",
)
os.environ.setdefault("JIMO_UPLOAD_SHARE_ID", "bDujbPw00CUHbFpQ")

from realtime_meeting.config import load_settings
from realtime_meeting.jimo_upload import JimoUploadClient
from realtime_meeting.jimo import MeetingAgent

SEG = r"tests/fixtures/manufacturing_role_meeting_v3/audio/segments/role_meeting_v3_001.wav"


async def main() -> None:
    settings = load_settings()
    print("jimo_configured:", settings.jimo_configured)
    print("jimo_upload_configured:", settings.jimo_upload_configured)
    print("upload_share_id:", repr(settings.jimo_upload_share_id))

    print("\n[1/2] uploading audio to Jimo COS ...")
    async with JimoUploadClient(settings) as up:
        res = await up.upload_audio(SEG)
    print("UPLOAD OK")
    print("  url:", res.url[:140])
    print("  file_id:", res.file_id)

    print("\n[2/2] requesting three-segment agent result ...")
    agent = MeetingAgent(settings)
    result = await agent.process(
        "validate-test",
        audio_files=[res.to_dict()],
        title="验证片段",
    )
    d = result.to_dict()
    print("AGENT OK")
    print("  transcript_items:", len(d["transcript"]))
    print("  summary_len:", len(d["summary_markdown"]))
    print("  todo_items:", len(d["todo"].get("items", [])))
    print("  summary_head:", d["summary_markdown"][:240].replace("\n", " "))


if __name__ == "__main__":
    asyncio.run(main())
