"""测试积墨 COS 音频上传链路（凭据从 .env 读取，不硬编码）。"""
import asyncio
from pathlib import Path

from realtime_meeting.jimo_upload import JimoUploadClient
from realtime_meeting.config import load_settings

AUDIO = Path(__file__).with_name("..") / "tests" / "fixtures" / "manufacturing_role_meeting_v3" / "audio" / "full_normal_random_pauses_16k.wav"

async def main():
    settings = load_settings()
    async with JimoUploadClient(settings) as client:
        result = await client.upload_audio(AUDIO, sign_download_url=False)
        print("UPLOAD_OK")
        print("object_url:", result.object_url)
        print("download_url:", result.download_url)
        print("file_id:", result.file_id)
        print("object_key:", result.object_key)

if __name__ == "__main__":
    asyncio.run(main())
