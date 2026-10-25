"""
起会记前端服务（默认加载模型 -> health ready -> 前端正常渲染会议列表）。
SessionManager 启动时会从 result/meetings/ 自动恢复已有会议，前端可直接打开。
"""
import uvicorn
from realtime_meeting.server import create_app

app = create_app()  # load_models=True 默认，模型本地缓存

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="warning")
