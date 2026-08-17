"""Build a short product brief for the paragraph-based Qwen architecture.

The product brief is deliberately generated from the current contract instead
of carrying historical implementation details.  ``python-docx`` remains an
optional documentation-only dependency; the runtime does not need it.
"""

from __future__ import annotations

from pathlib import Path

try:
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Inches, Pt
except ImportError as exc:  # pragma: no cover - documentation helper
    raise SystemExit("build_product_doc.py requires python-docx") from exc


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "meeting_product_brief_v2.docx"


def _style(document: Document) -> None:
    section = document.sections[0]
    section.top_margin = Inches(0.65)
    section.bottom_margin = Inches(0.65)
    section.left_margin = Inches(0.75)
    section.right_margin = Inches(0.75)
    styles = document.styles
    styles["Normal"].font.name = "Microsoft YaHei"
    styles["Normal"].font.size = Pt(10.5)
    styles["Title"].font.name = "Microsoft YaHei"
    styles["Title"].font.size = Pt(28)
    styles["Heading 1"].font.name = "Microsoft YaHei"
    styles["Heading 1"].font.size = Pt(17)
    styles["Heading 2"].font.name = "Microsoft YaHei"
    styles["Heading 2"].font.size = Pt(12.5)


def _paragraph(document: Document, text: str, *, bold: bool = False) -> None:
    paragraph = document.add_paragraph()
    run = paragraph.add_run(text)
    run.bold = bold


def _bullets(document: Document, values: list[str]) -> None:
    for value in values:
        document.add_paragraph(value, style="List Bullet")


def _table(document: Document, headers: list[str], rows: list[list[str]]) -> None:
    table = document.add_table(rows=1, cols=len(headers))
    table.style = "Light Shading Accent 1"
    for cell, value in zip(table.rows[0].cells, headers):
        cell.text = value
    for row in rows:
        cells = table.add_row().cells
        for cell, value in zip(cells, row):
            cell.text = value


def build_document() -> Path:
    document = Document()
    _style(document)
    document.core_properties.title = "实时会议转写产品说明（Qwen 段落架构）"
    document.core_properties.subject = "Qwen 双模型、自动语言方言判断、段落式实时转写与翻译"

    title = document.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.add_run("实时会议转写")
    subtitle = document.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.add_run("Qwen 双模型 + 自动语言/方言判断 + 段落式输出").bold = True

    document.add_heading("1. 产品定义", level=1)
    _paragraph(document, "系统在录音期间持续输出可读的会议段落：Qwen3-ASR-0.6B 负责识别语言和中文方言，Qwen3-ASR-1.7B 负责实时转写；英文和德文通过本地 OPUS-MT 翻译为简体中文，中文及中文方言直接规范为简体中文。")
    _bullets(document, [
        "一个连续的语言/方言语音段对应一个段落卡片，partial 只修订当前卡片。",
        "静音达到阈值或语言/方言稳定切换时结束段落；技术性音频切分不会制造额外显示段落。",
        "系统不保存人员身份、手动语言锁定或会后第二套 ASR 结果。",
        "旧会议数据不迁移；新存储契约使用 transcript schema 2.0。",
    ])

    document.add_heading("2. 模型与适配层", level=1)
    _table(document, ["用途", "模型", "传参策略"], [
        ["实时 ASR", "Qwen/Qwen3-ASR-1.7B", "en=English；de=German；中文=Chinese；粤语（香港/广东口音）=Cantonese；其余官方中文方言使用 Chinese 加对应方言上下文提示。"],
        ["语言/方言判断", "Qwen/Qwen3-ASR-0.6B", "新段开始约 0.8 秒判断一次；冲突连续确认，不对每个 partial 调用。"],
        ["实时失败回退", "Qwen/Qwen3-ASR-0.6B", "主模型异常时继续输出，并保留实际使用的 asr_model。"],
        ["英文/德文翻译", "OPUS-MT en→zh / de→zh", "按稳定原文前缀顺序翻译；旧 source_revision 结果丢弃。"],
        ["VAD", "FunASR FSMN-VAD", "保留既有音频阈值与静音结束机制。"],
    ])

    document.add_heading("3. 实时数据流", level=1)
    _paragraph(document, "浏览器 PCM 音频 → VAD/技术分段 → 0.6B 语言判断 → 1.7B 实时 ASR → 段落聚合器 → 稳定前缀翻译队列 → TranscriptStore 与 paragraph_update → 前端段落卡片。")
    _bullets(document, [
        "首次语言判断只形成候选；相同新结果连续确认后才切换稳定语言/方言。短暂 unknown 不切段。",
        "原文 partial 实时更新；译文只对稳定前缀异步更新；段落关闭时使用完整最终原文重新翻译。",
        "翻译队列按段落顺序单 worker 执行，失败自动重试；持续失败保留原文并提供 translation/retry。",
        "停止流程：flush 最后音频段 → 等待实时 ASR → 等待翻译队列 → recording_state=complete → ready_for_summary。",
    ])

    document.add_heading("4. 存储和 WebSocket 契约", level=1)
    _paragraph(document, "每个段落由同一个 segment_id 标识，通过 revision 追加修订事件。段落记录包含 id、segment_id、start、end、language、speech_variant、language_confidence、text、translation_zh、translation_status、revision、source_revision、closed、asr_model、language_source、translation_model。")
    _paragraph(document, "实时事件统一为 paragraph_update；前端按 segment_id 原地更新，不追加逐句节点。transcript.json 保存 paragraphs，transcript.jsonl 保存追加事件，schema_version 为 2.0。")
    _table(document, ["接口", "用途"], [
        ["GET /api/v2/meetings/{id}", "会议状态、活动段落、语言和翻译队列快照。"],
        ["GET /api/v2/meetings/{id}/transcript", "schema 2.0 段落列表。"],
        ["POST /api/v2/meetings/{id}/translation/retry", "重新排队失败的段落翻译。"],
        ["POST /api/v2/meetings/{id}/summary", "仅在翻译队列完成后生成纪要。"],
        ["POST /api/v2/meetings/{id}/todo", "基于已保存纪要生成 To-do。"],
    ])

    document.add_heading("5. 导出和验收重点", level=1)
    _bullets(document, [
        "meeting_transcript.md、translated_zh.md 和语言原文文件按段落输出时间、语言/方言、原文和译文。",
        "manifest.json 不含人员或旧后台处理字段；不再生成 speaker_segments.json。",
        "验收覆盖中英德切换、普通话与 Qwen 官方 22 类中文方言代表类别切换、长连续语音、短暂低音量、中英夹杂、噪声和多人同时讲话。",
        "测试验证同一 segment_id 的多次 partial 更新、稳定前缀的过期结果丢弃、中文不调用翻译模型、技术切分合并和停止顺序。",
    ])

    document.add_heading("6. 当前模型配置", level=1)
    _table(document, ["配置", "值"], [
        ["MEETING_ASR_PRIMARY", "Qwen/Qwen3-ASR-1.7B"],
        ["MEETING_ASR_FALLBACK", "Qwen/Qwen3-ASR-0.6B"],
        ["MEETING_ASR_LANGUAGE_ID", "Qwen/Qwen3-ASR-0.6B"],
        ["TRANSCRIPT_SCHEMA_VERSION", "2.0"],
        ["language_id_min_seconds", "0.8"],
    ])
    _paragraph(document, "模型能力与部署前检查以仓库 README.md、DEPLOYMENT.md、config.py、runtime.py、session.py 和 storage.py 为准。")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    document.save(OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    print(build_document())
