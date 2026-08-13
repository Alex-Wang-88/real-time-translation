from __future__ import annotations

from pathlib import Path
from typing import Iterable

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "会记_产品介绍_正式版.docx"


# ------------------------------
# Design tokens: standard_business_brief
# Named overrides:
# - Chinese body/East Asia font: Microsoft YaHei; Latin font remains Calibri.
# - Product title block uses a restrained navy/teal accent for customer-facing readability.
# - Compact data tables use 9.2 pt text so technical matrices remain scannable.
# ------------------------------
NAVY = "0B2545"
BLUE = "2E74B5"
DARK_BLUE = "1F4D78"
TEAL = "0F766E"
INK = "17202A"
MUTED = "5B6770"
LIGHT_BLUE = "E8EEF5"
LIGHT_TEAL = "E8F4F2"
LIGHT_GRAY = "F2F4F7"
PALE_GOLD = "FFF8E8"
GOLD = "A56A00"
PALE_RED = "FCECEC"
RED = "9B1C1C"
WHITE = "FFFFFF"
BORDER = "CCD5DF"

PAGE_WIDTH_DXA = 9360
TABLE_INDENT_DXA = 120
CELL_MARGINS = {"top": 90, "bottom": 90, "start": 120, "end": 120}


def rgb(hex_value: str) -> RGBColor:
    return RGBColor.from_string(hex_value)


def set_run_font(run, *, name: str = "Calibri", east_asia: str = "Microsoft YaHei", size: float | None = None,
                 color: str | None = None, bold: bool | None = None, italic: bool | None = None,
                 underline: bool | None = None) -> None:
    run.font.name = name
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.insert(0, rfonts)
    rfonts.set(qn("w:ascii"), name)
    rfonts.set(qn("w:hAnsi"), name)
    rfonts.set(qn("w:eastAsia"), east_asia)
    rfonts.set(qn("w:cs"), east_asia)
    if size is not None:
        run.font.size = Pt(size)
    if color is not None:
        run.font.color.rgb = rgb(color)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    if underline is not None:
        run.underline = underline


def set_style_font(style, *, name: str = "Calibri", east_asia: str = "Microsoft YaHei", size: float | None = None,
                   color: str | None = None, bold: bool | None = None) -> None:
    style.font.name = name
    rpr = style._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.insert(0, rfonts)
    rfonts.set(qn("w:ascii"), name)
    rfonts.set(qn("w:hAnsi"), name)
    rfonts.set(qn("w:eastAsia"), east_asia)
    rfonts.set(qn("w:cs"), east_asia)
    if size is not None:
        style.font.size = Pt(size)
    if color is not None:
        style.font.color.rgb = rgb(color)
    if bold is not None:
        style.font.bold = bold


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)
    shd.set(qn("w:val"), "clear")


def set_paragraph_shading(paragraph, fill: str) -> None:
    p_pr = paragraph._p.get_or_add_pPr()
    shd = p_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        p_pr.append(shd)
    shd.set(qn("w:fill"), fill)
    shd.set(qn("w:val"), "clear")


def set_paragraph_borders(paragraph, *, color: str, size: str = "6") -> None:
    p_pr = paragraph._p.get_or_add_pPr()
    borders = p_pr.find(qn("w:pBdr"))
    if borders is None:
        borders = OxmlElement("w:pBdr")
        p_pr.append(borders)
    for name in ("top", "left", "bottom", "right"):
        edge = borders.find(qn(f"w:{name}"))
        if edge is None:
            edge = OxmlElement(f"w:{name}")
            borders.append(edge)
        edge.set(qn("w:val"), "single")
        edge.set(qn("w:sz"), size)
        edge.set(qn("w:space"), "5")
        edge.set(qn("w:color"), color)


def set_cell_margins(cell, *, top: int = 90, bottom: int = 90, start: int = 120, end: int = 120) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = tc_pr.find(qn("w:tcMar"))
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for tag, value in (("top", top), ("bottom", bottom), ("start", start), ("end", end)):
        node = tc_mar.find(qn(f"w:{tag}"))
        if node is None:
            node = OxmlElement(f"w:{tag}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_cell_width(cell, width_dxa: int) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_w = tc_pr.find(qn("w:tcW"))
    if tc_w is None:
        tc_w = OxmlElement("w:tcW")
        tc_pr.append(tc_w)
    tc_w.set(qn("w:w"), str(width_dxa))
    tc_w.set(qn("w:type"), "dxa")


def set_table_borders(table, *, color: str = BORDER, size: str = "6", inside: bool = True) -> None:
    tbl_pr = table._tbl.tblPr
    borders = tbl_pr.find(qn("w:tblBorders"))
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    names = ["top", "left", "bottom", "right", "insideH", "insideV"] if inside else ["top", "left", "bottom", "right"]
    for name in names:
        edge = borders.find(qn(f"w:{name}"))
        if edge is None:
            edge = OxmlElement(f"w:{name}")
            borders.append(edge)
        edge.set(qn("w:val"), "single")
        edge.set(qn("w:sz"), size)
        edge.set(qn("w:space"), "0")
        edge.set(qn("w:color"), color)


def set_table_geometry(table, widths_dxa: list[int], *, indent_dxa: int = TABLE_INDENT_DXA) -> None:
    if sum(widths_dxa) != PAGE_WIDTH_DXA:
        raise ValueError(f"table widths must sum to {PAGE_WIDTH_DXA}, got {sum(widths_dxa)}")
    table.autofit = False
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    tbl = table._tbl
    tbl_pr = tbl.tblPr

    tbl_w = tbl_pr.find(qn("w:tblW"))
    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW")
        tbl_pr.append(tbl_w)
    tbl_w.set(qn("w:w"), str(PAGE_WIDTH_DXA))
    tbl_w.set(qn("w:type"), "dxa")

    tbl_ind = tbl_pr.find(qn("w:tblInd"))
    if tbl_ind is None:
        tbl_ind = OxmlElement("w:tblInd")
        tbl_pr.append(tbl_ind)
    tbl_ind.set(qn("w:w"), str(indent_dxa))
    tbl_ind.set(qn("w:type"), "dxa")

    layout = tbl_pr.find(qn("w:tblLayout"))
    if layout is None:
        layout = OxmlElement("w:tblLayout")
        tbl_pr.append(layout)
    layout.set(qn("w:type"), "fixed")

    old_grid = tbl.find(qn("w:tblGrid"))
    if old_grid is not None:
        tbl.remove(old_grid)
    grid = OxmlElement("w:tblGrid")
    for width in widths_dxa:
        col = OxmlElement("w:gridCol")
        col.set(qn("w:w"), str(width))
        grid.append(col)
    tbl.insert(1, grid)

    for row in table.rows:
        for cell, width in zip(row.cells, widths_dxa):
            set_cell_width(cell, width)
            set_cell_margins(cell, **CELL_MARGINS)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def repeat_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    header = OxmlElement("w:tblHeader")
    header.set(qn("w:val"), "true")
    tr_pr.append(header)


def set_cell_text(cell, text: str, *, size: float = 9.5, color: str = INK, bold: bool = False,
                  align: WD_ALIGN_PARAGRAPH = WD_ALIGN_PARAGRAPH.LEFT, style: str | None = None) -> None:
    cell.text = ""
    p = cell.paragraphs[0]
    if style:
        p.style = style
    p.alignment = align
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.line_spacing = 1.08
    for index, part in enumerate(str(text).split("\n")):
        if index:
            p.add_run().add_break()
        run = p.add_run(part)
        set_run_font(run, size=size, color=color, bold=bold)


def add_cell_paragraph(cell, parts: Iterable[tuple[str, dict]], *, size: float = 9.5,
                       align: WD_ALIGN_PARAGRAPH = WD_ALIGN_PARAGRAPH.LEFT) -> None:
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = align
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.line_spacing = 1.08
    for text, options in parts:
        run = p.add_run(text)
        set_run_font(run, size=options.pop("size", size), **options)


def mark_header(row, fill: str = LIGHT_BLUE) -> None:
    repeat_header(row)
    for cell in row.cells:
        set_cell_shading(cell, fill)
        for paragraph in cell.paragraphs:
            paragraph.paragraph_format.space_after = Pt(0)
            for run in paragraph.runs:
                set_run_font(run, size=9.3, color=NAVY, bold=True)


def add_page_number(paragraph) -> None:
    run = paragraph.add_run()
    fld_char1 = OxmlElement("w:fldChar")
    fld_char1.set(qn("w:fldCharType"), "begin")
    instr_text = OxmlElement("w:instrText")
    instr_text.set(qn("xml:space"), "preserve")
    instr_text.text = " PAGE "
    fld_char2 = OxmlElement("w:fldChar")
    fld_char2.set(qn("w:fldCharType"), "end")
    run._r.append(fld_char1)
    run._r.append(instr_text)
    run._r.append(fld_char2)
    set_run_font(run, size=9, color=MUTED)


def add_hyperlink(paragraph, text: str, url: str, *, size: float = 9.5) -> None:
    part = paragraph.part
    relationship_id = part.relate_to(url, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink", is_external=True)
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), relationship_id)
    new_run = OxmlElement("w:r")
    rpr = OxmlElement("w:rPr")
    color = OxmlElement("w:color")
    color.set(qn("w:val"), BLUE)
    rpr.append(color)
    underline = OxmlElement("w:u")
    underline.set(qn("w:val"), "single")
    rpr.append(underline)
    rfonts = OxmlElement("w:rFonts")
    rfonts.set(qn("w:ascii"), "Calibri")
    rfonts.set(qn("w:hAnsi"), "Calibri")
    rfonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    rpr.append(rfonts)
    sz = OxmlElement("w:sz")
    sz.set(qn("w:val"), str(int(size * 2)))
    rpr.append(sz)
    new_run.append(rpr)
    text_node = OxmlElement("w:t")
    text_node.text = text
    new_run.append(text_node)
    hyperlink.append(new_run)
    paragraph._p.append(hyperlink)


def add_field(paragraph, instruction: str, *, size: float = 9, color: str = MUTED) -> None:
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = f" {instruction} "
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.append(begin)
    run._r.append(instr)
    run._r.append(separate)
    run._r.append(end)
    set_run_font(run, size=size, color=color)


def configure_styles(doc: Document) -> None:
    styles = doc.styles
    normal = styles["Normal"]
    set_style_font(normal, size=11, color=INK)
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.10

    title = styles["Title"]
    set_style_font(title, size=30, color=NAVY, bold=True)
    title.paragraph_format.space_before = Pt(0)
    title.paragraph_format.space_after = Pt(6)
    title.paragraph_format.line_spacing = 1.0

    subtitle = styles["Subtitle"]
    set_style_font(subtitle, size=15, color=MUTED)
    subtitle.paragraph_format.space_before = Pt(0)
    subtitle.paragraph_format.space_after = Pt(16)
    subtitle.paragraph_format.line_spacing = 1.1

    for name, size, color, before, after in (
        ("Heading 1", 16, BLUE, 16, 8),
        ("Heading 2", 13, BLUE, 12, 6),
        ("Heading 3", 12, DARK_BLUE, 8, 4),
    ):
        style = styles[name]
        set_style_font(style, size=size, color=color, bold=True)
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.line_spacing = 1.05
        style.paragraph_format.keep_with_next = True

    for name in ("List Bullet", "List Number"):
        style = styles[name]
        set_style_font(style, size=10.8, color=INK)
        style.paragraph_format.left_indent = Inches(0.5)
        style.paragraph_format.first_line_indent = Inches(-0.25)
        style.paragraph_format.space_before = Pt(0)
        style.paragraph_format.space_after = Pt(6)
        style.paragraph_format.line_spacing = 1.167

    if "Kicker" not in styles:
        style = styles.add_style("Kicker", 1)
    else:
        style = styles["Kicker"]
    set_style_font(style, size=9.5, color=TEAL, bold=True)
    style.paragraph_format.space_before = Pt(0)
    style.paragraph_format.space_after = Pt(4)
    style.paragraph_format.line_spacing = 1.0

    if "Small Note" not in styles:
        style = styles.add_style("Small Note", 1)
    else:
        style = styles["Small Note"]
    set_style_font(style, size=9.2, color=MUTED)
    style.paragraph_format.space_before = Pt(0)
    style.paragraph_format.space_after = Pt(5)
    style.paragraph_format.line_spacing = 1.1


def configure_page(doc: Document) -> None:
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)
    section.header_distance = Inches(0.492)
    section.footer_distance = Inches(0.492)

    header = section.header
    header.is_linked_to_previous = False
    hp = header.paragraphs[0]
    hp.alignment = WD_ALIGN_PARAGRAPH.LEFT
    hp.paragraph_format.space_after = Pt(0)
    r = hp.add_run("会记  ·  产品介绍")
    set_run_font(r, size=8.8, color=MUTED, bold=True)

    footer = section.footer
    fp = footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    fp.paragraph_format.space_before = Pt(0)
    fp.paragraph_format.space_after = Pt(0)
    r = fp.add_run("正式产品说明  ·  ")
    set_run_font(r, size=8.8, color=MUTED)
    add_page_number(fp)


def add_para(doc: Document, text: str = "", *, style: str = "Normal", before: float | None = None,
             after: float | None = None, line: float | None = None, color: str | None = None,
             bold: bool = False, italic: bool = False, size: float | None = None,
             align: WD_ALIGN_PARAGRAPH | None = None, keep_next: bool = False) -> object:
    p = doc.add_paragraph(style=style)
    if before is not None:
        p.paragraph_format.space_before = Pt(before)
    if after is not None:
        p.paragraph_format.space_after = Pt(after)
    if line is not None:
        p.paragraph_format.line_spacing = line
    if align is not None:
        p.alignment = align
    if keep_next:
        p.paragraph_format.keep_with_next = True
    if text:
        run = p.add_run(text)
        set_run_font(run, size=size, color=color or INK, bold=bold, italic=italic)
    return p


def add_heading(doc: Document, text: str, level: int = 1) -> object:
    p = doc.add_paragraph(style=f"Heading {level}")
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    set_run_font(run, size={1: 16, 2: 13, 3: 12}[level], color={1: BLUE, 2: BLUE, 3: DARK_BLUE}[level], bold=True)
    return p


def add_bullet(doc: Document, text: str, *, bold_lead: str | None = None) -> None:
    p = doc.add_paragraph(style="List Bullet")
    if bold_lead and text.startswith(bold_lead):
        lead = p.add_run(bold_lead)
        set_run_font(lead, size=10.8, color=INK, bold=True)
        rest = p.add_run(text[len(bold_lead):])
        set_run_font(rest, size=10.8, color=INK)
    else:
        run = p.add_run(text)
        set_run_font(run, size=10.8, color=INK)


def add_number(doc: Document, text: str, *, bold_lead: str | None = None) -> None:
    p = doc.add_paragraph(style="List Number")
    if bold_lead and text.startswith(bold_lead):
        lead = p.add_run(bold_lead)
        set_run_font(lead, size=10.8, color=INK, bold=True)
        rest = p.add_run(text[len(bold_lead):])
        set_run_font(rest, size=10.8, color=INK)
    else:
        run = p.add_run(text)
        set_run_font(run, size=10.8, color=INK)


def add_callout(doc: Document, label: str, text: str, *, fill: str = LIGHT_TEAL, accent: str = TEAL,
                label_size: float = 10.2, text_size: float = 10.6) -> None:
    p = doc.add_paragraph(style="Normal")
    p.paragraph_format.left_indent = Inches(0.10)
    p.paragraph_format.right_indent = Inches(0.10)
    p.paragraph_format.space_before = Pt(5)
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.line_spacing = 1.12
    set_paragraph_shading(p, fill)
    set_paragraph_borders(p, color=accent, size="7")
    lead = p.add_run(label + "  ")
    set_run_font(lead, size=label_size, color=accent, bold=True)
    body = p.add_run(text)
    set_run_font(body, size=text_size, color=INK)


def add_two_col_label_table(doc: Document, rows: list[tuple[str, str]], *, header: tuple[str, str] | None = None,
                            fill: str = LIGHT_BLUE, label_width: int = 2700, body_size: float = 9.7):
    widths = [label_width, PAGE_WIDTH_DXA - label_width]
    table = doc.add_table(rows=1 if header else 0, cols=2)
    set_table_geometry(table, widths)
    set_table_borders(table, color=BORDER, size="6", inside=True)
    if header:
        set_cell_text(table.rows[0].cells[0], header[0], size=9.3, color=NAVY, bold=True)
        set_cell_text(table.rows[0].cells[1], header[1], size=9.3, color=NAVY, bold=True)
        mark_header(table.rows[0], fill=fill)
    for label, detail in rows:
        cells = table.add_row().cells
        set_cell_text(cells[0], label, size=body_size, color=NAVY, bold=True)
        set_cell_text(cells[1], detail, size=body_size, color=INK)
    return table


def add_matrix_table(doc: Document, headers: list[str], rows: list[list[str]], widths: list[int], *,
                     header_fill: str = LIGHT_BLUE, body_size: float = 9.1, header_size: float = 9.0,
                     status_column: int | None = None):
    table = doc.add_table(rows=1, cols=len(headers))
    set_table_geometry(table, widths)
    set_table_borders(table, color=BORDER, size="6", inside=True)
    for cell, header in zip(table.rows[0].cells, headers):
        set_cell_text(cell, header, size=header_size, color=NAVY, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT)
    mark_header(table.rows[0], fill=header_fill)
    for row_index, row_data in enumerate(rows):
        cells = table.add_row().cells
        for index, (cell, value) in enumerate(zip(cells, row_data)):
            fill = WHITE if row_index % 2 == 0 else "FAFBFC"
            if status_column is not None and index == status_column:
                if "已确定" in value or "基线" in value or "本地" in value:
                    fill = LIGHT_TEAL
                elif "压测" in value or "目标" in value or "建议" in value:
                    fill = PALE_GOLD
                elif "边界" in value or "限制" in value:
                    fill = PALE_RED
            set_cell_shading(cell, fill)
            align = WD_ALIGN_PARAGRAPH.CENTER if index == status_column else WD_ALIGN_PARAGRAPH.LEFT
            set_cell_text(cell, value, size=body_size, color=INK, align=align, bold=(index == status_column))
    return table


def add_source(doc: Document, title: str, url: str, note: str) -> None:
    p = doc.add_paragraph(style="Small Note")
    p.paragraph_format.left_indent = Inches(0.18)
    r = p.add_run(title + "：")
    set_run_font(r, size=9.2, color=MUTED, bold=True)
    add_hyperlink(p, "查看官方页面", url, size=9.0)
    n = p.add_run("  " + note)
    set_run_font(n, size=9.2, color=MUTED)


def add_model_inventory(doc: Document) -> None:
    add_heading(doc, "模型、参数与部署清单", 2)
    add_para(doc, "以下数据以 2026-08-13 本机实际模型文件、代码常量和模型配置为准。参数量表示模型复杂度，磁盘大小表示当前量化/转换格式的本机占用，两者不能互相替代；不同平台、量化方式和缓存版本会造成文件大小差异。")
    add_matrix_table(
        doc,
        ["阶段", "实际模型/实现", "关键参数", "当前本机大小/状态"],
        [
            ["实时 ASR", "faster-whisper；mobiuslabsgmbh/faster-whisper-large-v3-turbo（large-v3-turbo，约 809M 参数）", "WhisperModel；CUDA 时 int8_float16，CPU 时 int8；transcribe beam_size=1、vad_filter=False、condition_on_previous_text=False；录音期间常驻，停止后释放", "当前 CTranslate2 快照 1,621,665,983 B（1.51 GiB）；已缓存"],
            ["会后 ASR 精修", "faster-whisper；Systran/faster-whisper-large-v3（large-v3，约 1,550M 参数）", "CUDA 时 int8_float16、CPU 时 int8；transcribe beam_size=5、vad_filter=False、condition_on_previous_text=False；停录保存后按需加载；与其他 GPU 阶段串行", "当前 CTranslate2 快照 3,090,835,702 B（2.88 GiB）；已缓存"],
            ["VAD/实时分段", "FunASR AutoModel；funasr/fsmn-vad（FSMN，428,738 参数）", "16 kHz；80 维 Mel，LFR 后输入维 400；4 层 FSMN；linear 250、projection 128；项目分段参数：预滚动 240 ms、开始 80 ms、静音结束 350 ms、partial 900 ms、单段上限 8 s", "模型权重 model.pt 1,721,366 B（1.64 MiB）；完整本机 snapshot 4,024,535 B（3.84 MiB）；ready"],
            ["语言识别", "Whisper 返回 language + language_probability 为主；文本规则与可选 Lingua 为后备", "ASR 语言置信度 ≥0.65 时优先采信；缺失/低置信度才进入中文字符、英德规则和 Lingua；正式范围 zh/en/de", "无独立语言大模型；置信度和 language_source 写入 Utterance"],
            ["说话人重排", "Resemblyzer voice encoder；本地 pretrained.pt；能量 VAD + embedding + cosine online clustering", "约 1,423,616 参数；16 kHz mono；能量帧 30 ms；最大静音间隔 250 ms；最短语音 350 ms；embedding context 1.6 s；hop 0.8 s；cluster threshold 0.68；重叠映射阈值 15%；匿名 speaker_1/...；无专门 overlap model", "17,090,379 B（16.30 MiB）；权重预检已就绪；运行时按需加载；唯一说话人实现，无授权、无运行时下载、无回退"],
            ["本地翻译", "Marian OPUS-MT → CTranslate2 int8；en→zh 77,943,296 参数；de→zh 76,363,776 参数", "两者均为 d_model=512、encoder 6 层、decoder 6 层；CUDA int8_float16 / CPU int8；beam_size=2、max_decoding_length=384、repetition_penalty=1.05；SentencePiece", "en→zh 82,552,022 B（78.73 MiB）；de→zh 80,656,414 B（76.92 MiB）；均已就绪"],
            ["纪要与 To-do", "Jimo SSE 外部服务；两个 share 节点分别承担 summary、todo", "客户端固定发送 messages、sessionId、source=api、extra={}；不在客户端注入 model/temperature；Authorization 仅服务端环境变量；总结按状态分块，To-do 读取已保存纪要", "外部模型名、temperature 由 Jimo 节点配置，不由当前代码实际注入；文档不把 README 示例当成客户端运行时事实"],
        ],
        [1500, 3000, 3300, 1560],
        body_size=8.0,
        header_size=8.4,
    )
    add_para(doc, "本机必需本地模型合计约 4.56 GiB（按当前快照和项目转换目录求和，不含重复/旧缓存、Python/CUDA 依赖和积墨外部模型）。其中 Whisper 两套 ASR 约占 4.39 GiB，是主要磁盘占用。", style="Small Note")
    add_para(doc, "当前 HF refs/main：turbo=0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf；large-v3=edaa852ec7e145841d8ffdb056a99866b5f0a478。", style="Small Note")
    add_para(doc, "部署环境（已核对）：Windows 11 x64；AMD Ryzen AI 7 H 350（8 核 16 线程）；31.12 GiB RAM；RTX 5060 Laptop GPU 8,151 MiB，驱动 610.88；CUDA 12.8 / PyTorch 2.11.0+cu128；Python 3.11.14。默认 MEETING_DEVICE=auto、单活跃会议；GPU 重任务由单进程资源锁串行执行。", style="Small Note")

    add_heading(doc, "部署资源配置建议", 2)
    add_para(doc, "下表按当前单机架构给出部署口径。“已验证基线”是本机实际跑通环境；“最低可运行”只表示能够安装和执行，不代表实时延迟达标；生产发布前仍需使用目标机器和真实中英德混合音频做 30 分钟连续压测。")
    add_matrix_table(
        doc,
        ["资源", "最低可运行（不承诺实时）", "推荐单机生产基线", "已验证本机 / 说明"],
        [
            ["CPU", "x64，至少 4 核 8 线程；CPU 模式可运行，但 large-v3 精修和本地翻译耗时可能明显增加", "8 核 16 线程或以上；GPU 推理时 CPU 仍负责 WebSocket、音频编解码、分段、文件与任务调度", "AMD Ryzen AI 7 H 350，8 核 16 线程；单活跃会议"],
            ["系统内存", "16 GB；仅适合单会议、低并发和受控测试，需避免同时加载额外模型或大批量任务", "32 GB；为两套 Whisper、Python/CUDA 运行时、音频缓存和后处理留出余量", "31.12 GiB 可用物理内存；当前单会议基线"],
            ["GPU", "可不配 GPU，使用 CPU int8；功能可运行但不承诺实时体验", "NVIDIA CUDA GPU，8 GB 显存或以上；驱动、CUDA 与 PyTorch wheel 必须匹配", "RTX 5060 Laptop GPU；CUDA 可用；单进程 GPU 资源锁串行调度"],
            ["显存", "CPU 模式为 0；若启用 CUDA，不建议低于 8 GB", "8 GB 级；实时模型停止后释放，再加载 large-v3 精修，避免两套 ASR 同驻留", "物理显存 8,151 MiB；当前代码未实现显存硬配额"],
            ["磁盘", "至少预留 15 GB 给程序、Python/CUDA 依赖和当前模型；录音/结果空间另计", "建议预留 20 GB 以上系统盘空间，并为 result/meetings 单独规划容量或对象存储", "必需模型约 4.56 GiB；当前 .venv 约 5.41 GiB；项目内 OPUS-MT 约 155.65 MiB；旧缓存会额外占用"],
            ["系统与运行时", "Windows 10/11 x64；Python 3.11；Chrome 或 Edge；FFmpeg 建议安装", "Windows 11 x64；Python 3.11；CUDA 12.8 对应 PyTorch/torchaudio；生产关闭模型自动下载", "Windows 11；Python 3.11.14；CUDA 12.8；PyTorch/torchaudio 2.11.0+cu128"],
            ["单机并发", "1 场；MEETING_MAX_ACTIVE_MEETINGS=1", "当前单进程基线 1 场。增加并发前必须压测；不能把 10 路目标理解为单张 8 GB GPU 的承诺", "默认 1 场活跃会议；实时与精修队列严格有界"],
            ["网络", "本地 ASR、VAD、翻译和说话人重排可离线；首次准备模型时需要下载", "生产预下载并校验所有模型；仅纪要/To-do 使用积墨时需要访问外部 Jimo 节点", "生产建议 ASR_AUTODOWNLOAD=0、TRANSLATION_AUTODOWNLOAD=0"],
        ],
        [1300, 2600, 2800, 2660],
        body_size=8.2,
        header_size=8.5,
    )
    add_callout(doc, "容量边界", "企业第一阶段可以把 10 路同时会议作为压测目标，但当前单机产品基线仍是 1 路。多路部署应拆分为多个 GPU worker，并使用目标 GPU 测量首结果延迟、P50/P95、RTF、峰值显存、队列深度和恢复时间后再确定安全流数。", fill=PALE_GOLD, accent=GOLD)


def build_document() -> None:
    doc = Document()
    configure_styles(doc)
    configure_page(doc)
    doc.core_properties.title = "会记：实时会议记录、翻译与行动闭环产品介绍"
    doc.core_properties.subject = "产品经理视角的产品、竞品、技术栈、容量与 token 口径说明"
    doc.core_properties.author = ""
    doc.core_properties.comments = "产品介绍文档。"

    # First-page customer-pack style title block.
    p = doc.add_paragraph(style="Kicker")
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(5)
    r = p.add_run("产品介绍  ·  正式产品说明")
    set_run_font(r, size=9.5, color=TEAL, bold=True)

    p = doc.add_paragraph(style="Title")
    r = p.add_run("会记")
    set_run_font(r, size=30, color=NAVY, bold=True)

    p = doc.add_paragraph(style="Subtitle")
    r = p.add_run("实时会议记录、翻译与行动闭环")
    set_run_font(r, size=15, color=MUTED)

    add_para(
        doc,
        "让会议从“有人记录”变为“可实时理解、可复盘、可执行”。",
        after=14,
        color=INK,
        bold=True,
        size=12,
    )
    add_two_col_label_table(
        doc,
        [
            ("产品形态", "Windows 本地优先的实时会议工作台"),
            ("正式支持", "中文、英文、德文；英文/德文实时或会后翻译为简体中文"),
            ("处理边界", "音频、ASR、翻译、说话人分离默认本地；积墨负责纪要与 To-do-list"),
            ("本文口径", "产品规格口径；容量、质量和成本指标区分已配置参数、稳定运行范围和压测指标"),
        ],
        header=("项目", "产品定义"),
        fill=LIGHT_BLUE,
        label_width=2050,
        body_size=9.8,
    )

    add_callout(
        doc,
        "一句话定位",
        "一套本地优先的实时会议系统：录音中提供低延迟多语言转写；停录后自动完成高精度 ASR 精修、说话人重排和最终翻译，完成后由用户点击按钮生成纪要，并自动生成 To-do-list。录音保存与后台后处理相互独立，任何后处理失败都不影响已保存的音频和快速转写。",
        fill=LIGHT_TEAL,
        accent=TEAL,
    )

    add_heading(doc, "产品一览", 1)
    add_matrix_table(
        doc,
        ["实时理解", "会后可信", "本地优先", "可恢复"],
        [[
            "large-v3-turbo + WebSocket 持续输出局部结果",
            "large-v3 精修 + Resemblyzer 说话人重排",
            "ASR、翻译、Resemblyzer 说话人重排默认在本机运行",
            "阶段检查点、修订事件、失败重试与重启恢复",
        ]],
        [2340, 2340, 2340, 2340],
        header_fill=LIGHT_GRAY,
        body_size=9.5,
        header_size=9.4,
    )

    add_heading(doc, "1. 产品定位与目标用户", 1)
    add_para(
        doc,
        "会记面向需要“边开会边理解、会后快速复盘并形成执行清单”的团队。产品不是单纯的录音机或字幕组件，而是一条从音频采集到行动闭环的本地化处理链：实时阶段优先保证可用和低延迟，会后阶段再用更高质量的模型补齐准确率、说话人和结构化结果。",
    )
    add_heading(doc, "典型场景", 2)
    for text in [
        "跨语言项目会议：中文、英文、德文混合讨论，实时查看原文，英文/德文同步补充中文译文。",
        "研发评审与项目例会：保留决策、条件、风险、阻塞和原文时间范围，减少会后手工整理。",
        "访谈、销售与客户沟通：按发言人和时间轴复盘，输出会议纪要和可追踪的 To-do-list。",
        "对隐私敏感的本地会议：音频和模型推理留在部署侧，仅在配置积墨后把纪要所需文本发送给外部服务。",
    ]:
        add_bullet(doc, text)

    add_heading(doc, "产品承诺", 2)
    add_two_col_label_table(
        doc,
        [
            ("实时可见", "不需要等整场会议结束才看到结果；通过 WebSocket 持续更新转写、语言和翻译状态。"),
            ("保存优先", "停录后先完成音频、快速转写和 manifest 落盘，再异步进行 ASR 精修、说话人重排和最终翻译；三项完成后才开放“生成纪要和 To-do-list”按钮。"),
            ("结果可追溯", "TranscriptStore、transcript_events.jsonl、修订号、模型元数据和 speaker_segments.json 共同支撑结果审计与前端原地更新。"),
            ("故障可恢复", "后处理按阶段写入检查点；应用重启会把 running 任务重新排队，从最近完成阶段继续。"),
        ],
        header=("用户价值", "产品表现"),
        label_width=2050,
        body_size=9.7,
    )

    add_heading(doc, "2. 核心特点与特色能力", 1)
    add_heading(doc, "2.1 实时转写与翻译：先让会议可理解", 2)
    add_para(doc, "实时链路统一使用 large-v3-turbo，降低模型切换和隐式回退带来的不确定性。浏览器通过 AudioWorklet 采集 16 kHz、单声道 PCM16 音频，FastAPI WebSocket 接收并将 partial/final 结果推送到前端。中文原文直接显示，英文和德文进入本地 OPUS-MT 翻译队列后批量补充简体中文。")
    add_callout(doc, "体验重点", "实时链路的目标不是在每个瞬间给出最终答案，而是持续给出可用的中间结果；最终质量由停录后的精修和说话人重排完成。", fill=PALE_GOLD, accent=GOLD)

    add_heading(doc, "2.2 录音已保存与后台后处理彻底解耦", 2)
    add_para(doc, "这是产品体验的关键设计。录音状态 recording_state 只表示录音和快速结果是否已经落盘；postprocess 独立展示 ASR 精修、说话人重排、翻译、纪要和 To-do-list 的阶段状态；前三项自动执行，纪要由用户按钮触发，成功后自动生成 To-do-list。用户可以在录音已保存后立即查看和下载，后台任务即使排队、失败或需要重试，也不会让会议停留在无限期的“正在保存”。")

    add_heading(doc, "2.3 会后精修与说话人重排", 2)
    add_para(doc, "停录后自动按固定顺序执行：large-v3 ASR 精修 → Resemblyzer 声纹嵌入、能量 VAD 与余弦相似度聚类 → 将说话人时间段与转写句子按重叠比例对齐 → 使用最终文本批量翻译。自动阶段全部完成后，用户点击按钮生成积墨纪要；纪要原子保存成功后自动生成 To-do-list。实时阶段仅显示匿名“演讲人 1/2/…”，会后按照首次出现顺序稳定编号并原地更新。暂不做实名声纹录入和参与者身份绑定。")

    add_heading(doc, "2.4 版本化结果，而不是覆盖式写文件", 2)
    add_para(doc, "精修可能改变句子数量、边界和说话人标签。系统通过稳定的 source_segment_id、revision、删除/替换事件和兼容旧字段的 Utterance 结构，避免旧句子残留在前端；历史会议仍可读取 transcript.json、transcript.jsonl、meeting_transcript.md、translated_zh.md 和 manifest.json。")

    add_heading(doc, "2.5 模型状态透明，失败不伪装", 2)
    add_para(doc, "启动预检会检查 ASR、VAD、OPUS-MT 和 Resemblyzer 包及本地 pretrained.pt 权重。必需能力未准备好时，健康状态保持 loading/error 并禁止新建会议。翻译状态明确区分 pending、ready、not_needed、unsupported、failed；ASR 失败记录模型信息、错误和重试，不把错误结果静默伪装成成功。")

    add_heading(doc, "3. 从一次会议到行动闭环", 1)
    add_para(doc, "产品流程围绕“先保存、后完善”设计：")
    for text in [
        "会前预检：确认 large-v3-turbo、large-v3、OPUS-MT、VAD 和 Resemblyzer 权重已准备；未就绪时阻止新建会议。",
        "录音中：采集音频并按 30 分钟滚动切片保存；实时 ASR 输出转写，在线聚类提供匿名演讲人，翻译按批次异步补充。",
        "停录瞬间：设置 recording_state=finalizing，刷新音频和快速 ASR 队列，写入音频清单与 manifest，随后立即置为 recording_state=complete 并广播 recording_complete。",
        "后台后处理：postprocess 自动执行精修 ASR、Resemblyzer 说话人重排、对齐和最终翻译；全部完成后进入 ready_for_summary。用户点击按钮触发会议纪要，纪要成功后自动触发 To-do-list；每一阶段写入持久化检查点。",
        "结果交付：前端接收 postprocess_update 和单调递增的 snapshot_revision，原地更新原文、译文、说话人和下载文件；失败时展示具体阶段和重试入口。",
    ]:
        add_number(doc, text)

    add_heading(doc, "状态模型", 2)
    add_matrix_table(
        doc,
        ["状态对象", "状态值", "用户理解"],
        [
            ["recording_state", "starting / recording / finalizing / complete / error", "录音与快速结果是否已经保存"],
            ["postprocess.state", "queued / running / ready_for_summary / complete / partial / error", "后台精修、重排、翻译和生成任务的总体状态"],
            ["postprocess.current_stage", "asr_refine / diarization(Resemblyzer) / translation / summary / todo", "当前正在处理的阶段"],
            ["snapshot_revision", "单调递增整数", "前端拒绝旧 snapshot，避免旧消息覆盖新状态"],
        ],
        [1900, 3060, 4400],
        body_size=9.3,
    )

    add_heading(doc, "4. 产品优势与适用边界", 1)
    add_heading(doc, "相比“只做实时字幕”，优势在会后可信", 2)
    for text in [
        "实时与最终结果分层：用 turbo 换取实时性，用 large-v3 换取会后精度，用 Resemblyzer 完成无外部授权的匿名说话人重排。",
        "结果不是一次性覆盖：修订事件让前端和导出文件知道“什么被改了”，适合审阅、回溯和后续质量评估。",
        "失败有边界：模型缺失、翻译不支持、积墨超时和后处理异常都有明确状态；录音文件和快速转写仍然可用。",
        "本地优先降低数据外发和 API 依赖：核心音频理解链无需把原始音频上传到第三方，也不会为本地 ASR、翻译和 Resemblyzer 说话人重排消耗远程 token。",
    ]:
        add_bullet(doc, text)

    add_heading(doc, "产品边界", 2)
    add_callout(doc, "需要如实说明", "正式支持三种语言，翻译方向固定为 en→zh、de→zh；实时阶段的说话人是匿名编号而非实名身份；默认部署以单活跃会议、单 GPU worker 为稳定基线。更大规模并发、多路音频输入、更多语言和实名声纹属于独立扩展能力。", fill=PALE_RED, accent=RED)

    doc.add_page_break()
    add_heading(doc, "5. 与竞品的区别", 1)
    add_para(doc, "竞品对比以中国国内主流办公与会议产品为主：钉钉、企业微信、腾讯会议和飞书。它们优势在会议平台、组织通讯录、协作和云端服务；会记的差异在于本地优先的音频理解链、可控模型和会后结果修订。成本比较优先采用私有化、专属部署或混合云公开案例，不再拿普通 SaaS 月费直接横向比较。以下仅比较公开产品能力与公开采购信息，具体功能和价格受版本、套餐、地区、企业规模、定制范围和合同条款影响。")

    add_matrix_table(
        doc,
        ["比较维度", "会记", "国内会议/协作平台", "主要差异"],
        [
            ["产品形态", "独立的本地会议理解工作台：采集、转写、翻译、说话人、纪要和待办", "钉钉/企业微信/飞书：办公协作平台；腾讯会议：会议平台与会议资产", "竞品强在组织协作与会议生态；会记专注音频理解链和结果可追溯"],
            ["数据与部署", "ASR、翻译、Resemblyzer 说话人重排默认本地；仅将纪要/待办文本发送到积墨", "以云端会议、云录制、云端 AI 和组织权限为主；部分企业版支持更强的管理/混合云能力", "会记适合内网/敏感会议；竞品适合快速开通、统一管理和跨端协作"],
            ["实时转写/翻译", "实时中文/英文/德文转写；英文、德文本地译为简体中文", "腾讯会议支持实时字幕/转写与中英互译，企业版本语言更广；钉钉会议支持实时字幕与多语言翻译；飞书支持实时转写/翻译；企业微信持续完善字幕、转写和同传", "会记语言范围更聚焦，但推理链与数据边界可控"],
            ["纪要与待办", "停录后自动完成“精修→说话人重排→翻译”，用户按钮触发积墨纪要，成功后自动生成 To-do；阶段可见、可重试", "普遍提供智能纪要、章节、重点、发言人和待办，并可与文档/任务/群聊联动", "竞品协作分发更强；会记更强调原文、修订和阶段检查点"],
            ["说话人能力", "实时匿名编号；会后 Resemblyzer 重排；支持重叠片段元数据", "腾讯会议/飞书等支持发言人视图或声纹/身份关联；企业微信也提供转写/纪要能力", "会记默认不做实名声纹绑定，减少隐私和身份误识别风险"],
            ["集成与组织", "REST/WebSocket/API 文件下载；当前以单机/内网为主", "钉钉、企业微信、飞书接入组织通讯录、日历、文档、任务和消息；腾讯会议提供开放平台 API", "会记需通过企业版扩展层补齐 SSO、对象存储、队列和协作集成"],
            ["成本结构与使用成本", "本地 ASR、翻译和 Resemblyzer 说话人重排不产生远程 token；积墨按调用计费：0.03 元/会话，2 次会话/API 请求，即 0.06 元/请求。以 N 个转写块估算，无重试约 (N+2)×0.06 元，另计 GPU/服务器、存储、实施和运维", "私有化/专属版通常按项目报价，公开案例从约 25 万元/年到百万元级，可能包含软件授权、部署、定制、存储、运维、组织协同和容灾；腾讯会议完整混合云方案需商务询价", "会记应与竞品三年 TCO 比较：会记外部调用成本透明，但需要承担本地基础设施；竞品交付范围更大、采购金额更高，不能把单个合同清单项直接理解成完整私有化价格"],
        ],
        [1450, 2850, 2500, 2560],
        header_fill=LIGHT_BLUE,
        body_size=8.8,
        header_size=8.9,
    )
    add_callout(doc, "产品判断", "会记最适合作为“本地会议理解层”或企业内网会议记录服务；它不试图替代钉钉、企业微信、腾讯会议和飞书的会议与组织协作平台能力，而是补足敏感会议场景中的本地音频理解、会后精修和结果可追溯。", fill=LIGHT_TEAL, accent=TEAL)

    add_heading(doc, "私有化/专属部署成本参考", 2)
    add_para(doc, "由于会记是本地部署产品，真正可比的不是竞品公开 SaaS 月费，而是私有化、专属部署或混合云方案的三年总拥有成本（TCO）。下表整理公开采购预算、中标结果、项目投资清单和官方部署说明。金额可能包含授权、部署、定制、存储、运维、组织协同、硬件适配或容灾，不等于单纯软件许可；因此只作为报价锚点，不作为统一官方价。")
    add_matrix_table(
        doc,
        ["竞品", "公开私有化/专属部署案例", "金额口径", "对会记的参考"],
        [
            ["钉钉", "浙江工业大学“2026工大专属版钉钉”：预算 25.0 万元/年，最终中标 24.9 万元；重庆市公开清单另列中国烟草重庆公司“专属钉服务”投资 145 万元，含基础底座、专属安全、专属存储、专属打包和运营运维。", "约 25 万/年；大型专属项目 145 万", "小规模专属版可用约 25 万元/年作低位锚点；含平台底座、安全、存储和运维的项目应按百万元级另估。"],
            ["企业微信", "广东省机场管理集团：2 万用户、1 年私有化授权，含客户端定制打包和平台运维服务；项目限价 198 万元。", "198 万/年限价；约 99 元/用户/年（含定制运维）", "说明大型组织私有化授权与定制运维的采购上限；不能把 99 元/用户/年当作单纯软件许可单价。"],
            ["腾讯会议", "官方支持企业混合云/专网会议，媒体业务可部署在客户私有环境，并支持信创、虚拟化、物理机和私有云；未公开统一私有化报价。", "商务询价", "报价时应单列本地媒体节点、专网/公网双轨、云上热备、SDK/API、硬件兼容、容灾和运维；标准版月费不作私有化报价。"],
            ["飞书", "上海期货交易所 2026—2029 年数字化协同平台成交 1047.7146 万元；清单单独列“飞书企业旗舰版_数据私有化”1680 元、“飞书 AI 企业版”99000 元。", "全平台 1047.7 万/3 年；私有化行项 1680 元", "完整平台合同不能拆成会记可比的本地部署价格；1680 元只是合同清单单项，不能理解为完整私有化授权。"],
        ],
        [1300, 3600, 1650, 2810],
        header_fill=LIGHT_BLUE,
        body_size=8.55,
        header_size=8.8,
    )
    add_para(doc, "本表价格来源链接：", style="Small Note")
    add_source(doc, "钉钉专属版采购预算（浙江工业大学）", "https://www.ccgp.gov.cn/cggg/dfgg/jzxcs/202511/t20251125_25766613.htm", "公开预算 25 万元/年，履约期 1 年。")
    add_source(doc, "钉钉专属版中标结果（浙江工业大学）", "https://www.ccgp.gov.cn/cggg/dfgg/zbgg/202512/t20251210_25900892.htm", "公开中标价 24.9 万元。")
    add_source(doc, "专属钉服务投资清单（重庆市江北区）", "https://dsjj.cq.gov.cn/ztzl/cjkf/jfqd/2025/202503/P020250304333616288227.pdf", "公开投资金额 145 万元，含基础底座、专属安全、专属存储、专属打包和运营运维。")
    add_source(doc, "企业微信私有化授权采购公告（广东省机场集团）", "https://www.ccaonline.cn/mhzbgg/972792.html", "公开项目限价 198 万元，范围为 2 万用户 1 年授权、定制打包和平台运维。")
    add_source(doc, "腾讯会议官方混合云部署说明", "https://meeting.tencent.com/news/neiwanghuiyi.html", "用于官方支持混合云、专网会议和媒体下沉的能力依据；官方未公开统一私有化报价。")
    add_source(doc, "飞书平台采购合同公告（上海期货交易所）", "https://www.shfe.com.cn/publicnotice/collection/202604/P020260429502363045947.pdf", "公开 2026—2029 年全平台成交金额及数据私有化、AI 企业版的合同清单金额。")
    add_callout(doc, "成本判断", "会记不应与竞品 SaaS 月费比较，而应与竞品私有化/专属部署的三年 TCO 比较。公开案例显示，国内协同平台的专属/私有化项目可能从约 25 万元/年到百万元级，往往包含授权、部署、定制、存储、运维和组织协同；腾讯会议完整方案需商务询价。会记则把本地音频理解能力单独交付，外部调用成本可按积墨精确拆分，但需要单独计入 GPU/服务器、存储、实施、运维和积墨调用。以 15,000 字符、约 3 个转写块、无重试为例，积墨费用约 0.30 元。", fill=PALE_GOLD, accent=GOLD)

    add_heading(doc, "竞品优势与我们的取舍", 2)
    add_two_col_label_table(
        doc,
        [
            ("竞品强项", "组织通讯录、日历、会议室、文档/任务/群聊联动、跨端体验和云端运维；腾讯会议还具备原生会议上下文。"),
            ("会记选择", "先把本地音频理解、三语闭环、后处理可靠性和结果版本化做深，再通过 API、对象存储、队列和 SSO 逐步扩展企业能力。"),
            ("不做虚假承诺", "不把模型卡的 99 语言能力写成产品正式支持语言，不把架构上的可扩展性写成当前单机的并发承诺。"),
        ],
        header=("视角", "说明"),
        label_width=1800,
        body_size=9.5,
    )

    doc.add_page_break()
    add_heading(doc, "6. 技术架构概览", 1)
    add_callout(doc, "数据流", "浏览器音频 → FastAPI REST/WebSocket → 有界队列与 GPU 资源锁 → 实时 ASR/精修 ASR/OPUS-MT/Resemblyzer 引擎 → TranscriptStore 与 manifest → 积墨纪要/To-do → 前端增量更新与文件下载", fill=LIGHT_BLUE, accent=NAVY)

    add_matrix_table(
        doc,
        ["层级", "核心组件", "职责"],
        [
            ["采集与交互", "Chrome/Edge、AudioWorklet、PCM16 16 kHz mono、WebSocket", "采集麦克风、发送有序音频包、接收 partial/final、断线自动重连"],
            ["服务层", "Python 3.11、FastAPI、Starlette、Uvicorn", "REST API、WebSocket gateway、健康检查、metrics、静态前端和鉴权票据"],
            ["实时推理", "RealtimeAsrEngine；large-v3-turbo；VAD/分段；在线 speaker clustering", "以低延迟持续输出原文和匿名演讲人"],
            ["会后推理", "RefinementAsrEngine；large-v3；DiarizationEngine；Resemblyzer voice encoder", "精修文本、匿名说话人分离、时间段对齐和修订事件"],
            ["本地翻译", "TranslationEngine；OPUS-MT en→zh/de→zh；CTranslate2；SentencePiece", "批量翻译、模型预检、缓存、状态区分和失败可见"],
            ["资源调度", "GpuResourceManager；asyncio.Queue；checkpoint", "实时 ASR、精修 ASR、翻译和 Resemblyzer 共用 GPU 锁，严格串行，避免 8 GB 显存下的 OOM"],
            ["结果存储", "TranscriptStore；JSON/JSONL；transcript_events；speaker_segments；manifest", "兼容旧文件，保存修订、模型、阶段状态和完整音频清单"],
            ["外部智能生成", "积墨 SSE；summary 与 todo 两个节点", "基于最终文本生成中文会议纪要和结构化 To-do-list；支持重试与状态恢复"],
        ],
        [1700, 3200, 4460],
        body_size=9.1,
    )

    add_heading(doc, "7. 技术栈选择：为什么这样选", 1)
    add_matrix_table(
        doc,
        ["技术决策", "选择理由", "相对其他方案的取舍"],
        [
            ["Python 3.11", "ASR、CTranslate2、Resemblyzer、音频与数据处理生态完整；asyncio 适合 IO、队列和 WebSocket。", "Node/Go 更适合纯网关，但模型编排和音频科学计算需要跨进程/跨语言；当前阶段 Python 减少集成面。"],
            ["FastAPI + WebSocket", "同一服务承载 REST、WebSocket、健康检查和静态前端；异步接口天然适配实时事件。", "单纯 REST 轮询会增加状态延迟和请求量；独立消息网关虽可扩展，但当前产品的单机部署复杂度更高。"],
            ["faster-whisper / CTranslate2", "Whisper 生态成熟；CTranslate2 支持 GPU/CPU、量化和批量推理，适合本地部署。", "相比远程 ASR API，减少音频外发、网络抖动和按量 token；相比单一超大模型，便于实时/精修两阶段调度。"],
            ["large-v3-turbo + large-v3", "turbo 的解码层更少，适合实时；large-v3 只在停录后精修，兼顾延迟、准确度和 8 GB 显存。", "全程 large-v3 质量潜力更高但实时和显存压力更大；全程 turbo 体验快但会后质量上限较低。"],
            ["OPUS-MT + CTranslate2", "翻译方向固定、可本地运行、可批处理与缓存，远程 token 成本为零。", "LLM 翻译更灵活但成本、延迟和输出一致性更难控制；OPUS-MT 的产品范围聚焦 en/de→zh。"],
            ["Resemblyzer voice encoder", "本地声纹嵌入模型，配合能量 VAD 和在线余弦聚类完成匿名说话人分段；不需要外部授权或运行时下载。", "不提供专门的重叠语音检测模型；重叠片段按时间对齐保留 speaker_ids 和 speaker_overlap 元数据，质量边界需要 DER/JER 语料评测。"],
            ["版本化 TranscriptStore", "精修会改变句子边界；事件、修订号和 source_segment_id 能让前端正确删除/替换旧结果。", "直接多线程覆盖 transcript.jsonl 实现快但会产生竞态、旧句残留和难以审计的问题。"],
            ["单 GPU 锁 + 有界队列", "优先保证稳定、可预测和可恢复；队列满时可以明确拒绝或排队。", "并行加载会提高吞吐，但在 8 GB 显存基线下更易 OOM；企业版再按 GPU worker 横向扩展。"],
            ["积墨 SSE", "保留现有会议纪要和 To-do 能力，流式输出体验较好；本地端不再绑定一套通用 LLM。", "带来外部服务依赖和按会话计费，因此需要在 metrics 中记录 API 请求数、会话次数、费用、耗时、失败和重试。"],
        ],
        [2100, 3600, 3660],
        body_size=8.9,
        header_size=9.0,
    )

    doc.add_page_break()
    add_heading(doc, "8. 容量、并发与性能口径", 1)
    add_callout(doc, "阅读说明", "本节把“当前可写入产品规格的事实”和“必须基于 benchmark 才能承诺的数值”分开。模型卡、配置项和架构设计可以说明能力边界，但不能替代目标 GPU、混合语言语料和 30 分钟连续录音下的压测。", fill=PALE_GOLD, accent=GOLD)

    add_matrix_table(
        doc,
        ["指标", "当前产品口径", "状态", "验收/说明"],
        [
            ["正式支持语言", "3 种：中文、英文、德文", "产品规格", "中文原文；英文/德文翻译为简体中文；不把 Whisper 99 语言模型卡当成产品正式支持范围。"],
            ["输入音频", "16 kHz、单声道、PCM16；浏览器 AudioWorklet；音频包上限 256 KiB", "已配置", "服务端按包大小和顺序校验；默认本地保存 FLAC（无 ffmpeg 时 WAV）。"],
            ["分段与实时输出", "预滚动 240 ms；开始说话判定 80 ms；静音结束 350 ms；partial 间隔 900 ms；单段上限 8 s", "已配置", "这些是分段/推送参数，不等于端到端 P95 延迟；需以 benchmark 结果发布延迟指标。"],
            ["声音重叠", "在线阶段：重叠片段只保留主说话人，不做实时双人并行转写；会后阶段：保留 speaker_ids，单个句子最多记录 10 个重叠说话人；speaker_overlap 上限 1.0，speaker_confidence 为 0–1", "产品限制", "Resemblyzer 本身不提供专门重叠检测；按时间重叠比例 ≥15% 的说话人进入 speaker_ids。多人同时说话越多，WER/说话人错误率越可能下降。10 人是存储/接口保护上限，不是识别质量保证。"],
            ["匿名演讲人数量", "实时匿名编号建议上限 10 人；会后 Resemblyzer 重排单场会议建议上限 20 人；接口/前端最多展示 32 个匿名演讲人", "产品规格", "超过展示上限时按“其他演讲人”聚合或转入人工校正；匿名编号不等于实名身份，首次出现顺序编号。"],
            ["单场会议时长", "建议连续录音 ≤4 小时；音频每 30 分钟滚动切片；后处理时长随音频长度近似线性增长", "运营建议", "超过 4 小时建议拆分会议或按阶段停录，避免单任务积压。"],
            ["单实例活跃会议", "默认 1 场；MEETING_MAX_ACTIVE_MEETINGS=1", "部署基线", "单 GPU 本地部署。增加到多场前需按 GPU worker 横向扩展并重新压测。"],
            ["GPU 资源锁", "实时 ASR、large-v3、翻译和 Resemblyzer 共用 1 把进程内 GPU 锁，严格串行", "已实现", "避免 8 GB 显存同时执行重推理；当前不提供可配置 worker pool 或显存硬配额。"],
            ["有界队列", "实时推理 64；精修 16", "已配置", "两条音频处理队列有界；企业任务队列属于扩展能力。"],
            ["多人并发", "当前稳定基线：1 路活跃会议；企业压测目标：10 路同时会议", "基线/目标", "10 路是压测目标，不是单 GPU 承诺；需要目标 GPU、中英德混合语料和 30 分钟连续录音验收。"],
            ["断线恢复", "WebSocket 断线恢复窗口 15 s；stream ticket TTL 60 s；WebSocket 认证超时 5 s", "已配置", "窗口内恢复不会结束会议；前端通过 snapshot_revision 丢弃旧消息。"],
            ["音频保留", "完整会议结果默认保留 30 天；MEETING_KEEP_AUDIO=0 时，音频在精修和说话人重排成功后删除；失败时保留以便重试", "已配置", "retention_days=0 可关闭自动过期删除；受磁盘容量、企业合规和删除策略影响。"],
            ["质量指标", "WER、CER、语言识别准确率、BLEU、chrF、DER、JER、首结果延迟、分段延迟、RTF、P50/P95、峰值显存", "验收指标", "需要固定语料和 benchmark 才能给出具体数值。"],
            ["GPU 基线", "RTX 5060 Laptop 8 GB / 32 GB RAM；重任务串行执行", "部署基线", "当前未实施显存硬配额；更换 GPU 需重选 CUDA/PyTorch wheel 并重新压测。"],
        ],
        [1850, 2750, 1350, 3410],
        body_size=8.7,
        header_size=8.9,
        status_column=2,
    )

    add_model_inventory(doc)

    add_heading(doc, "9. 积墨调用成本与 Token 口径", 1)
    add_para(doc, "会记的核心语音理解链是本地模型，因此 token 不是所有阶段的成本单位；积墨的“会话次数”才是外部纪要与待办调用的计费口径：")
    add_two_col_label_table(
        doc,
        [
            ("本地 ASR", "large-v3-turbo 实时识别和 large-v3 会后精修不调用远程 LLM API，不产生远程 API token；主要消耗 GPU 时间、显存、CPU/RAM 和磁盘。"),
            ("本地翻译", "OPUS-MT en→zh、de→zh 通过 CTranslate2 本地批量翻译，不产生远程 API token；成本主要是推理时间和模型存储。"),
            ("本地说话人分离", "Resemblyzer voice encoder 使用本地 pretrained.pt，不产生远程 API token；必须将包和权重准备状态纳入健康检查。"),
            ("积墨会议纪要", "每一次积墨 API 请求消耗 2 次“会话次数”。积墨计费为 20 万次会话 6,000 元，因此 1 次会话=0.03 元；1 次 API 请求=2×0.03=0.06 元。"),
            ("积墨 To-do-list", "To-do 接口每次请求同样消耗 2 次会话次数，即 0.06 元/请求；如果自动重试，每次重试再增加 0.06 元。"),
            ("总成本公式", "积墨费用 = API 请求次数 × 0.06 元 = 会话次数 × 0.03 元；会话次数 = API 请求次数 × 2。"),
        ],
        header=("阶段", "调用与 token 口径"),
        label_width=2050,
        body_size=9.5,
    )
    add_callout(
        doc,
        "估算公式",
        "20 万次会话=6,000 元 → 0.03 元/会话。每次 API 请求消耗 2 次会话 → 0.06 元/请求。一次会议的积墨费用 =（摘要 API 请求数 + To-do API 请求数 + 重试请求数）×0.06 元。摘要按分块状态更新和最终生成执行，若有 N 个转写块，通常至少 N+1 次请求；To-do 通常 1 次，因此无重试时约（N+2）×0.06 元。",
        fill=PALE_GOLD,
        accent=GOLD,
        text_size=10.0,
    )
    add_para(doc, "以 15,000 字符的会议转写为例，N≈3：摘要通常 4 次 API 请求，To-do 通常 1 次，共 5 次请求、10 次会话，积墨费用约 0.30 元；若摘要状态压缩增加 1 次请求，则约 0.36 元；每次重试额外 0.06 元。Token 与积墨会话次数是两套口径：会话次数用于积墨计费，token 用于模型输入/输出用量统计。若积墨响应没有返回 provider usage，就不能从字符数精确反推 token；生产 metrics 应记录 input_tokens、output_tokens、API 请求数、会话次数、重试次数和预计费用。", style="Small Note")

    add_heading(doc, "10. 可靠性、安全与兼容性", 1)
    add_heading(doc, "可靠性设计", 2)
    for text in [
        "模型缺失时前置阻断：不让用户录完整场会议后才发现停录阶段无法精修。",
        "后处理失败不影响录音：recording_state 保持 complete，postprocess 标记 error/partial，提供阶段级重试。",
        "重启可恢复：running 任务自动变为 queued，从最近完成的 checkpoint 继续。",
        "断线可恢复：WebSocket 短暂断线保留 15 秒恢复窗口；前端只接受不早于当前 snapshot_revision 的状态。",
        "指标可观测：/api/v2/health 和 /api/v2/metrics 暴露模型就绪、队列、耗时、失败、重试、OOM、模型回退、积墨 API 请求数、会话次数和预计费用等指标。",
    ]:
        add_bullet(doc, text)
    add_heading(doc, "安全与数据边界", 2)
    for text in [
        "浏览器不接收积墨 URL 和 Authorization；密钥只在服务端环境变量中使用。",
        "音频、ASR、翻译和 Resemblyzer 说话人重排默认本地运行；只有启用积墨后，纪要/To-do 所需文本才会发往外部服务。",
        "本地结果保留 transcript.json、transcript.jsonl、manifest、事件日志和模型元数据，便于删除、审计和迁移。",
        "生产企业部署仍需补充录音加密、访问审计、SSO/OIDC、对象存储生命周期和跨区域传输评估；这些属于企业部署扩展项。",
    ]:
        add_bullet(doc, text)
    add_heading(doc, "兼容性", 2)
    add_para(doc, "保留既有 /api/v2 API、WebSocket 事件、结果文件名和历史会议读取能力，并新增 /api/v2/health、/api/v2/metrics、postprocess_update、postprocess 和 snapshot_revision。旧 Utterance 没有新增字段时使用默认值直接读取。")

    doc.add_page_break()
    add_heading(doc, "11. 产品路线与下一步", 1)
    add_matrix_table(
        doc,
        ["阶段", "目标", "关键验收"],
        [
            ["正式产品能力", "稳定运行实时转写、停录快速保存、会后精修、说话人重排、翻译、纪要和 To-do", "不再无限期“正在保存”；任务可重试、可恢复；旧会议可读取和下载。"],
            ["质量评测与发布", "建立固定中英德混合语料和可重复 benchmark", "WER/CER、语言识别、首结果/P50/P95、RTF、显存、BLEU/chrF、DER/JER 和 E2E 保存/后处理耗时。"],
            ["企业扩展版", "从单机本地 store/queue 扩展到可横向部署", "PostgreSQL 元数据、S3/MinIO 音频、Redis/RabbitMQ/NATS 队列、GPU worker pool、SSO、审计和配额。"],
            ["能力扩展版", "扩大语言和身份能力", "新增翻译方向、更多语言质量门槛；实名声纹必须经过用户授权、隐私评估和独立验收。"],
        ],
        [1750, 3200, 4410],
        body_size=9.2,
    )

    add_heading(doc, "30 秒产品介绍话术", 1)
    add_callout(
        doc,
        "对外介绍",
        "会记是一套本地优先的实时会议工作台，支持中文、英文和德文会议的实时转写与中译。它在录音期间用轻量实时模型帮助参会者跟上讨论，停录后自动用更高质量 ASR 和说话人分离模型完成精修；精修、重排和翻译全部完成后，用户点击按钮生成会议纪要，随后系统自动生成 To-do-list。与单纯字幕或云端会议机器人不同，会记把录音保存、后台后处理和结果修订拆开，既保留本地数据边界，也让失败可见、任务可重试、结果可追溯。",
        fill=LIGHT_TEAL,
        accent=TEAL,
        text_size=10.8,
    )

    add_heading(doc, "12. 资料依据与口径说明", 1)
    add_para(doc, "产品与技术参数依据本仓库的 README.md、DEPLOYMENT.md、pyproject.toml、realtime_meeting/config.py、runtime.py、session.py、storage.py 和已实现的产品能力整理；竞品与模型资料为公开官方页面，访问日期：2026-08-13。公开产品能力可能随版本、地区和套餐变化。")
    add_source(doc, "Whisper large-v3-turbo 模型卡", "https://huggingface.co/openai/whisper-large-v3-turbo", "用于模型能力、99 语言模型卡与 turbo/large-v3 解码层差异说明。")
    add_source(doc, "Resemblyzer 项目与 voice encoder", "https://github.com/resemble-ai/Resemblyzer", "用于本地 voice encoder、声纹 embedding 和 speaker verification/聚类实现口径；本项目按 16 kHz 单声道输入封装。")
    add_source(doc, "CTranslate2 性能文档", "https://opennmt.net/CTranslate2/performance.html", "用于量化、beam size 和本地翻译推理的技术依据。")
    add_source(doc, "钉钉会议智能翻译与实时字幕", "https://help.aliyun.com/zh/document_detail/208720.html", "用于钉钉会议实时语音转文字、实时字幕和中英翻译能力的公开口径。")
    add_source(doc, "钉钉 AI 听记", "https://www.dingtalk.com/qidian/page-EBTszUVP.html", "用于钉钉实时转写与智能总结的公开产品定位。")
    add_source(doc, "腾讯会议实时字幕", "https://meeting.tencent.com/support/topic/1861/index.html", "用于实时字幕、自动识别和中英互译能力的公开说明。")
    add_source(doc, "腾讯会议智能录制与 AI 纪要", "https://meeting.tencent.com/support/topic/1985/index.html", "用于章节、重点、发言人、纪要和待办等会后能力的公开说明。")
    add_source(doc, "腾讯会议官方混合云部署", "https://meeting.tencent.com/news/neiwanghuiyi.html", "用于专网会议、媒体下沉、混合云和 TCO 口径；不作为统一私有化报价。")
    add_source(doc, "企业微信 App Store 产品说明", "https://apps.apple.com/us/app/%E4%BC%81%E4%B8%9A%E5%BE%AE%E4%BF%A1-%E7%A7%81%E6%9C%89%E9%83%A8%E7%BD%B2/id1466928593?platform=mac", "用于企业微信智能纪要、实时字幕和同声传译等公开能力口径。")
    add_source(doc, "飞书妙记产品页", "https://www.feishu.cn/product/minutes?lailu=www.ciuic.cn", "用于飞书转写、AI 总结、章节、待办和协作能力的公开产品定位。")

    # Make all default table paragraphs explicit and avoid empty trailing paragraphs with excess spacing.
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    p.paragraph_format.widow_control = True

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    build_document()
