from __future__ import annotations

import sys
import time
from pathlib import Path


def main() -> None:
    input_path = Path(sys.argv[1]).resolve()
    output_dir = Path(sys.argv[2]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / (input_path.stem + ".pdf")

    import win32com.client  # type: ignore

    word = win32com.client.DispatchEx("Word.Application")
    word.Visible = False
    word.DisplayAlerts = 0
    document = None
    try:
        document = word.Documents.Open(str(input_path), ReadOnly=True, AddToRecentFiles=False)
        document.ExportAsFixedFormat(
            OutputFileName=str(pdf_path),
            ExportFormat=17,  # wdExportFormatPDF
            OpenAfterExport=False,
            OptimizeFor=0,  # wdExportOptimizeForPrint
            Range=0,
            Item=0,
            IncludeDocProps=True,
            KeepIRM=True,
            CreateBookmarks=1,
            DocStructureTags=True,
            BitmapMissingFonts=True,
            UseISO19005_1=False,
        )
        for _ in range(50):
            if pdf_path.exists() and pdf_path.stat().st_size > 0:
                break
            time.sleep(0.1)
    finally:
        if document is not None:
            document.Close(False)
        word.Quit()
    if not pdf_path.exists() or pdf_path.stat().st_size == 0:
        raise RuntimeError(f"Word did not create a PDF at {pdf_path}")
    print(pdf_path)


if __name__ == "__main__":
    main()
