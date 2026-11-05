"""Generate deterministic, entirely synthetic PDFs for offline tests and demos."""

from __future__ import annotations

import argparse
import io
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _chart_png() -> bytes:
    image = Image.new("RGB", (640, 340), "white")
    draw = ImageDraw.Draw(image)
    draw.line((70, 285, 600, 285), fill="#25314d", width=3)
    draw.line((70, 35, 70, 285), fill="#25314d", width=3)
    values = [("FY2022", 120), ("FY2023", 175), ("FY2024", 230)]
    colors = ("#4c78a8", "#72b7b2", "#f58518")
    for index, ((label, height), color) in enumerate(zip(values, colors, strict=True)):
        x0 = 115 + index * 155
        draw.rectangle((x0, 285 - height, x0 + 80, 285), fill=color)
        draw.text((x0 + 8, 292), label, fill="#111827")
        draw.text((x0 + 25, 268 - height), str((index + 2) * 100), fill="#111827")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _scan_png() -> bytes:
    image = Image.new("RGB", (1240, 1754), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=28)
    draw.text((100, 160), "SCANNED OPERATING REVIEW", font=font, fill="black")
    draw.text((100, 260), "Employee retention improved to 94 percent in FY2024.", font=font, fill="black")
    draw.text((100, 330), "This page deliberately has no PDF text layer.", font=font, fill="black")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def create_fixture(path: Path) -> Path:
    import pymupdf

    path.parent.mkdir(parents=True, exist_ok=True)
    document = pymupdf.open()
    page = document.new_page(width=612, height=792)
    page.insert_text((54, 55), "SYNTHETIC FINANCIAL REPORT", fontsize=17)
    page.insert_textbox(
        pymupdf.Rect(54, 75, 558, 145),
        "The company expanded its services segment during FY2024. This synthetic report contains no real company data.",
        fontsize=11,
    )
    page.insert_text((54, 175), "Table 1. Segment performance (USD millions)", fontsize=11)
    x = (54, 250, 360, 500)
    y = (195, 225, 255, 285, 315)
    for x_pos in x:
        page.draw_line((x_pos, y[0]), (x_pos, y[-1]), color=(0, 0, 0), width=0.8)
    for y_pos in y:
        page.draw_line((x[0], y_pos), (x[-1], y_pos), color=(0, 0, 0), width=0.8)
    rows = [
        ("Metric", "FY2023", "FY2024"),
        ("Revenue", "$1,100", "$1,250"),
        ("Operating profit", "(120)", "$180"),
        ("Margin", "10.9%", "14.4%"),
    ]
    for row_index, row in enumerate(rows):
        for column_index, value in enumerate(row):
            page.insert_text((x[column_index] + 6, y[row_index] + 20), value, fontsize=9)

    page = document.new_page(width=612, height=792)
    page.insert_text((54, 55), "REVENUE TREND", fontsize=17)
    page.insert_image(pymupdf.Rect(54, 100, 558, 390), stream=_chart_png())
    page.insert_text((54, 415), "Figure 1. Synthetic revenue trend for FY2022-FY2024", fontsize=11)
    page.insert_textbox(
        pymupdf.Rect(54, 445, 558, 510),
        "The chart is visual evidence. Its numeric series is illustrative; "
        "chart-to-data reconstruction is out of scope.",
        fontsize=10,
    )

    page = document.new_page(width=612, height=792)
    page.insert_image(page.rect, stream=_scan_png())
    document.set_metadata({"title": "Synthetic Financial Report", "author": "mmrag-finance tests"})
    document.save(path, garbage=4, deflate=True)
    document.close()
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/raw/synthetic_financial_report.pdf"))
    args = parser.parse_args()
    print(create_fixture(args.output))


if __name__ == "__main__":
    main()
