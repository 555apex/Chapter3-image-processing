"""Render the coursework Markdown with embedded Chinese fonts and math figures."""
import argparse
import hashlib
import html
import os
from pathlib import Path
import re
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, KeepTogether

def render(source, target, font=None):
    source, target = Path(source).resolve(), Path(target).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    font = font or "C:/Windows/Fonts/simsun.ttc"
    pdfmetrics.registerFont(TTFont("Chinese", font, subfontIndex=0))
    pdfmetrics.registerFontFamily("Chinese", normal="Chinese", bold="Chinese", italic="Chinese", boldItalic="Chinese")
    width = A4[0] - 96
    base = dict(fontName="Chinese", wordWrap="CJK", textColor=colors.HexColor("#243244"))
    styles = {
        "body": ParagraphStyle("body", fontSize=10.3, leading=16.4, spaceAfter=7, **base),
        "h1": ParagraphStyle("h1", fontSize=23, leading=32, spaceAfter=16, keepWithNext=True, **base),
        "h2": ParagraphStyle("h2", fontSize=15, leading=22, spaceBefore=13, spaceAfter=8, keepWithNext=True, **base),
        "h3": ParagraphStyle("h3", fontSize=12, leading=18, spaceBefore=10, spaceAfter=6, keepWithNext=True, **base),
        "cell": ParagraphStyle("cell", fontSize=8.6, leading=12.4, **base),
        "small": ParagraphStyle("small", fontSize=9, leading=14, spaceAfter=7, **base),
        "code": ParagraphStyle("code", fontSize=8.3, leading=12, backColor=colors.HexColor("#F1F4F7"), borderPadding=7, spaceAfter=8, **base),
    }
    cache = target.parent / ".cache/pdf-math"
    cache.mkdir(parents=True, exist_ok=True)
    def inline(s):
        # SimSun lacks U+2212; ASCII minus preserves mathematical meaning visibly.
        s = s.replace("\u2212", "-")
        s = html.escape(s)
        s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", lambda m: '<a href="'+m[2].removeprefix("&lt;").removesuffix("&gt;")+'" color="#17648A">'+m[1]+"</a>", s)
        s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
        s = s.replace(chr(96), "")
        return s
    def paragraph(s, style="body"):
        return Paragraph(inline(s), styles[style])
    def figure(path, max_height=325):
        with PILImage.open(path) as im:
            w, h = im.size
        scale = min(width/w, max_height/h)
        return Image(str(path), width=w*scale, height=h*scale)
    lines = source.read_text(encoding="utf-8").splitlines()
    story, i = [], 0
    title = lines[0].lstrip("# ")[:45]
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith(chr(96)*3):
            block = []
            i += 1
            while i < len(lines) and not lines[i].startswith(chr(96)*3):
                block.append(lines[i])
                i += 1
            # Wrap long commands so absolute Windows paths never overflow the page.
            for row in block:
                story.append(paragraph(row if row else " ", "code"))
            i += 1
            continue
        if line.startswith("$$"):
            expression = line[2:]
            while not expression.endswith("$$"):
                i += 1
                expression += " " + lines[i].strip()
            expression = expression[:-2].strip()
            key = hashlib.sha256(expression.encode()).hexdigest()[:16]
            path = cache / (key + ".png")
            if not path.exists():
                fig = plt.figure(figsize=(10, 1.0), dpi=180)
                fig.text(.5, .5, "$"+expression+"$", ha="center", va="center", fontsize=15)
                fig.savefig(path, bbox_inches="tight", pad_inches=.15, facecolor="white")
                plt.close(fig)
            with PILImage.open(path) as im:
                w, h = im.size
            scale = min(72/180, width/w)
            story.extend([Image(str(path), width=w*scale, height=h*scale), Spacer(1, 7)])
            i += 1
            continue
        image = re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", line)
        if image:
            story.extend([Spacer(1, 5), figure(source.parent / image[2]), Spacer(1, 7)])
            i += 1
            continue
        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                row = [x.strip() for x in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r"[:\-\s]+", x) for x in row):
                    rows.append(row)
                i += 1
            n = len(rows[0])
            table = Table([[paragraph(c, "cell") for c in row] for row in rows],
                          colWidths=[width/n]*n, repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#E7EDF3")),
                ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F7F9FB")]),
                ("VALIGN", (0,0), (-1,-1), "TOP"), ("LEFTPADDING",(0,0),(-1,-1),6),
                ("RIGHTPADDING",(0,0),(-1,-1),6), ("TOPPADDING",(0,0),(-1,-1),6),
                ("BOTTOMPADDING",(0,0),(-1,-1),6),
                ("LINEBELOW",(0,0),(-1,0),.6,colors.HexColor("#A7B8C8")),
            ]))
            story.extend([table, Spacer(1, 9)])
            continue
        heading = re.match(r"^(#{1,3}) (.*)", line)
        if heading:
            story.append(paragraph(heading[2], "h"+str(len(heading[1]))))
            i += 1
            continue
        if line.startswith("- "):
            story.append(paragraph("• " + line[2:]))
            i += 1
            continue
        block = [line]
        i += 1
        while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith(("#", "|", "![", "$$", "- ", chr(96)*3)):
            block.append(lines[i].strip())
            i += 1
        story.append(paragraph(" ".join(block)))
    def decorate(canvas, doc):
        canvas.saveState()
        canvas.setFont("Chinese", 8)
        canvas.setFillColor(colors.HexColor("#738498"))
        canvas.drawString(48, A4[1]-29, "计算机视觉 · 第三章")
        canvas.drawRightString(A4[0]-48, A4[1]-29, title[:28])
        canvas.setStrokeColor(colors.HexColor("#D3DDE6"))
        canvas.line(48, 41, A4[0]-48, 41)
        canvas.drawString(48, 28, "2026年9月30日 · 可追溯与可复现")
        canvas.drawRightString(A4[0]-48, 28, str(doc.page))
        canvas.restoreState()
    doc = SimpleDocTemplate(str(target), pagesize=A4, leftMargin=48, rightMargin=48,
                            topMargin=51, bottomMargin=54, title=title, author="计算机视觉课程作业")
    doc.build(story, onFirstPage=decorate, onLaterPages=decorate)
    print(target)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("target")
    ap.add_argument("--font")
    args = ap.parse_args()
    render(args.source, args.target, args.font)
