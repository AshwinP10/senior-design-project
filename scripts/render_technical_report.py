"""Render the measured Markdown report to a paginated, standalone PDF."""
from pathlib import Path
import re
import textwrap
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                               TableStyle, Image, PageBreak, Preformatted, KeepTogether)

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / 'output/pdf/precision_striker_technical_report.pdf'
    output.parent.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'docs/TECHNICAL_REPORT.md'
    styles = getSampleStyleSheet()
    styles['BodyText'].fontSize = 9.5
    styles['BodyText'].leading = 13.5
    styles['BodyText'].spaceAfter = 7
    styles['Title'].fontSize = 27
    styles['Title'].leading = 32
    styles['Title'].textColor = colors.HexColor('#163b55')
    styles['Heading1'].fontSize = 17
    styles['Heading1'].leading = 22
    styles['Heading1'].spaceBefore = 16
    styles['Heading1'].spaceAfter = 10
    styles['Heading1'].textColor = colors.HexColor('#163b55')
    styles['Heading1'].keepWithNext = True
    styles['Heading2'].fontSize = 12
    styles['Heading2'].leading = 16
    styles['Heading2'].spaceBefore = 12
    styles['Heading2'].keepWithNext = True
    styles.add(ParagraphStyle('Cell', fontSize=8, leading=10.5, spaceAfter=0))
    styles.add(ParagraphStyle('Caption', fontSize=8.5, leading=11, textColor=colors.HexColor('#52616c'), spaceAfter=12))
    styles.add(ParagraphStyle('CodeSmall', fontName='Courier', fontSize=8.3, leading=11.2,
                              backColor=colors.HexColor('#edf3f6'), borderPadding=8,
                              spaceBefore=7, spaceAfter=10))
    def inline(s):
        s = escape(s)
        code = []
        def protect(match):
            code.append(match[1])
            return f'@@CODE{len(code)-1}@@'
        s = re.sub(r'`([^`]+)`', protect, s)
        s = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', s)
        s = re.sub(r'\*([^*]+)\*', r'<i>\1</i>', s)
        s = re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)', r'<link href="\2" color="#126587">\1</link>', s)
        # Long provenance hashes need opportunities to wrap.
        s = re.sub(r'\b[a-f0-9]{64}\b', lambda m: '<br/>'.join(textwrap.wrap(m[0], 32)), s)
        for index, value in enumerate(code):
            s = s.replace(f'@@CODE{index}@@', '<font name="Courier">'+value+'</font>')
        return s
    flow = []
    lines = source.read_text(encoding='utf-8').splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line == '<!-- pagebreak -->':
            flow.append(PageBreak()); i += 1; continue
        if line.startswith('```'):
            block = []
            i += 1
            while i < len(lines) and not lines[i].startswith('```'):
                block.extend(textwrap.wrap(lines[i], width=96, replace_whitespace=False,
                                           drop_whitespace=False) or [''])
                i += 1
            flow.append(KeepTogether([Preformatted('\n'.join(block), styles['CodeSmall'])]))
            i += 1; continue
        if line.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                cells = [c.strip() for c in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r'[: -]+', c) for c in cells):
                    rows.append([Paragraph(inline(c), styles['Cell']) for c in cells])
                i += 1
            n = len(rows[0])
            widths = {2:[190,321],3:[144,181,186],4:[180,110,110,111],5:[163,87,87,87,87],6:[126,77,77,77,77,77]}.get(n, [511/n]*n)
            table = Table(rows, colWidths=widths, repeatRows=1, hAlign='LEFT')
            table.setStyle(TableStyle([
                ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#d8e8f0')),
                ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f3f7f9')]),
                ('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),6),
                ('BOTTOMPADDING',(0,0),(-1,-1),6),('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#9ab2c1'))]))
            flow.extend([table, Spacer(1,10)]); continue
        match = re.match(r'!\[(.*?)\]\((.*?)\)', line)
        if match:
            path = source.parent / match[2]
            w,h = ImageReader(str(path)).getSize()
            scale = min(511/w, 520/h)
            flow.append(Image(str(path), w*scale, h*scale))
            flow.append(Paragraph(inline(match[1]), styles['Caption']))
            i += 1; continue
        if line.startswith('# '):
            flow.append(Paragraph(inline(line[2:]), styles['Title'] if not flow else styles['Heading1']))
        elif line.startswith('## '):
            if line == '## References':
                flow.append(PageBreak())
            flow.append(Paragraph(inline(line[3:]), styles['Heading1']))
        elif line.startswith('### '):
            flow.append(Paragraph(inline(line[4:]), styles['Heading2']))
        else:
            para = [line]
            while i+1 < len(lines) and lines[i+1].strip() and not lines[i+1].startswith(('#','|','```','![')):
                i += 1
                para.append(lines[i].strip())
            flow.append(Paragraph(inline(' '.join(para)), styles['BodyText']))
        i += 1
    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor('#b3c8d4'))
        canvas.line(42, 38, 553, 38)
        canvas.setFont('Helvetica',7.5)
        canvas.setFillColor(colors.HexColor('#52616c'))
        canvas.drawString(42,25,'PRECISION STRIKER  /  EXPERIMENTAL RL  /  24 SEPTEMBER 2026')
        canvas.drawRightString(553,25,str(doc.page))
        canvas.restoreState()
    doc = SimpleDocTemplate(str(output), pagesize=(595,842), rightMargin=42,
                            leftMargin=42, topMargin=43, bottomMargin=53,
                            title='Precision Striker: RL Model, Training and Results',
                            author='Senior Design Project', pageCompression=1)
    doc.build(flow, onFirstPage=footer, onLaterPages=footer)
    print(output)


if __name__ == '__main__':
    main()
