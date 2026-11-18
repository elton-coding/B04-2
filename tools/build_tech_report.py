#!/usr/bin/env python3
"""把 docs/技术方案报告/正文.md 套入官方《初赛技术方案提交模板》，生成正式版与盲审版 docx/PDF。

- 正式版：封面保留"<队名>技术方案"与团队负责人、所在单位、填写时间（未提供的留空下划线，打印后手填并盖章）。
- 盲审版：封面去掉队名与全部身份栏，文档属性（作者/最后修改者）清空，PDF 元数据同样清空。
- 目录、页码由本机 Word 更新后导出 PDF（需安装 Microsoft Word）。

用法：
  python tools/build_tech_report.py --team 同频队 [--leader 张三] [--unit 某某学院] [--date 2026年12月]
输出：dist/技术方案/<队名>_技术方案_正式版.{docx,pdf}、dist/技术方案/技术方案_盲审版.{docx,pdf}
"""
from __future__ import annotations

import argparse
import copy
import os
import re
import shutil
import struct
import subprocess
import tempfile
import zipfile
from xml.sax.saxutils import escape

from lxml import etree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT, "docs", "技术方案报告")
TEMPLATE = os.path.join(ROOT, "(公开)B04赛题-初赛技术方案提交模板-11月12日更新.docx")

NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
      "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
      "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
      "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
      "pic": "http://schemas.openxmlformats.org/drawingml/2006/picture"}
W = "{%s}" % NS["w"]
DECL = " ".join(f'xmlns:{k}="{v}"' for k, v in NS.items())
TEXT_W = 8306  # 正文宽度（twip）：A4 11906 − 左右边距 1800×2

# 模板指定的 GB2312 字体在多数机器上没有，换成 Windows 自带的同族字体，避免 PDF 中被随机替换
FONT_MAP = {"仿宋_GB2312": "仿宋", "楷体_GB2312": "楷体", "方正小标宋简体": "华文中宋"}


# ---------------------------------------------------------------- 段落构件
def rpr(east="仿宋", size=32, bold=False, ascii_font="Times New Roman"):
    return (f'<w:rPr><w:rFonts w:ascii="{ascii_font}" w:eastAsia="{east}" w:hAnsi="{ascii_font}" w:cs="Times New Roman" w:hint="eastAsia"/>'
            f'{"<w:b/><w:bCs/>" if bold else ""}<w:kern w:val="0"/><w:sz w:val="{size}"/><w:szCs w:val="{size}"/></w:rPr>')


def runs(text, east="仿宋", size=32, bold=False):
    out = []
    for i, seg in enumerate(re.split(r"\*\*", text)):
        if seg:
            out.append(f'<w:r>{rpr(east, size, bold or i % 2 == 1)}<w:t xml:space="preserve">{escape(seg)}</w:t></w:r>')
    return "".join(out)


def p_body(text):
    return ('<w:p><w:pPr><w:spacing w:line="560" w:lineRule="exact"/><w:ind w:firstLineChars="200" w:firstLine="640"/>'
            f'<w:jc w:val="both"/></w:pPr>{runs(text)}</w:p>')


def p_abstract(text):
    return ('<w:p><w:pPr><w:spacing w:line="500" w:lineRule="exact"/><w:ind w:firstLineChars="200" w:firstLine="560"/>'
            f'<w:jc w:val="both"/></w:pPr>{runs(text, size=28)}</w:p>')


def p_h1(text):
    return f'<w:p><w:pPr><w:pStyle w:val="01GF"/><w:keepNext/><w:spacing w:before="360" w:after="120"/></w:pPr>{runs(text, "黑体", 32)}</w:p>'


def p_h2(text):
    return ('<w:p><w:pPr><w:keepNext/><w:spacing w:before="120" w:line="560" w:lineRule="exact"/>'
            f'<w:ind w:firstLineChars="200" w:firstLine="640"/><w:outlineLvl w:val="1"/></w:pPr>{runs(text, "楷体", 32, True)}</w:p>')


def p_h3(text):
    return ('<w:p><w:pPr><w:keepNext/><w:spacing w:line="560" w:lineRule="exact"/>'
            f'<w:ind w:firstLineChars="200" w:firstLine="640"/><w:outlineLvl w:val="2"/></w:pPr>{runs(text, "仿宋", 32, True)}</w:p>')


def p_caption(text, keep_next=False):
    return (f'<w:p><w:pPr>{"<w:keepNext/>" if keep_next else ""}<w:spacing w:before="60" w:after="160" w:line="360" w:lineRule="auto"/>'
            f'<w:jc w:val="center"/></w:pPr>{runs(text, "黑体", 24)}</w:p>')


def p_note(text):
    return ('<w:p><w:pPr><w:spacing w:after="120" w:line="400" w:lineRule="exact"/><w:jc w:val="both"/></w:pPr>'
            f'{runs(text, "楷体", 24)}</w:p>')


def png_size(path):
    with open(path, "rb") as f:
        head = f.read(24)
    return struct.unpack(">II", head[16:24])


def p_image(rid, path, width_cm, pid):
    pw, ph = png_size(path)
    cx = int(width_cm * 360000)
    cy = int(cx * ph / pw)
    name = os.path.basename(path)
    return ('<w:p><w:pPr><w:keepNext/><w:spacing w:before="120" w:line="240" w:lineRule="auto"/><w:jc w:val="center"/></w:pPr><w:r><w:drawing>'
            f'<wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="{cx}" cy="{cy}"/><wp:docPr id="{pid}" name="{name}"/>'
            '<wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr>'
            '<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:pic>'
            f'<pic:nvPicPr><pic:cNvPr id="{pid}" name="{name}"/><pic:cNvPicPr/></pic:nvPicPr>'
            f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
            f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>'
            '</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>')


def disp_w(s):
    return sum(2 if ord(c) > 0x2E80 else 1 for c in s)


def table(rows):
    ncol = len(rows[0])
    need = [max(max(disp_w(r[j]) for r in rows), 4) for j in range(ncol)]
    tot = sum(need)
    widths = [max(700, int(TEXT_W * n / tot)) for n in need]
    widths[-1] += TEXT_W - sum(widths)
    left = [need[j] > 22 for j in range(ncol)]
    grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in widths)
    out = [f'<w:tbl><w:tblPr><w:tblW w:w="{TEXT_W}" w:type="dxa"/><w:jc w:val="center"/>'
           '<w:tblBorders><w:top w:val="single" w:sz="12" w:space="0" w:color="000000"/>'
           '<w:bottom w:val="single" w:sz="12" w:space="0" w:color="000000"/></w:tblBorders>'
           '<w:tblLayout w:type="fixed"/><w:tblCellMar><w:left w:w="80" w:type="dxa"/><w:right w:w="80" w:type="dxa"/></w:tblCellMar>'
           f'</w:tblPr><w:tblGrid>{grid}</w:tblGrid>']
    for i, r in enumerate(rows):
        out.append('<w:tr><w:trPr><w:cantSplit/>' + ('<w:tblHeader/>' if i == 0 else "") + "</w:trPr>")
        for j, cell in enumerate(r):
            bd = '<w:tcBorders><w:bottom w:val="single" w:sz="6" w:space="0" w:color="000000"/></w:tcBorders>' if i == 0 else ""
            jc = "left" if (left[j] and i > 0) else "center"
            out.append(f'<w:tc><w:tcPr><w:tcW w:w="{widths[j]}" w:type="dxa"/>{bd}<w:vAlign w:val="center"/></w:tcPr>'
                       f'<w:p><w:pPr><w:spacing w:before="40" w:after="40" w:line="300" w:lineRule="exact"/><w:jc w:val="{jc}"/></w:pPr>'
                       f'{runs(cell, "宋体", 21, i == 0)}</w:p></w:tc>')
        out.append("</w:tr>")
    out.append("</w:tbl>")
    out.append('<w:p><w:pPr><w:spacing w:line="240" w:lineRule="auto"/></w:pPr></w:p>')
    return "".join(out)


# ---------------------------------------------------------------- 源文件解析
def parse_md(path):
    text = open(path, encoding="utf-8").read()
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    abstract, keywords, blocks = [], "", []
    sec = None
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        ln = lines[i].rstrip()
        if not ln.strip():
            i += 1
            continue
        if ln.startswith("# "):
            title = ln[2:].strip()
            sec = "abstract" if title == "摘要" else "body"
            if sec == "body":
                blocks.append(("h1", title))
        elif sec == "abstract":
            if ln.startswith("关键词"):
                keywords = ln
            else:
                abstract.append(ln.strip())
        elif ln.startswith("## "):
            blocks.append(("h2", ln[3:].strip()))
        elif ln.startswith("### "):
            blocks.append(("h3", ln[4:].strip()))
        elif ln.startswith("[表]"):
            cap = ln[3:].strip()
            rows = []
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            blocks.append(("table", cap, rows))
            continue
        elif ln.startswith("!["):
            m = re.match(r"!\[(.*?)\]\((.*?)\)(?:\{([\d.]+)\})?", ln)
            blocks.append(("image", m.group(1), os.path.join(SRC_DIR, m.group(2)), float(m.group(3) or 14.5)))
        elif ln.startswith("> "):
            blocks.append(("note", ln[2:].strip()))
        else:
            blocks.append(("p", ln.strip()))
        i += 1
    return abstract, keywords, blocks


# ---------------------------------------------------------------- 组装
def frag(xml):
    """把若干个 w:p / w:tbl 串解析为元素列表。"""
    root = etree.fromstring(f"<root {DECL}>{xml}</root>")
    return list(root)


def build_docx(out_docx, blind, team, leader, unit, date):
    tmp = tempfile.mkdtemp(prefix="b04tr_")
    zipfile.ZipFile(TEMPLATE).extractall(tmp)
    wd = os.path.join(tmp, "word")
    for fn in ("document.xml", "styles.xml", "fontTable.xml"):
        p = os.path.join(wd, fn)
        s = open(p, encoding="utf-8").read()
        for a, b in FONT_MAP.items():
            s = s.replace(a, b)
        open(p, "w", encoding="utf-8").write(s)

    tree = etree.parse(os.path.join(wd, "document.xml"))
    body = tree.getroot().find(W + "body")
    kids = list(body)

    def set_text(p, text):
        ts = p.findall(".//" + W + "t")
        ts[0].text = text
        for t in ts[1:]:
            t.text = ""

    # 封面
    set_text(kids[5], "技术方案" if blind else f"{team}技术方案")
    if blind:
        for k in kids[9:13]:
            body.remove(k)
    else:
        for k, val in zip(kids[10:13], (leader, unit, date)):
            ts = k.findall(".//" + W + "t")
            pad = max(2, 38 - 2 - disp_w(val or ""))
            ts[1].text = "  " + (val or "") + " " * pad
            for t in ts[2:]:
                t.text = ""

    # 摘要
    abstract, keywords, blocks = parse_md(os.path.join(SRC_DIR, "正文.md"))
    anchor = kids[21]
    for el in frag("".join(p_abstract(t) for t in abstract) +
                   (p_abstract("**关键词：**" + keywords.split("：", 1)[1]) if keywords else "")):
        anchor.addprevious(el)
    body.remove(anchor)

    # 目录：替换为新的 TOC 域（Word 打开后更新）
    sdt = kids[25]
    content = sdt.find(W + "sdtContent")
    for c in list(content):
        content.remove(c)
    for el in frag('<w:p><w:pPr><w:pStyle w:val="10"/><w:spacing w:line="360" w:lineRule="auto"/></w:pPr>'
                   '<w:r><w:fldChar w:fldCharType="begin"/></w:r><w:r><w:instrText xml:space="preserve"> TOC \\o "1-2" \\h \\z \\u </w:instrText></w:r>'
                   '<w:r><w:fldChar w:fldCharType="separate"/></w:r><w:r><w:t>目录待更新</w:t></w:r><w:r><w:fldChar w:fldCharType="end"/></w:r></w:p>'):
        content.append(el)

    # 正文：替换模板中的五章提示段
    rels_p = os.path.join(wd, "_rels", "document.xml.rels")
    rels = open(rels_p, encoding="utf-8").read()
    media = os.path.join(wd, "media")
    os.makedirs(media, exist_ok=True)
    xml, n_img = [], 0
    for b in blocks:
        kind = b[0]
        if kind == "h1":
            xml.append(p_h1(b[1]))
        elif kind == "h2":
            xml.append(p_h2(b[1]))
        elif kind == "h3":
            xml.append(p_h3(b[1]))
        elif kind == "p":
            xml.append(p_body(b[1]))
        elif kind == "note":
            xml.append(p_note(b[1]))
        elif kind == "table":
            xml.append(p_caption(b[1], keep_next=True))
            xml.append(table(b[2]))
        elif kind == "image":
            n_img += 1
            rid = f"rIdImg{n_img}"
            fn = f"image{n_img}.png"
            shutil.copy(b[2], os.path.join(media, fn))
            rels = rels.replace("</Relationships>",
                                f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/{fn}"/></Relationships>')
            xml.append(p_image(rid, b[2], b[3], 100 + n_img))
            xml.append(p_caption(b[1]))
    first, last = kids[27], kids[41]
    for el in frag("".join(xml)):
        first.addprevious(el)
    for k in kids[27:42]:
        body.remove(k)

    # 页脚页码（仅正文节）
    with open(os.path.join(wd, "footer1.xml"), "w", encoding="utf-8") as f:
        f.write(f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:ftr {DECL}><w:p><w:pPr><w:jc w:val="center"/></w:pPr>'
                '<w:r><w:fldChar w:fldCharType="begin"/></w:r><w:r><w:instrText xml:space="preserve"> PAGE </w:instrText></w:r>'
                '<w:r><w:fldChar w:fldCharType="separate"/></w:r><w:r><w:t>1</w:t></w:r><w:r><w:fldChar w:fldCharType="end"/></w:r></w:p></w:ftr>')
    rels = rels.replace("</Relationships>",
                        '<Relationship Id="rIdFtr1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer" Target="footer1.xml"/></Relationships>')
    open(rels_p, "w", encoding="utf-8").write(rels)
    sect = body.find(W + "sectPr")
    for el in frag('<w:footerReference w:type="default" r:id="rIdFtr1"/>'):
        sect.insert(0, el)
    pg = sect.find(W + "pgMar")
    for el in frag('<w:pgNumType w:start="1"/>'):
        pg.addnext(el)
    tree.write(os.path.join(wd, "document.xml"), xml_declaration=True, encoding="UTF-8", standalone=True)

    ct_p = os.path.join(tmp, "[Content_Types].xml")
    ct = open(ct_p, encoding="utf-8").read()
    ct = ct.replace('<Default Extension="xml"', '<Default Extension="png" ContentType="image/png"/><Default Extension="xml"')
    ct = ct.replace("</Types>", '<Override PartName="/word/footer1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/></Types>')
    open(ct_p, "w", encoding="utf-8").write(ct)

    # 文档属性：清空作者等个人信息
    core_p = os.path.join(tmp, "docProps", "core.xml")
    core = open(core_p, encoding="utf-8").read()
    core = re.sub(r"<dc:creator>.*?</dc:creator>", "<dc:creator></dc:creator>", core)
    core = re.sub(r"<cp:lastModifiedBy>.*?</cp:lastModifiedBy>", "<cp:lastModifiedBy></cp:lastModifiedBy>", core)
    core = re.sub(r"<dc:title>.*?</dc:title>", "<dc:title>电磁目标本体表征与多源数据融合智能体技术方案</dc:title>", core)
    open(core_p, "w", encoding="utf-8").write(core)
    app_p = os.path.join(tmp, "docProps", "app.xml")
    app = open(app_p, encoding="utf-8").read()
    app = re.sub(r"<Company>.*?</Company>", "<Company></Company>", app)
    open(app_p, "w", encoding="utf-8").write(app)

    if os.path.exists(out_docx):
        os.remove(out_docx)
    with zipfile.ZipFile(out_docx, "w", zipfile.ZIP_DEFLATED) as z:
        for dp, _, fs in os.walk(tmp):
            for f in fs:
                full = os.path.join(dp, f)
                z.write(full, os.path.relpath(full, tmp).replace(os.sep, "/"))
    shutil.rmtree(tmp, ignore_errors=True)


PS = r"""
$ErrorActionPreference = 'Stop'
$w = New-Object -ComObject Word.Application
$w.Visible = $false
$w.DisplayAlerts = 0
try {
  $d = $w.Documents.Open($env:B04_DOCX)
  foreach ($t in $d.TablesOfContents) { $t.Update() }
  $d.Fields.Update() | Out-Null
  foreach ($t in $d.TablesOfContents) { $t.Update(); $t.Range.Font.Bold = 0; $t.Range.Font.NameFarEast = 'SimSun'; $t.Range.Font.Size = 12 }
  $d.RemovePersonalInformation = $true
  $d.Save()
  $d.ExportAsFixedFormat($env:B04_PDF, 17, $false, 0, 0, 1, 1, 0, $true, $true, 1)
  $d.Close(0)
} finally { $w.Quit([ref]0); [void][Runtime.InteropServices.Marshal]::ReleaseComObject($w) }
"""


def word_finalize(docx, pdf):
    env = dict(os.environ, B04_DOCX=os.path.abspath(docx), B04_PDF=os.path.abspath(pdf))
    ps = os.path.join(tempfile.gettempdir(), "b04_word_finalize.ps1")
    with open(ps, "w", encoding="ascii") as f:
        f.write(PS)
    subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", ps], env=env, check=True, timeout=600)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True)
    ap.add_argument("--leader", default="")
    ap.add_argument("--unit", default="")
    ap.add_argument("--date", default="")
    ap.add_argument("--no-pdf", action="store_true")
    args = ap.parse_args()
    out = os.path.join(ROOT, "dist", "技术方案")
    os.makedirs(out, exist_ok=True)
    jobs = [(os.path.join(out, f"{args.team}_技术方案_正式版"), False), (os.path.join(out, "技术方案_盲审版"), True)]
    for stem, blind in jobs:
        build_docx(stem + ".docx", blind, args.team, args.leader, args.unit, args.date)
        if not args.no_pdf:
            word_finalize(stem + ".docx", stem + ".pdf")
        print("written", stem + (".docx/.pdf" if not args.no_pdf else ".docx"))
    # 盲审版自检：不得出现队名、负责人、单位
    import pypdf
    bad = [s for s in (args.team, args.leader, args.unit) if s]
    for stem, blind in jobs:
        if not blind:
            continue
        with zipfile.ZipFile(stem + ".docx") as z:
            blob = "".join(z.read(n).decode("utf-8", "ignore") for n in z.namelist() if n.endswith(".xml"))
        hits = [s for s in bad if s in blob]
        if not args.no_pdf:
            r = pypdf.PdfReader(stem + ".pdf")
            txt = "".join(p.extract_text() or "" for p in r.pages)
            hits += [s for s in bad if s in txt]
            meta = r.metadata or {}
            print("盲审版 PDF 元数据:", {k: v for k, v in meta.items()})
            print("盲审版 PDF 页数:", len(r.pages))
        print("盲审版身份信息检查:", "未发现" if not hits else f"发现 {hits}！")


if __name__ == "__main__":
    main()
