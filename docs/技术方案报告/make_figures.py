"""生成技术方案报告的插图：框图用 SVG 绘制后由 Edge 无头模式截图为 PNG，统计图用 matplotlib。

用法：python docs/技术方案报告/make_figures.py
"""
from __future__ import annotations

import html
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
SRC = os.path.join(HERE, "figures_src")
EDGE = next(p for p in (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe") if os.path.exists(p))

INK, INK2, LINE = "#1f1f1e", "#52514e", "#8a8984"
C = {"obs": "#2a78d6", "und": "#eb6834", "fus": "#1baf7a", "ont": "#6250d6", "gen": "#d0487a",
     "io": "#6b6a66", "eng": "#3b3a37"}


class Svg:
    def __init__(self, w, h):
        self.w, self.h, self.el = w, h, []
        self.el.append(
            '<defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M0,0 L10,5 L0,10 z" fill="{LINE}"/></marker></defs>')

    def box(self, x, y, w, h, title, lines=(), color=INK2, fill="#ffffff", title_size=19, size=15, bold=True, rx=8):
        self.el.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{color}" stroke-width="2"/>')
        n = 1 + len(lines)
        lh = size + 5
        top = y + h / 2 - ((title_size + 4) + lh * (n - 1)) / 2 + title_size - 2
        self.text(x + w / 2, top, title, title_size, color if bold else INK, bold)
        for i, s in enumerate(lines):
            self.text(x + w / 2, top + (title_size + 6) + i * lh, s, size, INK2)

    def band(self, x, y, w, h, label, color):
        self.el.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{color}" fill-opacity="0.07" stroke="{color}" stroke-opacity="0.45" stroke-width="1.5"/>')
        self.el.append(f'<rect x="{x}" y="{y}" width="46" height="{h}" rx="12" fill="{color}"/>')
        self.el.append(f'<rect x="{x + 30}" y="{y}" width="16" height="{h}" fill="{color}"/>')
        cy = y + h / 2 - (len(label) - 1) * 12
        for i, ch in enumerate(label):
            self.text(x + 23, cy + i * 24 + 7, ch, 20, "#ffffff", True)

    def text(self, x, y, s, size=15, color=INK, bold=False, anchor="middle"):
        self.el.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" text-anchor="{anchor}"'
                       f'{" font-weight=\"700\"" if bold else ""}>{html.escape(s)}</text>')

    def arrow(self, pts, dashed=False, color=LINE):
        d = " ".join(("M" if i == 0 else "L") + f"{x},{y}" for i, (x, y) in enumerate(pts))
        self.el.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2.2"{" stroke-dasharray=\"6 5\"" if dashed else ""} marker-end="url(#ar)"/>')

    def render(self, name):
        os.makedirs(SRC, exist_ok=True)
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" viewBox="0 0 {self.w} {self.h}" '
               f'font-family="Microsoft YaHei, SimHei, sans-serif">' + "".join(self.el) + "</svg>")
        page = (f'<!doctype html><html><head><meta charset="utf-8"><style>html,body{{margin:0;background:#fff}}</style></head>'
                f'<body>{svg}</body></html>')
        src = os.path.join(SRC, name + ".html")
        with open(src, "w", encoding="utf-8") as f:
            f.write(page)
        out = os.path.join(FIG, name + ".png")
        subprocess.run([EDGE, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2",
                        f"--window-size={self.w},{self.h}", f"--screenshot={out}", "file:///" + src.replace("\\", "/")],
                       check=True, capture_output=True, timeout=60)
        print("figure", out)


def architecture():
    W = 1100
    s = Svg(W, 1235)
    # 输入行
    ins = [("A 类脉冲观测", ["TOA/RF/PW/幅度/DOA", "→ 观测处理"]),
           ("B 类目标航迹", ["含虚假、静默航迹", "→ 关联融合（几何锚点）"]),
           ("配套文本资料", ["D1 公告 · D2 · D3", "→ 知识库：约束与本体"]),
           ("训练数据（离线）", ["波形指纹库 · 状态模型", "→ 目标理解"])]
    for i, (t, ls) in enumerate(ins):
        s.box(20 + i * 270, 12, 255, 108, t, ls, C["io"], title_size=21, size=16)
    X0, X1 = 20, 1080
    bands = [("观测处理", "obs", 150, 150), ("目标理解", "und", 330, 120), ("关联融合", "fus", 480, 182),
             ("本体表征", "ont", 692, 128), ("态势生成", "gen", 850, 128)]
    for lab, k, y, h in bands:
        s.band(X0, y, X1 - X0, h, lab, C[k])
    bx, inner = 80, 990

    def row(y, h, items, col, ts=20, sz=16):
        n = len(items)
        gap = 16
        w = (inner - gap * (n - 1)) / n
        for i, (t, ls) in enumerate(items):
            x = bx + i * (w + gap)
            s.box(x, y, w, h, t, ls, col, title_size=ts, size=sz)
            if i:
                s.arrow([(x - gap, y + h / 2), (x - 2, y + h / 2)])

    row(166, 118, [("数据接入", ["字段兼容 · 缺失清洗", "toa 单位 · 场景零点", "测向误差自校准"]),
                   ("CPI 串接", ["区间宽度判据", "PRI 恒定/抖动", "漏脉冲倍数容忍"]),
                   ("跨节点同源配对", ["传播时延+参数一致", "抖动序列平移复核"]),
                   ("发射 / 驻留组", ["同源 ∪ 首尾相接", "纯度 99.7%"])], C["obs"])
    s.box(bx, 346, 487, 88, "波形指纹型号识别", ["生成式似然，无类别先验，多 CPI 累积"], C["und"], title_size=20, size=16)
    s.box(bx + 503, 346, 487, 88, "两级工作状态识别", ["自估重访周期 · 周期命中 · 全局驻留链"], C["und"], title_size=20, size=16)
    row(496, 150, [("航迹锚定", ["距离感知残差", "TDOA · 型号一致", "传播与片段否决"]),
                   ("无锚定融合", ["同源支持度", "相位延续串接", "方位连续"]),
                   ("对象级精修", ["型号+几何", "+互斥", "迭代重指派"]),
                   ("错归纠正", ["时间互斥 · 补洞", "几何否决", "到达相位测距"]),
                   ("A-B 关联", ["匈牙利指派", "不关联虚拟列", "离散度 · TDOA"])], C["fus"], ts=19, sz=16)
    row(708, 96, [("态势要素抽取", ["运动 · 伴随 · 跟踪照射", "被动定位（测向+相位）"]),
                  ("电磁目标本体", ["14 类 · 15 关系", "证据 + 认识论状态"]),
                  ("图谱导出", ["OWL/Turtle · JSON-LD", "交互式图谱页面"])], C["ont"])
    row(866, 96, [("事实卡片", ["编号/时段/数值", "可回查结果表"]),
                  ("大模型研判", ["环境变量注入的模型", "事实锁定 · 失败降级"]),
                  ("八章态势报告", ["D4 结构 · 图表内嵌", "HTML / PDF"])], C["gen"])
    cx = 575
    for y1, y2 in ((120, 164), (300, 344), (450, 494), (662, 706), (820, 864), (978, 1006)):
        s.arrow([(cx, y1 + 1), (cx, y2)])
    outs = [("pulse_predictions", ["局部簇 · 对象 · 型号 · 状态"]), ("object_track_links", ["对象 ↔ 航迹 / no_match"]),
            ("本体与图谱", [".ttl / .jsonld / .html"]), ("态势报告", ["HTML / PDF"])]
    for i, (t, ls) in enumerate(outs):
        s.box(20 + i * 270, 1008, 255, 88, t, ls, C["io"], title_size=20, size=16)
    s.el.append(f'<rect x="20" y="1118" width="1060" height="104" rx="10" fill="{C["eng"]}"/>')
    s.text(550, 1152, "工程底座：Docker 一键运行 · 参数取自物理误差模型或数据在线估计 · 无场景硬编码", 19, "#ffffff", True)
    s.text(550, 1182, "单场景 860 万脉冲约 1 min · 分级降级与输出修复保证合法结果 · 两张结果表不依赖大模型", 17, "#e8e7e3")
    s.text(550, 1208, "大模型仅经评测注入的环境变量调用，不可用时自动降级为模板", 17, "#e8e7e3")
    s.render("fig_architecture")


def ontology():
    W = 1020
    s = Svg(W, 900)
    cw, ch = 240, 92
    cx = [20, 380, 740]
    ry = [56, 214, 372, 530, 688]

    def cls(c, r, name, en, color, extra=None):
        x, y = cx[c], ry[r]
        s.box(x, y, cw, ch, name, [en] + ([extra] if extra else []), color, title_size=22, size=16)
        return (x, y, cw, ch)

    def lab(x, y, t):
        tw = sum(17 if ord(c) > 127 else 9.2 for c in t)
        s.el.append(f'<rect x="{x - tw / 2 - 4}" y="{y - 16}" width="{tw + 8}" height="23" rx="4" fill="#ffffff"/>')
        s.text(x, y + 1, t, 17, INK2)

    obs, emi, kn, sit, ev = C["obs"], C["fus"], C["und"], C["gen"], C["ont"]
    node = cls(0, 0, "观测节点", "ObservationNode", obs)
    trk = cls(1, 0, "跟踪照射关系", "TrackingRelation", sit)
    evd = cls(2, 0, "证据", "Evidence", ev, "epistemicStatus")
    local = cls(0, 1, "局部辐射源", "LocalEmitter", emi)
    obj = cls(1, 1, "融合对象", "FusedObject", emi, "pulses · stateFrac")
    radar = cls(2, 1, "探测器型号", "RadarModel", kn, "rcsCandidates")
    pm = cls(2, 2, "平台型号", "PlatformModel", kn)
    scn = cls(0, 3, "场景", "Scenario", C["io"])
    track = cls(1, 3, "B 类航迹", "Track", obs, "domain · rcs · pattern")
    cat = cls(2, 3, "平台类别", "PlatformCategory", kn)
    area = cls(0, 4, "试验海区", "TestArea", C["io"])
    mot = cls(1, 4, "运动关系", "MotionRelation", sit)
    dom = cls(2, 4, "领域", "Domain", kn)
    pfx = cls(0, 2, "被动定位轨迹", "PassiveFix", sit, "phaseHit · speed")

    def R(n): return (n[0] + n[2], n[1] + n[3] / 2)
    def L(n): return (n[0], n[1] + n[3] / 2)
    def T(n): return (n[0] + n[2] / 2, n[1])
    def B(n): return (n[0] + n[2] / 2, n[1] + n[3])

    def e(p1, p2, t, dashed=False):
        s.arrow([p1, p2], dashed=dashed)
        if abs(p1[1] - p2[1]) < 1:  # 水平边：标签写在线上方，不遮挡箭头
            s.text((p1[0] + p2[0]) / 2, p1[1] - 9, t, 15, INK2)
        else:
            lab((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2, t)

    e(T(local), (B(node)[0], B(node)[1] + 2), "observedBy")
    e(R(local), (L(obj)[0] - 2, L(obj)[1]), "partOf")
    e(R(obj), (L(radar)[0] - 2, L(radar)[1]), "hasRadarModel")
    e(B(radar), (T(pm)[0], T(pm)[1] - 2), "mountedOn")
    e(B(pm), (T(cat)[0], T(cat)[1] - 2), "inCategory")
    e(B(cat), (T(dom)[0], T(dom)[1] - 2), "inDomain")
    e(B(trk), (T(obj)[0], T(obj)[1] - 2), "tracker")
    e(L(trk), (R(node)[0] + 2, R(node)[1]), "trackedNode")
    e(B(obj), (T(track)[0], T(track)[1] - 2), "linkedTrack")
    e(T(mot), (B(track)[0], B(track)[1] + 2), "relA / relB")
    e(L(track), (R(scn)[0] + 2, R(scn)[1]), "inScenario")
    e(B(scn), (T(area)[0], T(area)[1] - 2), "包含", dashed=True)
    e((obj[0] + 30, obj[1] + ch), (pfx[0] + cw - 30, pfx[1] - 2), "hasPassiveFix")
    e((pfx[0] + cw - 30, pfx[1] + ch), (track[0] + 30, track[1] - 2), "nearestTrack")
    # 证据（虚线）
    e(R(trk), (L(evd)[0] - 2, L(evd)[1]), "supportedBy", dashed=True)
    s.arrow([(obj[0] + cw - 40, obj[1]), (obj[0] + cw - 40, obj[1] - 30), (690, obj[1] - 30),
             (690, evd[1] + ch - 18), (evd[0] - 2, evd[1] + ch - 18)], dashed=True)
    s.arrow([B(mot), (B(mot)[0], 806), (1003, 806), (1003, evd[1] + ch / 2), (evd[0] + cw + 2, evd[1] + ch / 2)], dashed=True)
    lab(800, 806, "supportedBy（型号、关联、运动关系 → 证据）")
    # 图例
    lx = 20
    for t, col in (("观测与数据实体", obs), ("目标理解结果", emi), ("资料知识（D2 规则）", kn), ("态势关系", sit), ("证据", ev)):
        s.el.append(f'<rect x="{lx}" y="848" width="20" height="20" rx="3" fill="#fff" stroke="{col}" stroke-width="2.5"/>')
        s.text(lx + 28, 865, t, 17, INK2, anchor="start")
        lx += 28 + sum(17 if ord(c) > 127 else 9 for c in t) + 30
    s.text(20, 30, "实线：对象属性；虚线：证据挂接。epistemicStatus ∈ {观测事实, 资料知识, 推断, 假设}", 17, INK2, anchor="start")
    s.render("fig_ontology")


def dataflow():
    s = Svg(1000, 420)
    steps = [("A 类脉冲", ["8,600,592 个", "4 个观测节点"], C["io"]),
             ("CPI", ["133,612 个", "中位长度 62"], C["obs"]),
             ("发射 / 驻留组", ["79,888 次发射", "48,661 个驻留组"], C["obs"]),
             ("航迹锚定", ["89.0% 脉冲锚定", "12 条航迹判出型号"], C["fus"]),
             ("融合与纠错", ["11 个对象", "43 个局部簇"], C["fus"]),
             ("A-B 关联", ["9 个 matched", "2 个 no_match"], C["fus"]),
             ("本体 / 报告", ["1,556 个三元组", "含被动定位"], C["ont"])]
    w, h, gap = 220, 118, 30
    pos = [(20 + i * (w + gap), 20) for i in range(4)] + [(20 + (3 - i) * (w + gap), 190) for i in range(3)]
    for (t, ls, col), (x, y) in zip(steps, pos):
        s.box(x, y, w, h, t, ls, col, title_size=22, size=17)
    for i in range(3):
        s.arrow([(pos[i][0] + w, 79), (pos[i + 1][0] - 2, 79)])
    s.arrow([(pos[3][0] + w / 2, 138), (pos[4][0] + w / 2, 188)])
    for i in range(4, 6):
        s.arrow([(pos[i][0], 249), (pos[i + 1][0] + w + 2, 249)])
    s.text(20, 345, "纠错：时间互斥改归 92,048 脉冲、几何否决 3,468、到达相位改归 2,334；状态分布 搜索 77.9% / 粗跟 1.8% / 精跟 20.3%。", 17, INK2, anchor="start")
    s.text(20, 375, "16 条航迹中 9 条关联辐射源；2 个无航迹辐射源由被动定位给出位置与活动。", 17, INK2, anchor="start")
    s.text(20, 403, "容器内总耗时约 1 min（含读入与写出），结果表不依赖大模型。", 17, INK2, anchor="start")
    s.render("fig_dataflow")


def charts():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    blue, orange, aqua = "#2a78d6", "#eb6834", "#1baf7a"
    ink2, grid = "#52514e", "#e4e3df"

    def style(ax):
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("#b9b8b3")
        ax.tick_params(colors=ink2, labelsize=10)
        ax.grid(axis="x", color=grid, lw=0.8)
        ax.set_axisbelow(True)

    # 图：压力测试（最后一折模型 + 测试时段）
    names = ["完整航迹", "逐个去节点", "伪造航迹", "伴飞静默航迹", "节点改编号", "2% 杂散脉冲", "测向误差 +±1°",
             "航迹抽稀 1/3", "B 时间 +3 s", "B 时间 −6 s", "去掉一半航迹", "无 B 数据", "60 s 短场景"]
    cl = [0.9993, 0.9966, 0.9994, 0.9991, 0.9993, 0.9985, 0.9932, 0.9816, 0.9775, 0.9816, 0.9871, 0.9770, 0.9709]
    fu = [0.9993, 0.9985, 0.9993, 0.9988, 0.9993, 0.9984, 0.9955, 0.9938, 0.9930, 0.9938, 0.9945, 0.9933, 0.9630]
    st = [0.9590, 0.9410, 0.9590, 0.9589, 0.9590, 0.9235, 0.9472, 0.9435, 0.9430, 0.9381, 0.9540, 0.9488, 0.9356]
    fig, ax = plt.subplots(figsize=(8.6, 7.2), dpi=200)
    import numpy as np
    y = np.arange(len(names))[::-1]
    h = 0.26
    for off, vals, col, lab in ((h, cl, blue, "聚类逐簇 F1"), (0, fu, orange, "融合 ARI"), (-h, st, aqua, "状态宏 F1")):
        ax.barh(y + off, vals, height=h - 0.04, color=col, label=lab, edgecolor="white", linewidth=1)
    for yy, v in zip(y, cl):
        ax.text(v + 0.002, yy + h, f"{v:.3f}", va="center", fontsize=8.5, color=ink2)
    ax.set_yticks(y, names)
    ax.set_xlim(0.90, 1.008)
    style(ax)
    ax.legend(loc="lower left", frameon=False, fontsize=9.5, ncol=3, bbox_to_anchor=(0, -0.11))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_stress.png"), bbox_inches="tight")
    plt.close(fig)

    print("charts done")


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    architecture()
    ontology()
    dataflow()
    charts()
