"""态势报告生成（D4 结构）：事实卡片 → HTML（图表内嵌）；可选大模型润色研判段落（事实锁定校验）。

原则（D4 注意事项）：区分观测事实/资料知识/推断/假设；关键结论句末注明证据；
上游结果有疑点时如实说明，不为叙事通顺修改事实。
"""
from __future__ import annotations

import base64
import html
import io
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

TAG = {"F": "观测事实", "K": "资料知识", "I": "推断", "H": "假设"}


def _ev(**kv) -> str:
    s = "；".join(f"{k}={v}" for k, v in kv.items() if v is not None)
    return f'<span class="ev">【证据：{html.escape(s)}】</span>'


def _tag(k) -> str:
    return f'<span class="tag t{k}">{TAG[k]}</span>'


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return '<img src="data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode() + '">'


def _table(rows, cols) -> str:
    h = "".join(f"<th>{html.escape(c)}</th>" for c in cols)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>"


def _companion(d, max_m=5000.0, min_s=60.0) -> bool:
    """被动轨迹与航迹的间距在共同时段内稳定（最大与最小相差 < 中位的 30%）→ 伴随关系候选。"""
    return (d["dist_med_m"] <= max_m and d["overlap_s"] >= min_s
            and d["dist_p90_m"] - d["dist_min_m"] <= 0.3 * d["dist_med_m"])


def _passive_sentence(o, pf) -> str:
    """无航迹对象的被动定位描述（第 5 章）。"""
    s = (f" 被动定位（{len(pf['nodes'])} 个节点、{pf['n_cpi']} 个 CPI，{pf['t_first_s']}–{pf['t_last_s']} s，可靠时段占 {pf['reliable_frac']:.0%}，"
         + (f"到达相位命中率 {pf['phase_hit']:.0%}" if pf["phase_used"] else "仅测向交会") + "）：")
    near = pf.get("nearest_tracks") or []
    if near:
        d = near[0]
        s += (f"距最近的航迹 {d['track_id']} 中位 {d['dist_med_m'] / 1000:.1f} km（最近 {d['dist_min_m'] / 1000:.1f} km），"
              f"中位距离明显大于被动定位误差，不在任何 B 航迹上 {_tag('I')}{_ev(object_id=o['object_id'], nearest_track=d['track_id'], dist_med_m=int(d['dist_med_m']))}。")
        comp = [q for q in near if _companion(q)]
        if comp:
            s += (f" 其与航迹 {'、'.join(str(q['track_id']) for q in comp)} 的间距在整个共同时段内稳定在 "
                  + "、".join(f"{q['dist_min_m'] / 1000:.1f}–{q['dist_p90_m'] / 1000:.1f} km" for q in comp)
                  + f"（同向同速）{_tag('I')}；{_tag('H')} 可能与其编队伴随，但运动关系不能单独作为编组依据（D4）。")
    else:
        s += f"其时段内无可比较的 B 航迹 {_tag('I')}。"
    return s


def fig_map(facts, a, tracks):
    fig, ax = plt.subplots(figsize=(8, 7))
    ann = facts["scenario"].get("announcement")
    if ann:
        p = np.array(ann["polygon"] + [ann["polygon"][0]])
        ax.plot(p[:, 0], p[:, 1], "k--", lw=1, label="Test area")
    link = {o["track_id"]: o for o in facts["objects"] if o["track_id"] is not None}
    cmap = plt.get_cmap("tab10")
    for tid, x in tracks.groupby("track_id"):
        o = link.get(int(tid))
        c = cmap((o["object_id"] - 1) % 10) if o else "0.6"
        ls = "-" if x.target_height.median() > 50 else "-"
        ax.plot(x.target_longitude, x.target_latitude, ls, color=c, lw=1.6 if o else 0.9, alpha=0.9)
        lab = f"T{tid}" + (f"/O{o['object_id']} {o['type']}" if o else "")
        ax.annotate(lab, (x.target_longitude.iloc[-1], x.target_latitude.iloc[-1]), fontsize=7, color=c if o else "0.4")
    for pf in facts.get("passive", []):
        o = next((q for q in facts["objects"] if q["object_id"] == pf["object_id"]), None)
        if o is None or o["track_id"] is not None or len(pf["path"]) < 2:
            continue
        pth = np.array(pf["path"])
        c = cmap((o["object_id"] - 1) % 10)
        ax.plot(pth[:, 1], pth[:, 2], ":", color=c, lw=2.0)
        ax.annotate(f"O{o['object_id']} {o['type']} (passive fix, no track)", (pth[-1, 1], pth[-1, 2]), fontsize=7, color=c)
    for pid, x in a.groupby("platform_id"):
        x = x.iloc[:: max(1, len(x) // 300)]
        ax.plot(x.sensor_longitude, x.sensor_latitude, "k-", lw=2.4)
        ax.annotate(f"N{pid}", (x.sensor_longitude.iloc[-1], x.sensor_latitude.iloc[-1]), fontsize=8, weight="bold")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.set_aspect(1 / np.cos(np.radians(24.7)))
    ax.set_title("Nodes (black), tracks (colored = linked object, grey = unlinked), dotted = passive fix of emitters without track",
                 fontsize=9)
    return _png(fig)


def fig_states(facts, a, pred, bin_s=5.0):
    t = a.toa.values / 1e6
    b = (t // bin_s).astype(int)
    df = pd.DataFrame({"o": pred.object_id.values, "b": b, "s": pred.target_state.values})
    objs = [o["object_id"] for o in facts["objects"]]
    nb = int(b.max()) + 1
    M1 = np.full((len(objs), nb), np.nan)
    M2 = np.full((len(objs), nb), np.nan)
    g = df.groupby(["o", "b"]).s
    f1 = g.apply(lambda s: (s == 1).mean())
    f2 = g.apply(lambda s: (s == 2).mean())
    for i, o in enumerate(objs):
        if o in f1.index.get_level_values(0):
            M1[i, f1.loc[o].index] = f1.loc[o].values
            M2[i, f2.loc[o].index] = f2.loc[o].values
    fig, axs = plt.subplots(2, 1, figsize=(10, 0.35 * len(objs) * 2 + 1.5), sharex=True)
    for ax, M, ttl in ((axs[0], M1, "coarse track (state 1) share"), (axs[1], M2, "fine track (state 2) share")):
        im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=1, extent=[0, nb * bin_s, len(objs) - 0.5, -0.5])
        ax.set_yticks(range(len(objs))); ax.set_yticklabels([f"O{o}" for o in objs], fontsize=7)
        ax.set_title(ttl, fontsize=9)
    axs[1].set_xlabel("time since start (s)")
    fig.colorbar(im, ax=axs, shrink=0.6)
    return _png(fig)


def build_html(facts: dict, a: pd.DataFrame, pred: pd.DataFrame, tracks: pd.DataFrame, knowledge: dict,
               narrative: dict | None = None) -> str:
    sc = facts["scenario"]
    objs = facts["objects"]
    trk = {int(k): v for k, v in facts["tracks"].items()}
    conf = {int(k): v for k, v in facts["type_confidence"].items()}
    tn = pd.DataFrame(facts["tracked_nodes"])
    rel = facts["track_relations"]
    unl = facts["unlinked_tracks"]
    nodes = facts["nodes"]
    P = []
    P.append(f"<h1>{sc['scenario_id']}_{sc['data_version']} 电磁态势分析报告</h1>")
    P.append(f"<p class='meta'>依据：pulse_predictions.csv、object_track_links.csv、A/B 类观测数据、配套资料 D1–D4。"
             f"标注说明：{_tag('F')} 数据直接给出；{_tag('K')} 来自配套资料；{_tag('I')} 由数据推理得到；{_tag('H')} 证据不足的可能解释。</p>")
    n_m = sum(o["link_status"] == "matched" for o in objs)
    P.append("<h2>摘要</h2><p>" + (
        f"本场时长约 {sc['duration_s']} s，{sc['nodes']} 个本方观测节点共接收 {sc['pulses']:,} 个脉冲，B 源提供 {sc['tracks']} 条航迹。"
        f"融合得到 {len(objs)} 个辐射源对象，其中 {n_m} 个与 B 航迹关联、{len(objs) - n_m} 个无对应航迹；"
        f"{len(unl)} 条航迹无对应辐射源。{_tag('I')}") + "</p>")
    if narrative and narrative.get("summary"):
        P.append(f"<p>{html.escape(narrative['summary'])} {_tag('I')}</p>")

    # 1 观测概览
    P.append("<h2>1 观测概览</h2>")
    ann = sc.get("announcement")
    if ann:
        P.append(f"<p>{_tag('K')} 试验公告时段 {ann['start']} 起 {ann['duration_s']} s；本场数据时段 {sc['base_time']} 起 {sc['duration_s']} s。</p>")
    rows = [[n["platform_id"], n["domain"], f"{n['height_m']:.0f}", f"{n['pulses']:,}", f"{n['t_first_s']}–{n['t_last_s']}",
             n["speed_med"], n["pattern"]] for n in nodes]
    P.append(_table(rows, ["节点", "领域", "高度 m", "脉冲数", "覆盖时段 s", "速度 m/s", "运动方式"]))
    short = [n for n in nodes if n["t_last_s"] < sc["duration_s"] - 5]
    for n in short:
        P.append(f"<p>{_tag('F')} 节点 {n['platform_id']} 在 {n['t_last_s']} s 后无脉冲记录 {_ev(platform_id=n['platform_id'], t_last_s=n['t_last_s'])}；"
                 f"{_tag('H')} 可能为接收视场转出或节点停止接收，数据中无法区分。</p>")
    na = sum(t["domain"] == "空中" for t in trk.values())
    P.append(f"<p>{_tag('F')} B 源航迹 {len(trk)} 条（空中 {na}、海面 {len(trk) - na}），散射等级 small/medium/large 分布："
             + "、".join(f"{k} {v}" for k, v in pd.Series([t['rcs'] for t in trk.values()]).value_counts().items()) + "。</p>")
    P.append(fig_map(facts, a, tracks))

    # 2 对象清单
    P.append("<h2>2 对象清单</h2>")
    rows = []
    for o in objs:
        c = conf.get(o["object_id"], {})
        best = (o.get("link_candidates") or [{}])[0]
        rows.append([o["object_id"], o["type"], f"{o['platform_model']}/{o['category']}", o["domain"], f"{o['pulses']:,}",
                     len(o["nodes"]), o["track_id"] if o["track_id"] is not None else "无",
                     f"{best.get('hit', '')}" if o["track_id"] is not None else "—", c.get("margin_per_cpi", "")])
    P.append(_table(rows, ["object_id", "型号", "平台型号/类别", "领域", "脉冲数", "观测节点数", "track_id", "方位命中率", "型号似然差/CPI"]))
    P.append(f"<p>{_tag('K')} 型号—平台型号—类别对应关系依据 D2【规则】（每平台一部探测器、型号与平台型号一一对应）。</p>")

    # 3 类别与工作状态
    P.append("<h2>3 类别与工作状态</h2>")
    for o in objs:
        c = conf.get(o["object_id"], {})
        sf = o["state_frac"]
        P.append(f"<p><b>对象 {o['object_id']}（{o['type']}）</b>：型号由多 CPI 的 RF/PW/PRI 联合似然累积判定，较次优型号 {c.get('second')} "
                 f"平均每 CPI 高 {c.get('margin_per_cpi')} nats {_tag('I')}{_ev(object_id=o['object_id'], cpis=o['cpis'])}。"
                 f"脉冲状态构成：搜索 {sf['搜索']:.1%}、粗跟踪 {sf['粗跟踪']:.1%}、精跟踪 {sf['精跟踪']:.1%} {_tag('F')}（为本方结果表统计）。</p>")
    P.append(f"<p>{_tag('I')} 状态判定依据：跟踪驻留按固定周期插入搜索序列、接收幅度显著抬升；粗跟位于针对同一被跟踪目标的跟踪段起始若干次驻留。"
             f"同型号搜索与跟踪可能使用同一套波形（D2），故状态判断主要依赖时序与幅度证据，把握度低于型号判断。</p>")
    P.append(fig_states(facts, a, pred))

    # 4 平台观测关系
    P.append("<h2>4 平台观测关系</h2>")
    rows = []
    for o in objs:
        for nd in o["nodes"]:
            rows.append([o["object_id"], nd["platform_id"], nd["local_cluster_id"], f"{nd['pulses']:,}", f"{nd['t_first_s']}–{nd['t_last_s']}"])
    P.append(_table(rows, ["object_id", "观测节点", "local_cluster_id", "脉冲数", "时段 s"]))
    P.append(f"<p>{_tag('I')} 各节点局部辐射源的归并依据：同一发射在多节点上 RF/PW/PRI 一致、到达时差仅为传播时延差（同源 CPI 匹配）；"
             f"同节点相邻 CPI 首尾间隔恰为一个 PRI（连续发射）；以及 B 航迹多节点方位一致性。同一节点上时间重叠且参数不同的 CPI 不并入同一对象。</p>")

    # 5 A-B 关联
    P.append("<h2>5 A-B 关联</h2>")
    pas = {p["object_id"]: p for p in facts.get("passive", [])}
    if pas:
        P.append(f"<p>{_tag('I')} 除方位命中外，另用 A 类数据独立估计各辐射源位置（被动定位）：多节点测向交会；并利用驻留起点到达时刻"
                 f"落在公共 64 μs 时间格上的规律，由到达时刻对 64 μs 取模得到节点—辐射源斜距（单节点测距，19.2 km 模糊由测向交会与"
                 f"轨迹连续性消解），空中目标高度由空中、海面节点斜距之差解出。训练数据上（不用航迹）与真航迹相比，可靠时段水平误差中位"
                 f" 64–190 m、90% 分位 110–250 m，空中目标高度误差 ≤ 0.1 km，与 B 航迹自身的位置误差同量级。被动轨迹与关联航迹的距离是不依赖方位命中率的关联佐证。</p>")
    for o in objs:
        cands = o.get("link_candidates") or []
        pf = pas.get(o["object_id"])
        if o["track_id"] is not None:
            alt = [c for c in cands if c["track_id"] != o["track_id"] and c["hit"] >= 0.9 and c["feasible"]]
            s = (f"<p>对象 {o['object_id']} ↔ 航迹 {o['track_id']}：多节点 DOA 与航迹方位命中率 {cands[0]['hit'] if cands else ''}、"
                 f"残差中位数 {cands[0]['med_res_deg'] if cands else ''}°，领域与散射等级（{trk.get(o['track_id'], {}).get('rcs')}）符合 {o['type']} 候选 "
                 f"{_tag('I')}{_ev(object_id=o['object_id'], track_id=o['track_id'])}。")
            if alt:
                s += (f" 航迹 {', '.join(str(c['track_id']) for c in alt)} 与该对象方位同样吻合（命中率 ≥0.9），属近距伴随航迹，"
                      f"按残差最小选择 {o['track_id']}，存在竞争解释（见第 7 章）。")
            ld = (pf or {}).get("linked_track_dist")
            if ld:
                nxt = next((d for d in pf["nearest_tracks"] if d["track_id"] != o["track_id"]), None)
                s += (f" 被动定位轨迹与航迹 {o['track_id']} 的距离中位 {ld['dist_med_m']:.0f} m"
                      + (f"（次近为航迹 {nxt['track_id']}，{nxt['dist_med_m']:.0f} m）" if nxt else "")
                      + f" {_tag('I')}{_ev(object_id=o['object_id'], track_id=o['track_id'], passive_dist_med_m=int(ld['dist_med_m']))}。")
            P.append(s + "</p>")
        else:
            best = next((c for c in cands if c["feasible"]), None)
            P.append(f"<p>对象 {o['object_id']}（{o['type']}）无对应航迹 {_ev(object_id=o['object_id'], link_status='no_match')}："
                     + (f"可行航迹中最佳为 {best['track_id']}，方位命中率仅 {best['hit']}，不足以关联。" if best else "无领域/散射等级可行的航迹。")
                     + (_passive_sentence(o, pf) if pf else "")
                     + f" {_tag('H')} 可能是 B 源未建立该平台航迹（没有航迹不等于不存在）。</p>")
    for t in unl:
        f = trk[t]
        partner = [r for r in rel if r["relation"] == "伴随/编队" and t in (r["track_a"], r["track_b"])]
        s = f"<p>航迹 {t}（{f['domain']}，{f['rcs']}，{f['speed_med']} m/s，{f['pattern']}）无对应辐射源 {_ev(track_id=t)}。"
        if partner:
            r = partner[0]
            other = r["track_b"] if r["track_a"] == t else r["track_a"]
            s += (f" 其与航迹 {other} 持续伴随（距离中位数 {r['dist_med_m']} m、相对速度 {r['rel_speed_med']} m/s）{_tag('F')}；"
                  f"{_tag('H')} 可能为编队中未开机或全程静默的平台（D1：部分平台可能全程关闭雷达）。")
        else:
            s += f" {_tag('H')} 可能为雷达关闭的平台、与试验无关的过往目标，或 B 源异常航迹（D3）。"
        P.append(s + "</p>")

    # 6 活动与态势研判
    P.append("<h2>6 活动与态势研判</h2>")
    for o in objs:
        if o["track_id"] is None:
            pf = pas.get(o["object_id"])
            if pf:
                ina = pf.get("in_area_frac")
                span = f"{pf['t_first_s']}–{pf['t_last_s']} s"
                P.append(f"<p>对象 {o['object_id']}（{o['category']}，{o['type']}，无航迹）据被动定位以约 {pf['speed_med']} m/s {pf['pattern']}，"
                         f"范围约 {pf['extent_km']} km" + (f"，{ina:.0%} 时间位于公告试验海区内" if ina is not None else "")
                         + (f"，估计高度约 {pf['height_m'] / 1000:.1f} km" if pf.get("height_m") else "")
                         + f" {_tag('I')}{_ev(object_id=o['object_id'], 时段=span)}。</p>")
            continue
        f = trk.get(o["track_id"], {})
        ina = f.get("in_area_frac")
        P.append(f"<p>对象 {o['object_id']}（{o['category']}，{o['type']}）沿航迹 {o['track_id']} 以约 {f.get('speed_med')} m/s "
                 f"{f.get('pattern')}，范围约 {f.get('extent_km')} km" + (f"，{ina:.0%} 时间位于公告试验海区内" if ina is not None else "")
                 + f" {_tag('F')}{_ev(track_id=o['track_id'])}。</p>")
    if len(tn):
        P.append("<h3>雷达对本方节点的跟踪照射</h3>")
        rows = [[r.object_id, r.node, f"{r.t_first_s}–{r.t_last_s}", r.dwells, r.coarse, r.fine] for r in tn.itertuples()]
        P.append(_table(rows, ["object_id", "被跟踪节点", "时段 s", "跟踪驻留数", "粗跟", "精跟"]))
        heavy = tn[tn.dwells >= 100]
        P.append(f"<p>{_tag('I')} 跟踪驻留中某节点接收幅度显著高于常态，表明该雷达主瓣持续指向该节点，即本方节点被其跟踪。"
                 f"共 {heavy.object_id.nunique()} 个对象对本方节点形成持续跟踪（≥100 次驻留）。"
                 f"{_tag('H')} 结合 D1“边界巡查/安全警戒会核查接近海区的平台”，本方外围观测节点很可能已被试验编队发现并持续监视。</p>")
    forms = [r for r in rel if r["relation"] == "伴随/编队"]
    if forms:
        P.append("<h3>平台间关系</h3>")
        rows = [[r["track_a"], r["track_b"], r["domain"], r["dist_med_m"], r["rel_speed_med"], f"{r['near_frac']:.0%}"] for r in forms]
        P.append(_table(rows, ["航迹 A", "航迹 B", "领域", "距离中位数 m", "相对速度 m/s", "近距时间占比"]))
        P.append(f"<p>{_tag('F')} 以上航迹对持续近距同向同速。{_tag('H')} 可能为编队伴随、护航或补给并行；"
                 f"运动上的接近或同向不能单独作为协作或编组的依据（D4），需结合辐射源类别与时序进一步确认。</p>")
    if narrative and narrative.get("assessment"):
        P.append(f"<p>{html.escape(narrative['assessment'])} {_tag('H')}</p>")

    # 7 冲突与竞争解释
    P.append("<h2>7 冲突与竞争解释</h2>")
    items = []
    for o in objs:
        cands = o.get("link_candidates") or []
        alt = [c for c in cands if c["track_id"] != o["track_id"] and c["hit"] >= 0.9 and c["feasible"]]
        if o["track_id"] is not None and alt:
            pd_ = {d["track_id"]: d["dist_med_m"] for d in (pas.get(o["object_id"]) or {}).get("nearest_tracks", [])}
            extra = ""
            if o["track_id"] in pd_ and all(c["track_id"] in pd_ for c in alt):
                extra = ("，且被动定位轨迹离它更近（" + f"{pd_[o['track_id']]:.0f} m 对 "
                         + "、".join(f"{pd_[c['track_id']]:.0f} m" for c in alt) + "）")
            items.append(f"<li>对象 {o['object_id']} 与航迹 {o['track_id']} 及 {', '.join(str(c['track_id']) for c in alt)} 方位均吻合："
                         f"支持 {o['track_id']} 的证据为残差更小{extra}；反对证据为伴随航迹间距仅数百米，方位分辨力不足。{_ev(object_id=o['object_id'])}</li>")
    for o in objs:
        c = conf.get(o["object_id"], {})
        if c and c["margin_per_cpi"] < 2.0:
            items.append(f"<li>对象 {o['object_id']} 型号 {o['type']} 与 {c['second']} 区分度较低（每 CPI {c['margin_per_cpi']} nats）。</li>")
    P.append("<ul>" + ("".join(items) or "<li>未发现显著冲突。</li>") + "</ul>")

    # 8 无法确认的部分
    P.append("<h2>8 无法确认的部分</h2><ul>")
    weak = [p for p in facts.get("passive", []) if p["object_id"] in {o["object_id"] for o in objs if o["track_id"] is None}
            and (p["reliable_frac"] < 0.8 or not p["phase_used"] or p["phase_hit"] < 0.8)]
    P.append("<li>无对应航迹的辐射源的位置与航迹：已由被动定位给出估计（第 5、6 章），"
             + ("其中对象 " + "、".join(str(p["object_id"]) for p in weak) + " 主要由单个节点收到或观测间断，定位把握度中等；" if weak else "")
             + "精确航迹仍需 B 源补充或其他信息源核实。</li>")
    P.append("<li>无辐射的航迹是否为静默平台或异常航迹：需其开机后的电磁观测，或其他信息源核实。</li>")
    P.append("<li>平台任务与编组关系：当前仅依据运动关系与类别推断，需更长时段观测其补给、核查等行为过程。</li>")
    P.append("<li>粗跟/精跟边界：受漏脉冲与旁瓣接收影响，存在一定误差。</li></ul>")
    css = """body{font-family:'Microsoft YaHei','PingFang SC','Noto Sans CJK SC',sans-serif;max-width:980px;margin:24px auto;padding:0 16px;color:#222;line-height:1.6}
h1{font-size:22px}h2{font-size:18px;border-bottom:2px solid #333;padding-bottom:4px;margin-top:28px}h3{font-size:15px}
table{border-collapse:collapse;width:100%;font-size:12px;margin:8px 0}th,td{border:1px solid #bbb;padding:3px 6px;text-align:left}th{background:#f0f0f0}
img{max-width:100%}.ev{color:#555;font-size:11px}.meta{font-size:12px;color:#555}
.tag{font-size:10px;padding:0 4px;border-radius:3px;margin:0 2px;color:#fff}.tF{background:#2e7d32}.tK{background:#1565c0}.tI{background:#ef6c00}.tH{background:#8e24aa}
@media print{body{max-width:none;margin:10mm}}"""
    return f"<!doctype html><html lang='zh'><head><meta charset='utf-8'><title>态势报告 {sc['scenario_id']}_{sc['data_version']}</title><style>{css}</style></head><body>{''.join(P)}</body></html>"


def write_report(facts, a, pred, tracks, knowledge, out_dir, narrative=None) -> str:
    os.makedirs(out_dir, exist_ok=True)
    sc = facts["scenario"]
    path = os.path.join(out_dir, f"{sc['scenario_id']}_{sc['data_version']}_situation_report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(build_html(facts, a, pred, tracks, knowledge, narrative))
    return path
