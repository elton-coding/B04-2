"""电磁目标本体：概念层（TBox）+ 场景实例层（ABox），导出 Turtle / JSON-LD / 交互式图谱 HTML。

每条断言带认识论状态（观测事实 / 资料知识 / 推断 / 假设）与证据节点，满足 D4"区分事实、知识、推断与假设"。
"""
from __future__ import annotations

import json
import os

from rdflib import RDF, RDFS, XSD, BNode, Graph, Literal, Namespace, OWL

ONT = Namespace("urn:b04:onto#")

CLASSES = {
    "Scenario": "场景", "ObservationNode": "本方观测节点", "LocalEmitter": "局部辐射源（节点内聚类）",
    "FusedObject": "融合对象（跨平台统一辐射源）", "RadarModel": "探测器型号", "PlatformModel": "平台型号",
    "PlatformCategory": "平台类别", "Domain": "领域", "Track": "B 类航迹", "TrackingRelation": "跟踪照射关系",
    "MotionRelation": "运动关系（伴随/接近）", "Evidence": "证据", "TestArea": "试验海区",
}
OBJ_PROPS = {
    "inScenario": "属于场景", "observedBy": "被节点观测", "partOf": "归并为", "hasRadarModel": "挂载型号",
    "mountedOn": "型号对应平台型号", "inCategory": "属于类别", "inDomain": "所在领域", "linkedTrack": "关联航迹",
    "tracker": "跟踪方", "trackedNode": "被跟踪节点", "relA": "关系一方", "relB": "关系另一方", "supportedBy": "证据支持",
}


def _lit(v):
    if isinstance(v, bool):
        return Literal(v, datatype=XSD.boolean)
    if isinstance(v, int):
        return Literal(v, datatype=XSD.integer)
    if isinstance(v, float):
        return Literal(round(v, 4), datatype=XSD.decimal)
    return Literal(str(v))


def build_graph(facts: dict, knowledge: dict) -> Graph:
    g = Graph()
    g.bind("b04", ONT)
    for c, lab in CLASSES.items():
        g.add((ONT[c], RDF.type, OWL.Class))
        g.add((ONT[c], RDFS.label, Literal(lab, lang="zh")))
    for p, lab in OBJ_PROPS.items():
        g.add((ONT[p], RDF.type, OWL.ObjectProperty))
        g.add((ONT[p], RDFS.label, Literal(lab, lang="zh")))
    sc = facts["scenario"]
    sid = f"{sc['scenario_id']}_{sc['data_version']}"
    I = Namespace(f"urn:b04:{sid}:")

    def ev(subject, text, status, **kv):
        e = BNode()
        g.add((e, RDF.type, ONT.Evidence))
        g.add((e, ONT.text, Literal(text, lang="zh")))
        g.add((e, ONT.epistemicStatus, Literal(status, lang="zh")))
        for k, v in kv.items():
            g.add((e, ONT[k], _lit(v)))
        g.add((subject, ONT.supportedBy, e))

    S = I["scenario"]
    g.add((S, RDF.type, ONT.Scenario))
    g.add((S, RDFS.label, Literal(sid)))
    for k in ("base_time", "duration_s", "pulses"):
        g.add((S, ONT[k], _lit(sc[k])))
    # 资料知识：型号 → 平台型号 → 类别 → 领域（D2【规则】）
    for code, m in knowledge.get("models", {}).items():
        R, PM = ONT[f"model_{code}"], ONT[f"pm_{m['platform_model']}"]
        C, D = ONT[f"cat_{m['category']}"], ONT[f"dom_{m['domain']}"]
        g.add((R, RDF.type, ONT.RadarModel)); g.add((R, RDFS.label, Literal(code)))
        g.add((PM, RDF.type, ONT.PlatformModel)); g.add((PM, RDFS.label, Literal(m["platform_model"], lang="zh")))
        g.add((C, RDF.type, ONT.PlatformCategory)); g.add((C, RDFS.label, Literal(m["category"], lang="zh")))
        g.add((D, RDF.type, ONT.Domain)); g.add((D, RDFS.label, Literal({"air": "空中", "sea": "海面"}[m["domain"]], lang="zh")))
        g.add((R, ONT.mountedOn, PM)); g.add((PM, ONT.inCategory, C)); g.add((C, ONT.inDomain, D))
        g.add((R, ONT.rcsCandidates, Literal(",".join(m.get("rcs", [])))))
    for n in facts["nodes"]:
        N = I[f"node_{n['platform_id']}"]
        g.add((N, RDF.type, ONT.ObservationNode)); g.add((N, RDFS.label, Literal(f"节点 {n['platform_id']}", lang="zh")))
        g.add((N, ONT.inScenario, S))
        for k in ("domain", "pulses", "t_first_s", "t_last_s", "speed_med", "pattern"):
            g.add((N, ONT[k], _lit(n[k])))
    for tid, t in facts["tracks"].items():
        T = I[f"track_{tid}"]
        g.add((T, RDF.type, ONT.Track)); g.add((T, RDFS.label, Literal(f"航迹 {tid}", lang="zh")))
        g.add((T, ONT.inScenario, S))
        for k in ("domain", "rcs", "height_m", "speed_med", "pattern", "in_area_frac"):
            if k in t:
                g.add((T, ONT[k], _lit(t[k])))
    conf = facts.get("type_confidence", {})
    for o in facts["objects"]:
        O = I[f"object_{o['object_id']}"]
        g.add((O, RDF.type, ONT.FusedObject)); g.add((O, RDFS.label, Literal(f"对象 {o['object_id']}", lang="zh")))
        g.add((O, ONT.inScenario, S))
        g.add((O, ONT.hasRadarModel, ONT[f"model_{o['type']}"]))
        for k in ("pulses", "t_first_s", "t_last_s"):
            g.add((O, ONT[k], _lit(o[k])))
        c = conf.get(o["object_id"]) or conf.get(str(o["object_id"]))
        if c:
            ev(O, f"型号 {o['type']}：多 CPI 波形指纹似然累积，较次优 {c['second']} 平均每 CPI 高 {c['margin_per_cpi']} nats",
               "推断", margin=float(c["margin_per_cpi"]))
        for st, v in o["state_frac"].items():
            g.add((O, ONT[f"stateFrac_{st}"], _lit(float(v))))
        for nd in o["nodes"]:
            LE = I[f"local_{nd['local_cluster_id']}"]
            g.add((LE, RDF.type, ONT.LocalEmitter)); g.add((LE, RDFS.label, Literal(f"局部簇 {nd['local_cluster_id']}", lang="zh")))
            g.add((LE, ONT.observedBy, I[f"node_{nd['platform_id']}"])); g.add((LE, ONT.partOf, O))
            for k in ("pulses", "t_first_s", "t_last_s"):
                g.add((LE, ONT[k], _lit(nd[k])))
        if o["track_id"] is not None:
            T = I[f"track_{o['track_id']}"]
            g.add((O, ONT.linkedTrack, T))
            best = (o.get("link_candidates") or [{}])[0]
            ev(O, f"与航迹 {o['track_id']} 方位命中率 {best.get('hit')}，残差中位数 {best.get('med_res_deg')}°", "推断",
               hit=float(best.get("hit", 0)), med_res_deg=float(best.get("med_res_deg", 0)))
    for r in facts.get("tracked_nodes", []):
        TR = I[f"tracking_{r['object_id']}_{r['node']}"]
        g.add((TR, RDF.type, ONT.TrackingRelation))
        g.add((TR, ONT.tracker, I[f"object_{r['object_id']}"])); g.add((TR, ONT.trackedNode, I[f"node_{r['node']}"]))
        for k in ("t_first_s", "t_last_s", "dwells", "coarse", "fine"):
            g.add((TR, ONT[k], _lit(r[k])))
        ev(TR, "跟踪驻留中该节点接收幅度显著高于该对象常态（主瓣持续照射）", "推断", dwells=int(r["dwells"]))
    for r in facts.get("track_relations", []):
        MR = I[f"rel_{r['track_a']}_{r['track_b']}"]
        g.add((MR, RDF.type, ONT.MotionRelation)); g.add((MR, RDFS.label, Literal(r["relation"], lang="zh")))
        g.add((MR, ONT.relA, I[f"track_{r['track_a']}"])); g.add((MR, ONT.relB, I[f"track_{r['track_b']}"]))
        for k in ("dist_med_m", "dist_min_m", "rel_speed_med", "near_frac"):
            g.add((MR, ONT[k], _lit(r[k])))
        ev(MR, "B 类航迹位置序列计算的距离与相对速度", "观测事实")
    return g


def graph_html(facts: dict, path: str):
    """ECharts 力导向图：节点 / 局部簇 / 对象 / 型号 / 航迹 及其关系。"""
    nodes, links, cats = [], [], ["观测节点", "局部簇", "融合对象", "型号", "航迹"]
    seen = set()

    def add(id_, name, cat, size, tip=""):
        if id_ not in seen:
            seen.add(id_)
            nodes.append({"id": id_, "name": name, "category": cat, "symbolSize": size, "value": tip})
    for n in facts["nodes"]:
        add(f"n{n['platform_id']}", f"节点{n['platform_id']}", 0, 34, f"{n['domain']} {n['pattern']}")
    for tid, t in facts["tracks"].items():
        add(f"t{tid}", f"航迹{tid}", 4, 20, f"{t['domain']} {t['rcs']} {t['pattern']}")
    for o in facts["objects"]:
        add(f"o{o['object_id']}", f"对象{o['object_id']}·{o['type']}", 2, 30, f"{o['category']} {o['pulses']} 脉冲")
        add(f"m{o['type']}", f"{o['type']} {o['platform_model']}", 3, 24, o["category"])
        links.append({"source": f"o{o['object_id']}", "target": f"m{o['type']}", "label": "挂载型号"})
        for nd in o["nodes"]:
            add(f"l{nd['local_cluster_id']}", f"簇{nd['local_cluster_id']}", 1, 10, f"{nd['pulses']} 脉冲")
            links.append({"source": f"l{nd['local_cluster_id']}", "target": f"o{o['object_id']}", "label": "归并"})
            links.append({"source": f"l{nd['local_cluster_id']}", "target": f"n{nd['platform_id']}", "label": "观测于"})
        if o["track_id"] is not None:
            links.append({"source": f"o{o['object_id']}", "target": f"t{o['track_id']}", "label": "关联"})
    for r in facts.get("tracked_nodes", []):
        links.append({"source": f"o{r['object_id']}", "target": f"n{r['node']}", "label": f"跟踪照射({r['dwells']})",
                      "lineStyle": {"type": "dashed", "color": "#c0392b"}})
    for r in facts.get("track_relations", []):
        if r["relation"] == "伴随/编队":
            links.append({"source": f"t{r['track_a']}", "target": f"t{r['track_b']}", "label": "伴随",
                          "lineStyle": {"color": "#2e86de", "width": 2}})
    data = json.dumps({"nodes": nodes, "links": links, "categories": [{"name": c} for c in cats]}, ensure_ascii=False)
    sc = facts["scenario"]
    html = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>电磁目标图谱 {sc['scenario_id']}_{sc['data_version']}</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<script src="https://cdnjs.cloudflare.com/ajax/libs/echarts/5.5.0/echarts.min.js"></script>
<style>html,body{{margin:0;height:100%;background:#fff;font-family:sans-serif}}#g{{width:100%;height:100%}}</style></head>
<body><div id="g"></div><script>
const D={data};const c=echarts.init(document.getElementById('g'));
c.setOption({{title:{{text:'电磁目标本体图谱 {sc['scenario_id']}_{sc['data_version']}',left:10,top:10}},tooltip:{{formatter:p=>p.dataType==='node'?p.data.name+'<br>'+p.data.value:p.data.label}},
legend:{{data:D.categories.map(x=>x.name),top:40}},series:[{{type:'graph',layout:'force',roam:true,draggable:true,
label:{{show:true,fontSize:10}},force:{{repulsion:220,edgeLength:[40,120]}},categories:D.categories,data:D.nodes,
links:D.links,edgeLabel:{{show:false}},lineStyle:{{opacity:.7,curveness:.1}}}}]}});
window.onresize=()=>c.resize();</script></body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


def export(facts: dict, knowledge: dict, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    sc = facts["scenario"]
    stem = os.path.join(out_dir, f"{sc['scenario_id']}_{sc['data_version']}_ontology")
    g = build_graph(facts, knowledge)
    g.serialize(stem + ".ttl", format="turtle")
    g.serialize(stem + ".jsonld", format="json-ld", indent=1)
    graph_html(facts, stem + "_graph.html")
    return {"triples": len(g), "ttl": stem + ".ttl", "jsonld": stem + ".jsonld", "html": stem + "_graph.html"}
