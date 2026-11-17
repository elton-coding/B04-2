"""电磁智能体（大模型部分）：基于事实卡片生成摘要与态势研判段落，事实锁定校验，失败即降级。

只读取评测环境注入的 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL / LLM_USE_JSON_SCHEMA / LLM_ENABLE_THINKING，
不写死模型名、地址或密钥；未注入时直接返回 None，报告使用模板文本。
"""
from __future__ import annotations

import json
import logging
import os
import re
import time

log = logging.getLogger("b04")

SYSTEM = ("你是电磁态势分析员。只能依据用户提供的 JSON 事实作答，不得编造编号、数值或时段；"
          "涉及对象和航迹时使用 object_id、track_id；区分事实与推断，推断要说明依据；"
          "运动上的接近或同向不能单独作为协作依据。输出 JSON：{\"summary\": 不超过200字, \"assessment\": 不超过600字}。")


def llm_available() -> bool:
    return all(os.environ.get(k) for k in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"))


def _chat(messages: list[dict], timeout: int = 120, retries: int = 3) -> str | None:
    import requests

    body = {"model": os.environ["LLM_MODEL"], "messages": messages, "temperature": 0.2}
    if os.environ.get("LLM_USE_JSON_SCHEMA") == "1":
        body["response_format"] = {"type": "json_object"}
    if os.environ.get("LLM_ENABLE_THINKING") == "0":
        body["chat_template_kwargs"] = {"enable_thinking": False}
    for k in range(retries):
        try:
            r = requests.post(f"{os.environ['LLM_BASE_URL'].rstrip('/')}/chat/completions",
                              headers={"Authorization": f"Bearer {os.environ['LLM_API_KEY']}"}, json=body, timeout=timeout)
            if r.status_code == 429:  # 限流 120 次/分钟：退避重试
                time.sleep(2 ** k * 2)
                continue
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:  # noqa: BLE001
            log.warning("LLM 调用失败(%d/%d): %s", k + 1, retries, e)
            time.sleep(2 ** k)
    return None


def _compact(facts: dict) -> dict:
    """压缩为模型可读的关键事实（控制上下文长度）。"""
    return {
        "scenario": {k: facts["scenario"][k] for k in ("scenario_id", "data_version", "duration_s", "pulses", "nodes", "tracks")},
        "nodes": [{k: n[k] for k in ("platform_id", "domain", "t_first_s", "t_last_s", "speed_med", "pattern")} for n in facts["nodes"]],
        "objects": [{k: o[k] for k in ("object_id", "type", "category", "domain", "pulses", "track_id", "link_status", "state_frac")}
                    for o in facts["objects"]],
        "tracks": {k: {kk: v[kk] for kk in ("domain", "rcs", "speed_med", "pattern", "extent_km") if kk in v} | {"in_area_frac": v.get("in_area_frac")}
                   for k, v in facts["tracks"].items()},
        "tracked_nodes": [{k: r[k] for k in ("object_id", "node", "dwells")} for r in facts["tracked_nodes"]],
        "track_relations": [{k: r[k] for k in ("track_a", "track_b", "relation", "dist_med_m", "rel_speed_med")} for r in facts["track_relations"]],
        "unlinked_tracks": facts["unlinked_tracks"],
    }


def _validate(text: str, facts: dict) -> bool:
    """事实锁定：文中出现的 object/track/节点编号必须存在于事实中。"""
    objs = {o["object_id"] for o in facts["objects"]}
    trks = {int(k) for k in facts["tracks"]}
    nodes = {n["platform_id"] for n in facts["nodes"]}
    for m in re.finditer(r"(?:object_id|对象)\s*[=＝]?\s*(\d+)", text):
        if int(m.group(1)) not in objs:
            return False
    for m in re.finditer(r"(?:track_id|航迹)\s*[=＝]?\s*(\d+)", text):
        if int(m.group(1)) not in trks:
            return False
    for m in re.finditer(r"\b(10[34]\d{3})\b", text):
        if int(m.group(1)) not in nodes:
            return False
    return True


def narrative(facts: dict) -> dict | None:
    if not llm_available():
        return None
    msg = [{"role": "system", "content": SYSTEM},
           {"role": "user", "content": "事实如下：\n" + json.dumps(_compact(facts), ensure_ascii=False, default=str)}]
    for _ in range(2):
        out = _chat(msg)
        if not out:
            return None
        out = re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip()
        m = re.search(r"\{.*\}", out, flags=re.S)
        try:
            d = json.loads(m.group(0)) if m else None
        except json.JSONDecodeError:
            d = None
        if d and _validate(json.dumps(d, ensure_ascii=False), facts):
            return {"summary": str(d.get("summary", ""))[:400], "assessment": str(d.get("assessment", ""))[:1200]}
        msg.append({"role": "user", "content": "上次输出含不存在的编号或格式错误，请严格依据事实重新输出 JSON。"})
    return None
