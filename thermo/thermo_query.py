# -*- coding: utf-8 -*-
"""
热力学智能体
================================
支持能力：
- 精确 (exact)：按氧化物 + 温度(°C) 精确查找
- 模糊 (fuzzy)：氧化物名支持模糊匹配（Al2O3 ~> AL2O3, al203, "alumina" 等近似字符串）
- 插值 (interpolate)：若该温度没有精确数据，按温度对各热力学属性做插值计算（每种氧化物独立插值）
- 曲线 (curve)：返回温度范围内的属性曲线（可用于绘图、进一步计算）
- 批量 (batch)：一次性查询多个氧化物与多个温度点
数据源：
- 默认从同目录下优先加载 `thermo_data.xlsx`，若不存在则回退到 `thermo_data.csv`
- 需要包含以下字段（大小写严格）：
- oxide, temperature_C, deltaH_kJ, deltaS_J/K, deltaG_kJ, K, Log(K), pO2
"""

from __future__ import annotations
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Iterable, Any

import pandas as pd
from langchain.tools import tool
from difflib import get_close_matches
#--------------------------------------------------
# config 配置
#--------------------------------------------------
import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatZhipuAI
# 导入 API Key
load_dotenv(r"C:\Users\12279\ZHIPU.env")
api_key = os.getenv('API_KEY')
llm_Thermo = ChatZhipuAI(model="glm-4-air-250414",api_key=api_key,temperature=0.6)
#--------------------------------------------------
# 以当前文件位置为基准
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent     # .../thermo
DATA_DIR = BASE_DIR / "data"                   # .../thermo/data

def p_data(*parts) -> Path:
    """拼出包内数据文件的绝对路径"""
    return DATA_DIR.joinpath(*parts)
#--------------------------------------------------
#config 配置
thermo_raw_data_xlsx = p_data("thermo_data.xlsx") # 热力学数据库地址
thermo_raw_data_csv = p_data("thermo_data.csv")
#--------------------------------------------------

# =========
# I/O & Data
# =========

REQUIRED_COLUMNS = [
    "oxide", "temperature_C", "deltaH_kJ", "deltaS_J/K", "deltaG_kJ", "K", "Log(K)", "pO2"
]

def _load_thermo_df(path_xlsx: str = thermo_raw_data_xlsx, path_csv: str = thermo_raw_data_csv) -> pd.DataFrame:
    """Load thermo dataframe from xlsx or csv and perform basic cleaning.

    - Prefer xlsx, fallback to csv
    - Enforce required columns
    - Coerce numeric columns
    - Strip oxide strings and keep as-is for exact match; also maintain a normalized key for fuzzy
    """
    """
    从 xlsx 或 csv 文件中加载热力学数据表，并进行基础清理。

    - 优先加载 xlsx 文件，若不存在则回退到 csv
    - 强制检查所需列是否存在
    - 将数值列强制转换为数值类型
    - 去除氧化物字符串的首尾空格，保持原始值用于精确匹配；
      同时维护一个规范化的键用于模糊匹配
    """
    df = None
    try:
        df = pd.read_excel(path_xlsx)
    except Exception:
        try:
            df = pd.read_csv(path_csv)
        except Exception as e:
            raise FileNotFoundError(
                f"Cannot load thermo data; tried '{path_xlsx}' and '{path_csv}'.\n{e}"
            )

    # enforce columns
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Thermo data is missing required columns: {missing}")

    # strip oxide and build normalized key
    df["oxide"] = df["oxide"].astype(str).str.strip()
    df["_oxide_norm"] = df["oxide"].str.upper().str.replace(" ", "", regex=False)

    # coerce numerics
    num_cols = ["temperature_C", "deltaH_kJ", "deltaS_J/K", "deltaG_kJ", "K", "Log(K)", "pO2"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # drop rows without temperature or oxide
    df = df.dropna(subset=["_oxide_norm", "temperature_C"]).copy()

    # sort for interpolation
    df = df.sort_values(["_oxide_norm", "temperature_C"]).reset_index(drop=True)
    return df


# lazy singleton dataframe
_THERMO_DF: Optional[pd.DataFrame] = None


def get_df() -> pd.DataFrame:
    global _THERMO_DF
    if _THERMO_DF is None:
        _THERMO_DF = _load_thermo_df()
    return _THERMO_DF

# ============ 
# 提示信息 
# ============ 
try:
    _THERMO_DF = _load_thermo_df()
    print("✅ 热力学数据库加载成功/thermodynamic database loading successful!")
except Exception as e:
    print(f"❌ 热力学数据库加载失败: {e}/thermodynamic database failed to load: {e}")


# =========
# 核心逻辑
# =========

def _norm_oxide(s: str) -> str:
    return str(s).strip().upper().replace(" ", "")


def _fuzzy_pick(query: str, candidates: Iterable[str], cutoff: float = 0.75) -> Optional[str]:
    """Use difflib to pick the closest candidate key; return None if below cutoff"""
    """使用 difflib 选择最接近的候选键；若相似度低于阈值则返回 None"""
    q = _norm_oxide(query)
    # difflib works better when we pass human-like strings; we already normalized
    matches = get_close_matches(q, list(set(candidates)), n=1, cutoff=cutoff)
    return matches[0] if matches else None


NUMERIC_FIELDS = ["deltaH_kJ", "deltaS_J/K", "deltaG_kJ", "K", "Log(K)", "pO2"]


def _format_value(field: str, val: Any) -> Any:
    if pd.isna(val):
        return None
    if field in ("K", "pO2"):
        try:
            return f"{float(val):.2e}"
        except Exception:
            return None
    # other numeric keep as float
    try:
        return float(val)
    except Exception:
        return None


@dataclass
class QueryResult:
    oxide: str
    temperature: float
    matched_oxide: Optional[str] = None
    mode: str = "exact"  # exact | interpolated | fuzzy-exact | fuzzy-interpolated
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        base = {
            "oxide": self.oxide,
            "temperature": self.temperature,
            "mode": self.mode,
        }
        if self.matched_oxide and self.matched_oxide != self.oxide:
            base["matched_oxide"] = self.matched_oxide
        if self.error:
            base["error"] = self.error
        if self.data:
            base.update(self.data)
        return base


# ---------- 1) 对于给定的氧化物进行精确查询或内插计算 ----------

def _interp_row_for_temp(df_o: pd.DataFrame, T: float) -> Tuple[str, Dict[str, Any]]:
    """Given a sub-dataframe for one oxide (sorted by temperature),
    return mode and data dict for temperature T.

    - If exact T exists, mode = "exact"
    - Else do linear interpolation between nearest neighbors, mode = "interpolated"
    - If T out of bounds (< min or > max), perform edge linear extrapolation using two closest points
    """
    """给定某一氧化物的子 DataFrame（按温度排序），
    返回指定温度 T 的模式（mode）和数据字典。

    - 如果存在精确的 T，mode = "exact"
    - 否则在最近的两个点之间进行线性插值，mode = "interpolated"
    - 如果 T 超出范围（< 最小值或 > 最大值），则使用最近的两个点做边界线性外推
    """
    # exact hit?
    hit = df_o[df_o["temperature_C"] == T]
    if not hit.empty:
        row = hit.iloc[0]
        data = {f: _format_value(f, row[f]) for f in NUMERIC_FIELDS}
        return "exact", data

    # no exact -> locate neighbors
    temps = df_o["temperature_C"].to_numpy()
    # find position to insert
    import bisect
    pos = bisect.bisect_left(list(temps), T)

    def lin(aT, aV, bT, bV, t):
        if any(pd.isna(x) for x in (aV, bV)):
            return None
        if bT == aT:
            return float(aV)
        return float(aV) + (float(bV) - float(aV)) * ( (t - aT) / (bT - aT) )

    def edge_pair(idx1, idx2):
        r1, r2 = df_o.iloc[idx1], df_o.iloc[idx2]
        return r1, r2

    # choose neighbors for interpolation/extrapolation
    if 0 < pos < len(df_o):
        r1, r2 = edge_pair(pos-1, pos)
    elif pos == 0 and len(df_o) >= 2:
        r1, r2 = edge_pair(0, 1)  # low-side extrapolation
    elif pos >= len(df_o) and len(df_o) >= 2:
        r1, r2 = edge_pair(len(df_o)-2, len(df_o)-1)  # high-side extrapolation
    else:
        # not enough points to interpolate
        row = df_o.iloc[0]
        data = {f: _format_value(f, row[f]) for f in NUMERIC_FIELDS}
        return "exact" if row["temperature_C"] == T else "interpolated", data

    aT, bT = float(r1["temperature_C"]), float(r2["temperature_C"])
    data = {}
    for f in NUMERIC_FIELDS:
        aV, bV = r1[f], r2[f]
        v = lin(aT, aV, bT, bV, float(T))
        # format for K / pO2 at the end
        data[f] = v
    # final formatting
    data = {k: _format_value(k, v) for k, v in data.items()}
    return "interpolated", data


# ---------- public helpers ----------

def query_one(oxide: str, temperature_C: float, fields: Optional[List[str]] = None,
              fuzzy: bool = True, fuzzy_cutoff: float = 0.75) -> QueryResult:
    df = get_df()
    ox_norm = _norm_oxide(oxide)

    # pick subset by exact or fuzzy oxide
    df_exact = df[df["_oxide_norm"] == ox_norm]
    picked_key = None
    picked_df = df_exact
    mode_prefix = ""

    if picked_df.empty and fuzzy:
        picked_key = _fuzzy_pick(oxide, df["_oxide_norm"].unique(), cutoff=fuzzy_cutoff)
        if picked_key:
            picked_df = df[df["_oxide_norm"] == picked_key]
            mode_prefix = "fuzzy-"
        else:
            return QueryResult(oxide=oxide, temperature=float(temperature_C), error=f"No oxide matched (exact or fuzzy)")

    if picked_df.empty:
        return QueryResult(oxide=oxide, temperature=float(temperature_C), error=f"No oxide data available")

    picked_df = picked_df.sort_values("temperature_C")
    mode, data = _interp_row_for_temp(picked_df, float(temperature_C))

    # keep only required fields
    if fields:
        keep = [f for f in fields if f in NUMERIC_FIELDS]
        data = {k: v for k, v in data.items() if k in keep}

    # record pretty matched oxide from original strings
    matched_oxide = picked_df.iloc[0]["oxide"] if picked_key else None

    return QueryResult(
        oxide=oxide,
        temperature=float(temperature_C),
        matched_oxide=matched_oxide,
        mode=(mode_prefix + mode) if mode_prefix else mode,
        data=data,
    )


def query_curve(oxide: str, t_min: float, t_max: float, step: float = 50.0,
                fields: Optional[List[str]] = None, fuzzy: bool = True,
                fuzzy_cutoff: float = 0.75) -> Dict[str, Any]:
    """Return a curve (list of points) for given temperature range.
    Each point contains the requested fields.
    """
    """返回给定温度范围内的曲线（点的列表）
    每个点包含所请求的字段
    """
    if step <= 0:
        raise ValueError("step must be positive")
    pts = []
    T = t_min
    while T <= t_max + 1e-9:  # include end
        r = query_one(oxide, T, fields=fields, fuzzy=fuzzy, fuzzy_cutoff=fuzzy_cutoff)
        if r.error:
            pts.append({"temperature": float(T), "error": r.error})
        else:
            d = {k: v for k, v in r.data.items()} if r.data else {}
            d.update({"temperature": float(T)})
            pts.append(d)
        T += step
    result = {
        "oxide": oxide,
        "t_min": float(t_min),
        "t_max": float(t_max),
        "step": float(step),
        "points": pts,
    }
    return result


# ============
# LangChain 工具定义
# ============

_ALLOWED_FIELDS = {
    "pO2": "pO2",
    "deltaH_kJ": "deltaH_kJ",
    "deltaS_J/K": "deltaS_J/K",
    "deltaG_kJ": "deltaG_kJ",
    "Log(K)": "Log(K)",
    "K": "K",
}


def _parse_kv_input(input: str) -> Dict[str, str]:
    parts = {}
    for seg in input.split(";"):
        if "=" in seg:
            k, v = seg.split("=", 1)
            parts[k.strip().lower()] = v.strip()
    return parts


@tool #return_direct=True)
def query_thermodynamic_data(input: str) -> str:
    """
    📌 Purpose: Retrieve thermodynamic data for one or more oxides at a specified temperature.
    Optionally, specify which thermodynamic properties to return using the fields parameter.

    ⚙️ Required Input:
    - oxide: One or more oxide chemical formulas, separated by commas.
    - temperature: The query temperature in °C (must be a numeric value).

    ⚙️ Optional Input:
    - fields: A comma-separated list of properties to return (case-sensitive).
      If not specified, all available properties will be returned by default.
    - fuzzy: yes/no (default: yes)

    ✅ Allowed field names (case-sensitive): pO2, deltaH_kJ, deltaS_J/K, deltaG_kJ, Log(K), K

    🧩 Example inputs:
    - "oxide=Al2O3; temperature=900"
    - "oxide=Al2O3,Cr2O3; temperature=1000; fields=pO2"
    - "oxide=Al2O3; temperature=900; fields=pO2,deltaH_kJ; fuzzy=no"
    """
    try:
        kv = _parse_kv_input(input)
    except Exception:
        return json.dumps({"error": "Invalid input format. Use: oxide=Al2O3,Cr2O3; temperature=900; fields=pO2,deltaH_kJ"}, ensure_ascii=False)

    oxide_str = kv.get("oxide", "")
    if not oxide_str:
        return json.dumps({"error": "No oxide specified."}, ensure_ascii=False)
    oxides = [o.strip() for o in oxide_str.split(",") if o.strip()]

    temp_str = kv.get("temperature")
    if not temp_str:
        return json.dumps({"error": "No temperature specified."}, ensure_ascii=False)
    try:
        T = float(temp_str)
    except ValueError:
        return json.dumps({"error": "Invalid temperature value."}, ensure_ascii=False)

    fields = [f.strip() for f in kv.get("fields", "").split(",") if f.strip()]
    if fields:
        for f in fields:
            if f not in _ALLOWED_FIELDS:
                return json.dumps({"error": f"Invalid field: {f}"}, ensure_ascii=False)
    else:
        fields = list(_ALLOWED_FIELDS.values())

    fuzzy = kv.get("fuzzy", "yes").lower() not in ("no", "false", "0")

    results: List[Dict[str, Any]] = []
    for ox in oxides:
        r = query_one(ox, T, fields=fields, fuzzy=fuzzy)
        results.append(r.to_dict())

    return json.dumps(results, indent=2, ensure_ascii=False)


@tool #(return_direct=True)
def query_thermo_curve(input: str) -> str:
    """
    📌 Purpose: Return property curves for an oxide across a temperature range.

    ⚙️ Required Input:
    - oxide: One oxide chemical formula
    - t_min: Minimum temperature in °C
    - t_max: Maximum temperature in °C

    ⚙️ Optional Input:
    - step: Temperature step in °C (default 50)
    - fields: Comma-separated property names (default: all)
    - fuzzy: yes/no (default: yes)

    🧩 Example:
    - "oxide=Al2O3; t_min=600; t_max=1200; step=100; fields=deltaG_kJ,pO2"
    """
    kv = _parse_kv_input(input)
    oxide = kv.get("oxide", "").strip()
    if not oxide:
        return json.dumps({"error": "No oxide specified."}, ensure_ascii=False)

    def _num(name: str, required=True, default=None):
        s = kv.get(name)
        if s is None:
            if required:
                raise ValueError(f"Missing {name}")
            return default
        try:
            return float(s)
        except Exception:
            raise ValueError(f"Invalid {name}")

    try:
        t_min = _num("t_min")
        t_max = _num("t_max")
        step = _num("step", required=False, default=50.0)
    except ValueError as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

    fields = [f.strip() for f in kv.get("fields", "").split(",") if f.strip()]
    if fields:
        for f in fields:
            if f not in _ALLOWED_FIELDS:
                return json.dumps({"error": f"Invalid field: {f}"}, ensure_ascii=False)
    else:
        fields = list(_ALLOWED_FIELDS.values())

    fuzzy = kv.get("fuzzy", "yes").lower() not in ("no", "false", "0")

    curve = query_curve(oxide, t_min, t_max, step=step, fields=fields, fuzzy=fuzzy)

    # format numeric values for K/pO2 at the point level
    pts = []
    for p in curve["points"]:
        q = {}
        for k, v in p.items():
            if k in ("K", "pO2") and v is not None and not isinstance(v, str):
                try:
                    q[k] = f"{float(v):.2e}"
                except Exception:
                    q[k] = None
            else:
                q[k] = v
        pts.append(q)
    curve["points"] = pts

    return json.dumps(curve, indent=2, ensure_ascii=False)


@tool #(return_direct=True)
def query_thermo_batch(input: str) -> str:
    """
    📌 Purpose: Batch query multiple oxides and temperatures.

    Two input formats (choose one):
    1) KV string: "oxide=Al2O3,Cr2O3; temperature=800,900; fields=deltaG_kJ,pO2"
    - All (oxide × temperature) combinations will be queried.
    2) JSON list:
    [
        {"oxide": "Al2O3", "temperature": 900, "fields": ["deltaG_kJ"]},
        {"oxide": "Cr2O3", "temperature": 1000}
    ]
    - Each entry can include its own 'fields'; if not provided, all fields are returned by default.
    """
    input = input.strip()

    def _run_one(ox: str, T: float, fields: Optional[List[str]]):
        if fields:
            for f in fields:
                if f not in _ALLOWED_FIELDS:
                    return {"oxide": ox, "temperature": T, "error": f"Invalid field: {f}"}
        else:
            fields2 = list(_ALLOWED_FIELDS.values())
            fields = fields2
        r = query_one(ox, T, fields=fields, fuzzy=True)
        return r.to_dict()

    if input.startswith("["):
        # JSON array
        try:
            arr = json.loads(input)
        except Exception:
            return json.dumps({"error": "Invalid JSON array"}, ensure_ascii=False)
        results: List[Dict[str, Any]] = []
        for item in arr:
            ox = str(item.get("oxide", "")).strip()
            if not ox:
                results.append({"error": "Missing oxide"})
                continue
            try:
                T = float(item.get("temperature"))
            except Exception:
                results.append({"oxide": ox, "error": "Invalid temperature"})
                continue
            fields = item.get("fields")
            results.append(_run_one(ox, T, fields))
        return json.dumps(results, indent=2, ensure_ascii=False)

    # KV style
    kv = _parse_kv_input(input)
    oxide_str = kv.get("oxide", "")
    temp_str = kv.get("temperature", "")
    if not oxide_str or not temp_str:
        return json.dumps({"error": "Need both oxide and temperature"}, ensure_ascii=False)

    oxides = [o.strip() for o in oxide_str.split(",") if o.strip()]
    temps: List[float] = []
    for s in [t.strip() for t in temp_str.split(",") if t.strip()]:
        try:
            temps.append(float(s))
        except Exception:
            return json.dumps({"error": f"Invalid temperature: {s}"}, ensure_ascii=False)

    fields = [f.strip() for f in kv.get("fields", "").split(",") if f.strip()] or None

    results: List[Dict[str, Any]] = []
    for ox in oxides:
        for T in temps:
            results.append(_run_one(ox, T, fields))
    return json.dumps(results, indent=2, ensure_ascii=False)


# ============
# 提示词设定
# ============

THERMO_SYSTEM_PROMPT = """
You are the Thermo_agent, specialized in consulting thermodynamic data (e.g., pO₂, ΔG).
Responsibilities:
1) Use the provided tools (query_thermodynamic_data / query_thermo_curve / query_thermo_batch) to retrieve oxide formulas and thermodynamic data at specific temperatures.
2) When you have completed all the tasks, Report your findings directly and concisely.

Important:
- Only handle questions related to thermodynamic data. If the question is outside your scope, respond with "I don't know, other agents may help this question".
- Prefer exact data; if missing, use interpolation. If the oxide name is ambiguous, attempt fuzzy match and disclose it in the result.
""".strip()


# ----------- 定义智能体和工具 -----------
from langgraph.prebuilt import create_react_agent
def build_thermo_agent(create_react_agent, llm_Thermo):
    tools = [query_thermodynamic_data, query_thermo_curve, query_thermo_batch]
    agent = create_react_agent(
        model=llm_Thermo,
        tools=tools,
        prompt=THERMO_SYSTEM_PROMPT,
        name="Thermo_agent",
    )
    return agent

Thermo_agent = build_thermo_agent(create_react_agent, llm_Thermo)
#----------------------------
# 使用方法
#----------------------------

#--------打印格式定义---------
from langchain_core.messages import convert_to_messages

def pretty_print_message(message, indent=False):
    pretty_message = message.pretty_repr(html=True)
    if not indent:
        print(pretty_message)
        return

    indented = "\n".join("\t" + c for c in pretty_message.split("\n"))
    print(indented)

def pretty_print_messages(update, last_message=False):
    is_subgraph = False
    if isinstance(update, tuple):
        ns, update = update
        # skip parent graph updates in the printouts
        if len(ns) == 0:
            return

        graph_id = ns[-1].split(":")[0]
        print(f"Update from subgraph {graph_id}:")
        print("\n")
        is_subgraph = True

    for node_name, node_update in update.items():
        update_label = f"Update from node {node_name}:"
        if is_subgraph:
            update_label = "\t" + update_label

        print(update_label)
        print("\n")

        messages = convert_to_messages(node_update["messages"])
        if last_message:
            messages = messages[-1:]

        for m in messages:
            pretty_print_message(m, indent=is_subgraph)
        print("\n")

from langchain_core.messages import convert_to_messages

#--------输出格式管理----------
from typing import Optional
from langchain_core.messages import convert_to_messages, AIMessage, BaseMessage

def _extract_last_ai_message(update) -> Optional[BaseMessage]:
    """
    从单个 stream chunk 的 update 中提取最后一个 AIMessage。
    兼容 (ns, update) 和纯 update 两种形态。
    """
    # 兼容 (namespace, update) 形式
    if isinstance(update, tuple):
        _, update = update

    last_ai: Optional[BaseMessage] = None
    for node_name, node_update in update.items():
        # node_update 形如 {"messages": [...]}
        msgs = convert_to_messages(node_update.get("messages", []))
        for m in msgs:
            if isinstance(m, AIMessage):
                last_ai = m
    return last_ai

def thermo_alloy_chat(query: str, debug: bool = True) -> Optional[str]:
    """
    向 ML_agent 发送一次性问题并按需打印：
      - debug=True : 完整打印全部 streaming 过程
      - debug=False: 仅返回最后一个 AI 答案
    """
    final_ai_message: Optional[BaseMessage] = None

    if debug:
        # 完整过程打印
        for chunk in Thermo_agent.stream(
            {"messages": [{"role": "user", "content": query}]}
        ):
            pretty_print_messages(chunk, last_message=False)
            last_ai = _extract_last_ai_message(chunk)
            if last_ai is not None:
                final_ai_message = last_ai
    else:
        # 静默收集，直到结束只返回最终答案
        for chunk in Thermo_agent.stream(
            {"messages": [{"role": "user", "content": query}]}
        ):
            last_ai = _extract_last_ai_message(chunk)
            if last_ai is not None:
                final_ai_message = last_ai

    return getattr(final_ai_message, "content", None)

