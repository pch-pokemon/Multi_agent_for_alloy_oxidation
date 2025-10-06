# -*- coding: utf-8 -*-
"""
机器学习智能体 (高温合金氧化增重预测)
============================
支持能力：
- 单条 (one)：解析一条文本或字典输入，返回预测的 mg/cm²
- 批量文本 (batch_text)：分号(;) 或换行分隔的多条文本，逐条解析并预测
- CSV 批量预测 (csv_predict)：读取 CSV，写出 predictions.csv 并返回绝对路径（含原始行 + 预测列）
- CSV 评估 (csv_evaluate)：CSV 需包含真实标签列（默认尝试 mass_gain/weight_gain/target/y/mg_cm2/gt/ground_truth），返回 {"MAE":..., "RMSE":..., "R2":...}

数据源：
- 默认从同目录下优先加载训练好的机器学习模型，且特征值与当前代码需一一对应。
"""
from __future__ import annotations
from typing import List, Dict
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
import re, os, json
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
#--------------------------------------------------
# config 配置
#--------------------------------------------------
import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatZhipuAI
# 导入 API Key
load_dotenv(r"C:\Users\12279\ZHIPU.env")
api_key = os.getenv('API_KEY')
llm_ML = ChatZhipuAI(model="glm-4-air-250414",api_key=api_key,temperature=0.6)
#--------------------------------------------------
# 以当前文件位置为基准
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent     # .../ml_predict
DATA_DIR = BASE_DIR / "data"                   # .../ml_predict/data

def p_data(*parts) -> Path:
    """拼出包内数据文件的绝对路径"""
    return DATA_DIR.joinpath(*parts)

# ------------------ 1) 合金式解析 ------------------
@tool
def parse_alloy_composition(formula: str) -> str:
    """
    Parse alloy formulas like 'Co-10Al-5W-2Ti' and return a 'k=v' string 
    (with the remaining balance assigned to the first element as the matrix).  
    Example return: 'Al=10.0, Co=83.0, Ti=2.0, W=5.0'
    """
    s = formula.replace(" ", "").replace("·", "-").replace("_", "-")
    m = re.match(r"([A-Z][a-z]?)(.*)", s)
    if not m:
        raise ValueError("No base element found in formula.")
    base = m.group(1)
    tail = m.group(2).replace("-", " ")
    # 无前导数字的元素名自动加 1（例如 ' Co Al10 W5 ' -> '1Co 10Al 5W'）
    tail = re.sub(r"\b([A-Z][a-z]?)\b", r"1\1", tail)
    matches = re.findall(r"(\d+(?:\.\d+)?)([A-Z][a-z]?)", tail)
    if not matches:
        raise ValueError("No alloy elements parsed.")
    elems = {}
    for pct, el in matches:
        elems[el] = elems.get(el, 0.0) + float(pct)
    total = sum(elems.values())
    balance = 100.0 - total
    if balance < -1e-6:
        raise ValueError(f"Total composition exceeds 100% (sum={total:.3f}).")
    elems[base] = elems.get(base, 0.0) + max(balance, 0.0)
    ordered = sorted(elems.items(), key=lambda x: x[0])
    return ", ".join(f"{k}={round(v, 4)}" for k, v in ordered)


# ------------------ 2) 预测核心类 ------------------
class MLInfer:
    """
    Unified predictor: supports single sample, text batch, and CSV prediction & CSV evaluation.
    Features (aligned with the model): processing parameters + elements + conditions
    """
    """
    统一预测器：支持单条/文本批量/CSV 预测 & CSV 评估
    特征（对齐模型）：工艺 + 元素 + 条件
    """
    # 模型训练时的列顺序
    feature_order: List[str] = [
        'solu temp','solu time','aging temp1','aging time1','aging temp2','aging time2',
        'Co','Al','W','Ni','Cr','Mo','Fe','Nb','C','Hf','Si','Ta','Ti','Y','V','B','Zr','Ir','Mn','Sc','La','Re',
        'Temperature','Time'
    ]

    # 常见别名映射（输入标准化到 feature_order）
    alias_map: Dict[str, str] = {
        # 条件
        'temperature': 'Temperature', 'temp': 'Temperature', 'temperature_c': 'Temperature', 'test_temperature': 'Temperature',
        'time': 'Time', 'test_time': 'Time', 'duration': 'Time',
        # 工艺（下划线/空格归一）
        'solu_temp': 'solu temp', 'solution_temp': 'solu temp', 'solution temperature': 'solu temp',
        'solu_time': 'solu time', 'solution_time': 'solu time', 'solution time': 'solu time',
        'aging_temp1': 'aging temp1', 'aging temperature1': 'aging temp1',
        'aging_time1': 'aging time1', 'aging time1': 'aging time1',
        'aging_temp2': 'aging temp2', 'aging temperature2': 'aging temp2',
        'aging_time2': 'aging time2', 'aging time2': 'aging time2',
    }

    def __init__(self, model_path: str = "rf_oxidation_model.pkl"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.model = joblib.load(model_path)

    # --- helpers ---
    @staticmethod
    def _norm_key(k: str) -> str:
        """
        Normalize key names: strip whitespace → lowercase → convert to snake_case → apply alias mapping → restore to model feature name
        """
        """
        标准化键名：去两端空白 -> 小写 -> 下划线化 -> 别名映射 -> 恢复模型名
        """
        raw = k.strip()
        low = raw.lower().replace("-", "_").replace(" ", "_")
        return MLInfer.alias_map.get(low, raw if raw in MLInfer.feature_order else raw.title() if raw.title() in MLInfer.feature_order else raw)

    @classmethod
    def _kv_string_to_dict(cls, s: str) -> Dict[str, float]:
        """
        Parse 'Al=8.1, Co=83.6, Temperature=900, Time=10' into a dict 
        (missing features default to 0).
        - Case-insensitive, whitespace/underscore interchangeable, common aliases supported
        - Only keep columns in feature_order; unknown keys are ignored
        """
        """
        将 'Al=8.1, Co=83.6, Temperature=900, Time=10' 解析为 dict（未出现的特征默认 0）。
        - 不区分大小写、空格/下划线互转、支持常见别名
        - 仅保留 feature_order 中的列，未知键忽略
        """
        d = {k: 0.0 for k in cls.feature_order}
        # 允许科学计数法/负号
        for k, v in re.findall(r"([A-Za-z _]+)=\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", s):
            key_std = cls._norm_key(k)
            if key_std in d:
                d[key_std] = float(v)
        return d

    def _predict_df(self, df: pd.DataFrame) -> np.ndarray:
        # 缺列补 0，多列裁剪顺序
        for col in self.feature_order:
            if col not in df.columns:
                df[col] = 0.0
        X = df[self.feature_order]
        return self.model.predict(X)

    # --- public APIs ---
    def predict_one(self, s: str) -> float:
        df = pd.DataFrame([self._kv_string_to_dict(s)])
        return float(self._predict_df(df)[0])

    def predict_text_batch(self, text: str) -> List[float]:
        """
        Text batch: multiple 'k=v' entries separated by semicolons or newlines.
        """
        """
        文本批量：分号或换行分隔多条 'k=v'。
        """
        lines = [ln.strip() for ln in re.split(r"[;\n]+", text) if ln.strip()]
        rows = [self._kv_string_to_dict(ln) for ln in lines]
        df = pd.DataFrame(rows)
        ys = self._predict_df(df)
        return [float(v) for v in ys]

    def predict_csv(self, csv_path: str, save_path: str = "predictions.csv") -> str:
        df = pd.read_csv(csv_path)
        preds = self._predict_df(df)
        out = df.copy()
        out["pred"] = preds
        out.to_csv(save_path, index=False)
        return os.path.abspath(save_path)

    def evaluate_csv(self, csv_path_with_gt: str, y_col_candidates: List[str] = None) -> Dict[str, float]:
        if y_col_candidates is None:
            y_col_candidates = [
                "Mass gain(mg/cm^2)",'mass gain(mg/cm^2)','mass_gain','Mass_gain',
                'mass_gain','weight_gain','target','y','mg_cm2','gt','ground_truth',""]
        df = pd.read_csv(csv_path_with_gt)
        # 找 y 列（不区分大小写）
        found = None
        cols_lower = {c.lower(): c for c in df.columns}
        for k in y_col_candidates:
            if k.lower() in cols_lower:
                found = cols_lower[k.lower()]
                break
        if not found:
            raise ValueError(f"Cannot find ground-truth column; tried {y_col_candidates}")
        y_true = df[found].astype(float).to_numpy()
        y_pred = self._predict_df(df).astype(float)
        mae  = float(np.mean(np.abs(y_pred - y_true)))
        rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        r2 = float(1.0 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)) if np.var(y_true) > 0 else 0.0
        return {"MAE": mae, "RMSE": rmse, "R2": r2}


# 全局实例（供工具使用）放入训练好的模型
try:
    _mlinfer = MLInfer(model_path=p_data("rf_oxidation_model.pkl"))
    print("✅ ML模型加载成功/ML model loading successful!")
except Exception as e:
    print(f"❌ ML模型加载失败：{e}/ML model failed to load: {e}")
    raise

# ------------------ 3) 工具映射输出定义 ------------------

import re
from langchain.tools import tool

# -------- 规范与映射 --------
# 以“下划线+小写”的形式作为内部规范键
_COND_KEYS = {
    "temperature", "time",
    "solu_temp", "solu_time",
    "aging_temp1", "aging_time1",
    "aging_temp2", "aging_time2",
}

# 常见别名（包含空格/大小写/连字符等写法）
_ALIAS2CANON = {
    # 温度/时间
    "temperature": "temperature", "temp": "temperature", "t": "temperature",
    "time": "time", "duration": "time", "hours": "time",

    # 固溶（solution / solu）
    "solution_temp": "solu_temp", "solution_temperature": "solu_temp",
    "solutiontemp": "solu_temp", "solu_temp": "solu_temp",
    "sol_temp": "solu_temp", "soltemp": "solu_temp", "solutemp": "solu_temp",

    "solution_time": "solu_time", "solutiontime": "solu_time",
    "solu_time": "solu_time", "sol_time": "solu_time", "soltime": "solu_time",

    # Aging1
    "aging_temp1": "aging_temp1", "aging1_temp": "aging_temp1",
    "aging_temperature1": "aging_temp1", "aging1temperature": "aging_temp1",

    "aging_time1": "aging_time1", "aging1_time": "aging_time1",
    "aging_duration1": "aging_time1", "aging1duration": "aging_time1",

    # Aging2
    "aging_temp2": "aging_temp2", "aging2_temp": "aging_temp2",
    "aging_temperature2": "aging_temp2", "aging2temperature": "aging_temp2",

    "aging_time2": "aging_time2", "aging2_time": "aging_time2",
    "aging_duration2": "aging_time2", "aging2duration": "aging_time2",
}

# 元素大小写映射（按化学符号）
_ELEM_SYMBOL = {
    "co":"Co","ni":"Ni","fe":"Fe","al":"Al","w":"W","cr":"Cr","ta":"Ta","ti":"Ti",
    "mo":"Mo","mn":"Mn","si":"Si","nb":"Nb","re":"Re","hf":"Hf","ir":"Ir","la":"La",
    "b":"B","c":"C","y":"Y","sc":"Sc","v":"V","zr":"Zr"
}

_BASE_ORDER = {"co": 0, "ni": 1, "fe": 2}  # 基体优先

def _clean_key(k: str) -> str:
    """转小写；空格/连字符→下划线；去掉多余符号"""
    k = str(k).strip()
    k = k.replace("(", "").replace(")", "")
    k = re.sub(r"[\s\-]+", "_", k.strip())
    return k.lower()

def _normalize_keys(d: dict) -> dict:
    """规范键名并做别名映射（条件键归一化；元素键保持原值但内部用小写索引）"""
    out = {}
    for k, v in d.items():
        ck = _clean_key(k)
        canon = _ALIAS2CANON.get(ck, ck)     # 条件别名归一
        out[canon] = v
    return out

def _as_float(x):
    try:
        return float(x)
    except Exception:
        return None

def _is_zero(x) -> bool:
    fx = _as_float(x)
    return fx is not None and abs(fx) < 1e-12

def _format_num(x) -> str:
    fx = _as_float(x)
    if fx is None:
        return str(x)
    return f"{int(fx)}" if abs(fx - int(fx)) < 1e-9 else f"{fx:.1f}"

def _format_alloy(d: dict) -> str:
    """
    只拼接“元素=数值”，按 Co/Ni/Fe 优先，其余按字母序；
    过滤值为 0 的元素；元素名按化学符号大小写输出。
    """
    elems = []
    for k, v in d.items():
        if k in _COND_KEYS:
            continue
        lk = _clean_key(k)  # 允许用户把元素写成 Al / AL / al
        if lk in _ELEM_SYMBOL and not _is_zero(v):
            elems.append((_ELEM_SYMBOL[lk], v))

    # 排序
    def _sort_key(item):
        sym = item[0]         # e.g., 'Co'
        lk = sym.lower()      # 'co'
        return (_BASE_ORDER.get(lk, 10), sym)

    elems.sort(key=_sort_key)
    return ", ".join([f"{sym}={_format_num(val)}" for sym, val in elems])

def _format_conditions(d: dict) -> str:
    """
    (at 900°C 10h; solution: 1150°C 1h; aging1: 900°C 20h; aging2: 800°C 50h)
    仅在对应值非零时显示该段。
    """
    parts = []

    # 主条件
    T = d.get("temperature"); t = d.get("time")
    head = []
    if T is not None and not _is_zero(T): head.append(f"{_format_num(T)}°C")
    if t is not None and not _is_zero(t): head.append(f"{_format_num(t)}h")
    if head:
        parts.append("at " + " ".join(head))

    # 各工艺段
    def step(label, kt, ktm):
        TT = d.get(kt); tt = d.get(ktm)
        seg = []
        if TT is not None and not _is_zero(TT): seg.append(f"{_format_num(TT)}°C")
        if tt is not None and not _is_zero(tt): seg.append(f"{_format_num(tt)}h")
        if seg:
            parts.append(f"{label}: " + " ".join(seg))

    step("solution", "solu_temp", "solu_time")
    step("aging1",   "aging_temp1", "aging_time1")
    step("aging2",   "aging_temp2", "aging_time2")

    return (" (" + "; ".join(parts) + ")") if parts else ""

# ================= 工具函数 =================
# (return_direct=True)
@tool
def predict_mass_gain_one(input: str) -> str:
    """
    Single Prediction.
    Input Example: 'Al=8.1, Co=83.6, W=8.3, Temperature=900, Time=10'
    Notes:
      - Supports optional process parameters: solu_temp / solu_time / aging_temp1 / aging_time1 / aging_temp2 / aging_time2 (aliases supported).
      - Any optional parameters not provided or equal to 0 will be omitted in display.
    """
    y = _mlinfer.predict_one(input)
    parsed = _mlinfer._kv_string_to_dict(input)
    norm   = _normalize_keys(parsed)
    alloy  = _format_alloy(norm)
    conds  = _format_conditions(norm)
    return f"{alloy}{conds} → Predicted weight gain: {float(y):.4f} mg/cm²"

# (return_direct=True)
@tool
def predict_mass_gain_batch(text: str) -> str:
    """
    Batch Text Prediction with readable outputs (numbered lines).
    Use semicolons (;) or line breaks to separate multiple 'k=v' entries.
    """
    lines = [ln.strip() for ln in re.split(r"[;\n]+", text) if ln.strip()]
    ys = _mlinfer.predict_text_batch(text)

    out_lines = []
    for i, (ln, y) in enumerate(zip(lines, ys), 1):
        parsed = _mlinfer._kv_string_to_dict(ln)
        norm   = _normalize_keys(parsed)
        alloy  = _format_alloy(norm)
        conds  = _format_conditions(norm)
        out_lines.append(f"{i}. {alloy}{conds} → Predicted weight gain: {float(y):.4f} mg/cm²")

    return "\n".join(out_lines)

@tool
def predict_mass_gain_csv(csv_path: str) -> str:
    """
    CSV Batch Prediction: Reads the given csv_path, writes out predictions.csv, and returns the absolute path.
    Notes:
      - The CSV must contain at least Temperature and Time columns.
      - Missing element or process parameter columns will be treated as 0 by default.动补 0。
    """
    out_path = _mlinfer.predict_csv(csv_path, save_path="predictions.csv")
    return f"Saved predictions to: {out_path}"


@tool
def evaluate_mass_gain_csv(csv_path_with_gt: str) -> str:
    """
    CSV Evaluation: The CSV must contain a ground-truth label column (by default, it will look for one of: mass_gain / weight_gain / target / y / mg_cm2 / gt / ground_truth).  
    Returns a JSON string: {"MAE": ..., "RMSE": ..., "R2": ...}:...}
    """
    metrics = _mlinfer.evaluate_csv(csv_path_with_gt)
    return json.dumps(metrics, ensure_ascii=False)


# ------------------ 4) 提示词设定 ------------------

ML_SYSTEM_PROMPT = """
You are the ML_agent for oxidation mass-gain predictions (mg/cm²).
Your tools:
- parse_alloy_composition: parse 'Co-10Al-5W-2Ti' → 'k=v' with balance.
- predict_mass_gain_one: single 'k=v' prediction.
- predict_mass_gain_batch: text-batch; lines/semicolons separate multiple 'k=v'.
- predict_mass_gain_csv: CSV batch prediction.
- evaluate_mass_gain_csv: CSV evaluation with ground truth.

Rules:
1) If user gives alloy formulas, call parse_alloy_composition first, then add Temperature & Time.
2) If multiple alloys, build a text with one 'k=v' per line and call predict_mass_gain_batch.
3) Optional heat-treatment keys (e.g., solu_temp/aging_temp1...) are accepted if provided; otherwise treated as 0.
4) When you have completed all the tasks, Report your results succinctly with units mg/cm². 
5) If question is out of scope, reply "I don't know, other agents may help this question".
""".strip()


# 定义智能体和工具

def build_ml_agent(create_react_agent, llm_ML):
    tools = [
        parse_alloy_composition,
        predict_mass_gain_one,
        predict_mass_gain_batch,
        predict_mass_gain_csv,
        evaluate_mass_gain_csv,]
    agent = create_react_agent(
        model=llm_ML,
        tools=tools,
        prompt=ML_SYSTEM_PROMPT,
        name="ML_agent",
    )
    return agent

ML_agent = build_ml_agent(create_react_agent, llm_ML)
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

def ml_alloy_chat(query: str, debug: bool = True) -> Optional[str]:
    """
    向 ML_agent 发送一次性问题并按需打印：
      - debug=True : 完整打印全部 streaming 过程
      - debug=False: 仅返回最后一个 AI 答案
    """
    final_ai_message: Optional[BaseMessage] = None

    if debug:
        # 完整过程打印
        for chunk in ML_agent.stream(
            {"messages": [{"role": "user", "content": query}]}
        ):
            pretty_print_messages(chunk, last_message=False)
            last_ai = _extract_last_ai_message(chunk)
            if last_ai is not None:
                final_ai_message = last_ai
    else:
        # 静默收集，直到结束只返回最终答案
        for chunk in ML_agent.stream(
            {"messages": [{"role": "user", "content": query}]}
        ):
            last_ai = _extract_last_ai_message(chunk)
            if last_ai is not None:
                final_ai_message = last_ai

    return getattr(final_ai_message, "content", None)
