
# -*- coding: utf-8 -*-
"""
高温合金专家多智能体 
角色：协调者
============================
支持能力：
- 基于用户问题的复杂度 自适应调用 机器学习智能体，热力学智能体，RAG与Ref子智能体 来解决问题

重要参数变动：
- 修改源官网底层代码，为每个 子智能体 设置各自的 output_mode 方法 控制协调者的信息可见范围
- degug 参数：默认为True，打印协调者如何思考并分配给各个智能体解决问题的过程。设置为 False，仅显示最终答案，对界面端友好
- return_text: 默认为 False, 显示 role 角色，设置 return_text=True 可以得到干净的 FINAL ANSWER 内容

数据源：
- 默认从同目录下的文件夹下加载各个子智能体（代码运行需要与这些文件配合使用）
"""

# config 配置
#----------------------------
import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatZhipuAI
# 导入 API Key
load_dotenv('C:/Users/12279/ZHIPU.env')
api_key = os.getenv('API_KEY')

#-------------------------------
# 为每一个智能体设定他们各自的 llm
#-------------------------------
"""
我们强烈建议在 没有使用 微调专业知识领域的llm时，为协调者设置为最高级的llm。作为大脑角色，其是思考设计问题如何解决的核心人物。
"""
# from langchain_openai import ChatOpenAI
# load_dotenv('C:/Users/12279/OPENAI.env')
# openai_api_key = os.getenv('API_KEY')
# llm_supervisor = ChatOpenAI(model="gpt-5",temperature=0.5,api_key= openai_api_key, base_url="https://api.chatanywhere.tech/v1") # ★ 核心 llm

llm_supervisor = ChatZhipuAI(model="glm-4.5",api_key=api_key, temperature=0.5) # ★ 核心 llm
llm_ML = ChatZhipuAI(model="glm-4-air",api_key=api_key,temperature=0.5)
llm_Thermo = ChatZhipuAI(model="glm-4-air",api_key=api_key,temperature=0.5)
llm_RAG = ChatZhipuAI(model="glm-4.5",api_key=api_key,temperature=0.5)
llm_Ref = ChatZhipuAI(model="glm-4.5",api_key=api_key,temperature=0.5)
#-------------------------------

#---------------------------
# 建立各个子智能体
#---------------------------
import sys
sys.path.append("..") 

from langgraph.prebuilt import create_react_agent
from ml_predict.ml import build_ml_agent

ML_agent = build_ml_agent(create_react_agent, llm_ML)

from thermo.thermo_query import build_thermo_agent
Thermo_agent = build_thermo_agent(create_react_agent, llm_Thermo)

from rag_core.rag_core import build_rag_agent

RAG_agent = build_rag_agent(create_react_agent, llm_RAG)

from ref_rec.ref_REC import build_ref_agent

Ref_agent = build_ref_agent(create_react_agent, llm_Ref)

#----------------------------
# 创建自定义转接工具
#----------------------------
from langgraph.types import Send
from langchain.tools import tool
from typing import Annotated
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import InjectedState, ToolNode
from langgraph.types import Command

from langgraph.store.memory import InMemoryStore

def create_task_description_handoff_tool(
    *, agent_name: str, description: str | None = None
):
    name = f"transfer_to_{agent_name}"
    description = description or f"Ask {agent_name} for help."

    @tool(name, description=description)
    def handoff_tool(
        # this is populated by the supervisor LLM
        task_description: Annotated[
            str,
            "The task description of the next agent.",
        ],
        # these parameters are ignored by the LLM
        state: Annotated[MessagesState, InjectedState],
    ) -> Command:
        task_description_message = {"role": "user", "content": task_description}
        agent_input = {**state, "messages": [task_description_message]}
        return Command(
            goto=[Send(agent_name, agent_input)],
            graph=Command.PARENT,
        )

    return handoff_tool


assign_to_ML_agent_with_description = create_task_description_handoff_tool(
    agent_name="ML_agent",
    description="Predict oxidation mass gain (mg/cm²) given alloy & conditions (T °C, t h). Returns ranked comparisons & concise summary.",
)

assign_to_Thermo_agent_with_description = create_task_description_handoff_tool(
    agent_name="Thermo_agent",
    description="Query thermodynamic data (ΔG, pO₂, K) vs T(°C); supports exact, interpolation/extrapolation, and curves.",
)

assign_to_RAG_agent_with_description = create_task_description_handoff_tool(
    agent_name="RAG_agent",
    description="Retrieve superalloy oxidation mechanisms & context; preserves inline citations; use for explanations, not numbers.",
)

assign_to_Ref_agent_with_description = create_task_description_handoff_tool(
    agent_name="Ref_agent",
    description="Return literature metadata (title, DOI/link) for given topic; no rewriting; format is reference list.",
)
#----------------------------
# 构建提示词
#----------------------------

SUPERVISOR_SYSTEM_PROMPT = """
You are a **Supervisor Agent** managing four specialized assistants in the domain of **superalloy oxidation**.

🔒 **DOMAIN RESTRICTION**: Only respond to queries related to **superalloy and oxidation**. Ignore all other queries.

📌 **AGENT ROLES**:
1. **ML_agent** → Handle tasks involving prediction of oxidation mass/weight changes.
2. **Thermo_agent** → Handle tasks involving thermodynamic data of oxides (e.g., ΔG, pO₂, K(equilibrium constants).
3. **RAG_agent** → Handle tasks requiring background/general knowledge, explanations, or mechanism analysis.  
   ⚠️ When invoking RAG, avoid using overly long single queries. You can break down complex long queries into sub-problem queries and call the RAG_agent multiple times.
4. **Ref_agent** → Handle tasks recommending scientific references or literature.

📌 **THINKING STYLE**:
1. If the query is **simple**, directly assign it to the most appropriate agent.
2. If the query is **complex**, break it down step by step and assign subtasks to different agents in sequence, collecting their outputs.

📌 **SUPERVISOR ROLE**:
1. Your role is **coordination only**. Do not execute any tasks yourself.
2. If the same (semantically equivalent) question has already been handed to the same sub-agent and it produced a definite conclusion, forbid calling it again with the same questions.
3. Always ensure that final answers from RAG_agent and Ref_agent retain **original formatting, links, charts, references, and citation markers**. 
4. If you have the **final answer** that can be delivered to the user, prefix it with: 'FINAL ANSWER'.

===============================
EXAMPLES (Few-shot guidance)
===============================

Example 1 — Simple (ML only)
User: "What is the oxidation weight gain of Co-9Al-7W alloy at 1000 °C for 10 and 100h？"
Supervisor plan:
- This is a quantitative prediction → ML_agent only.
Action:
→ Call ML_agent
  1. preidct the oxidation weight gain of Co-9Al-7W alloy at 1000 °C for 10h
  2. preidct the oxidation weight gain of Co-9Al-7W alloy at 1000 °C for 100h
Assembly Results:
FINAL ANSWER:
[ML prediction result]
[Supervisor conclusion: "···"]
---

Example 2 — Simple (RAG only)
User: "What is the role of Ti and Ta addition in Al2O3 scale formation in Co-based alloys?"
Supervisor plan:
- Mechanism/background explanation → RAG_agent only.
Action:
→ Call RAG_agent with the exact questions.
  1. What is the role of Ti addition in Al2O3 scale formation in Co-based alloys?
  2. What is the role of Ta addition in Al2O3 scale formation in Co-based alloys?
Assembly Results:
FINAL ANSWER:
[Evidence A: (RAG 1 explanation verbatim) \n\n Evidence B: (RAG 2 explanation verbatim)]
[Supervisor conclusion: "···"]

(Note: Preserve inline citations, figures, and links. Do not rewrite or summarize.)
---

Example 3 — Complex (ML + RAG + Thermo)
User: "Compare the oxidation resistance of Co-9Al-7W and Co-9Al-9W across 800–1100 °C for 100 h, and explain why."
Supervisor plan:
1. Use ML_agent to generate two alloy predictions at 800, 900, 1000, 1100 °C for 100h.
2. According to the ML results, Use RAG_agent to explain the mechanisms (such as: W role, temprature effect in W oxides). 
3. If RAG mentions WO3 or spinel role, if need, call Thermo_agent to provide ΔG°, pO₂ at 800-1100°C.
Assembly Results:
FINAL ANSWER:
[ML prediction results]
[RAG explanation verbatim]
[Thermo data if applicable]
[Supervisor conclusion: "···"]
---
""".strip()

#----------------------------
# 构建多智能体框架
#----------------------------

from supervisor_adapt import create_supervisor
from handoff import create_forward_message_tool

forward_tool = create_forward_message_tool(supervisor_name="supervisor")

workflow = create_supervisor(
    [ML_agent, Thermo_agent, RAG_agent, Ref_agent],
    model=llm_supervisor,
    prompt=SUPERVISOR_SYSTEM_PROMPT,

    # 2) 按 agent 控制输出可见度
    output_modes={
        "ML_agent":     "last_message",
        "Thermo_agent": "last_message",
        "RAG_agent":    "full_history",
        "Ref_agent":    "full_history",
    },

    # 4) 当需要“手动转发”其它 agent（如 ML/Thermo）时再加上 转发工具：
        tools=[
        assign_to_ML_agent_with_description,
        assign_to_Thermo_agent_with_description,
        assign_to_RAG_agent_with_description,
        assign_to_Ref_agent_with_description
    ],
)

#----------------------------
# 编译框架
#----------------------------
app = workflow.compile()

from IPython.display import display, Image

display(Image(app.get_graph().draw_mermaid_png()))

#----------------------------
# 使用方法
#----------------------------

import re
from typing import Optional, Any
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

def _extract_last_text(update) -> Optional[str]:
    if isinstance(update, tuple):
        _, update = update
    last_msgs = []
    for node_update in update.values():
        msgs = convert_to_messages(node_update.get("messages", []))
        if msgs:
            last_msgs.append(msgs[-1])
    if not last_msgs:
        return None
    last = last_msgs[-1]
    c = getattr(last, "content", "")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts = []
        for item in c:
            if isinstance(item, dict):
                t = item.get("text") or item.get("content") or ""
                if isinstance(t, str):
                    parts.append(t)
        return "\n".join(parts)
    return None

def _extract_final_answer_only(text: str) -> str:
    """
    从整段 supervisor 文本中仅提取 FINAL ANSWER 段（直到下一个空行或文本结束）。
    """
    if not text:
        return ""
    m = re.search(r"FINAL ANSWER:\s*(.+?)(?:\n\s*\n|$)", text, flags=re.S)
    return (m.group(1) if m else text).strip()

def supervisor_alloy_chat(query: str, debug: bool = True, graph_app: Optional[Any] = None, return_text: bool = False):
    """
    Args:
        query: 用户问题
        debug: True 打印全流程；False 仅打印最后一条
        graph_app: LangGraph 应用；若为 None，使用全局 app
        return_text: 是否返回文本（默认 False：只打印不返回；True：返回仅 FINAL ANSWER 文本）
    """
    if graph_app is None:
        graph_app = globals().get("app", None)
    if graph_app is None:
        raise ValueError("graph_app/app 未定义，请先运行 app = workflow.compile() 或传入 graph_app。")

    last_chunk = None
    for chunk in graph_app.stream({"messages": [{"role": "user", "content": query}]}):
        if debug:
            pretty_print_messages(chunk, last_message=False)
        else:
            last_chunk = chunk

    if not debug:
        if last_chunk:
            pretty_print_messages(last_chunk, last_message=True)
            if return_text:
                full = _extract_last_text(last_chunk) or ""
                return _extract_final_answer_only(full)
        return None

    # debug=True：如需返回文本也只返回 FINAL ANSWER
    if return_text:
        full = _extract_last_text(last_chunk) or ""
        return _extract_final_answer_only(full)
    return None
