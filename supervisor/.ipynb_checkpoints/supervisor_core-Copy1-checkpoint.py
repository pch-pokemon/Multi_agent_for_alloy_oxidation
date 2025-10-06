
# -*- coding: utf-8 -*-
"""
高温合金专家多智能体 
角色：协调者
============================
支持能力：
- 基于用户问题的复杂度 自适应调用 机器学习智能体，热力学智能体，RAG与Ref子智能体 来解决问题

参数支持：
- degug 参数：默认为True，打印协调者如何思考并分配给各个智能体解决问题的过程。设置为 False，仅显示最终答案，对界面端友好

重要变动：
- 修改源官网 langgraph.supervisor 底层代码，增加为每个 子智能体 设置各自的 output_mode 方法 控制协调者的信息可见范围，降低ml和thermo智能体过多工具调用信息对协调者的干扰
- 修改源官网 langgraph.handoff 底层代码，增加协调者为下一个智能体的任务描述，增强 debug=True 过程的可读性

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
我们强烈建议在 没有使用 微调专业知识领域的llm时，为协调者设置为最高级的llm, 如GPT-5。作为大脑角色，其是思考设计问题如何解决的核心人物。
"""
# from langchain_openai import ChatOpenAI
# load_dotenv('C:/Users/12279/OPENAI.env')
# openai_api_key = os.getenv('API_KEY')
# llm_supervisor = ChatOpenAI(model="gpt-5",temperature=0.5,api_key= openai_api_key, base_url="https://api.chatanywhere.tech/v1") # ★ 核心 llm

llm_supervisor = ChatZhipuAI(model="glm-4.5",api_key=api_key, temperature=0.6) # ★ 核心 llm
llm_ML = ChatZhipuAI(model="glm-4-air",api_key=api_key,temperature=0.6)
llm_Thermo = ChatZhipuAI(model="glm-4-air",api_key=api_key,temperature=0.6)
llm_RAG = ChatZhipuAI(model="glm-4.5",api_key=api_key,temperature=0.6)
llm_Ref = ChatZhipuAI(model="glm-4.5",api_key=api_key,temperature=0.6)

#-------------------------------
# 建立各个子智能体
#---------------------------
import sys
sys.path.append("..") 

from langgraph.prebuilt import create_react_agent
from ml_predict.ml import build_ml_agent
ML_agent = build_ml_agent(create_react_agent, llm_ML)
print(f"llm_ML 已准备就绪: {llm_ML.model_name}")

from thermo.thermo_query import build_thermo_agent
Thermo_agent = build_thermo_agent(create_react_agent, llm_Thermo)
print(f"llm_Thermo 已准备就绪: {llm_Thermo.model_name}")

from rag_core.rag_core import build_rag_agent
RAG_agent = build_rag_agent(create_react_agent, llm_RAG)
print(f"llm_RAG 已准备就绪: {llm_RAG.model_name}")

from ref_rec.ref_REC import build_ref_agent
Ref_agent = build_ref_agent(create_react_agent, llm_Ref)
print(f"llm_Ref 已准备就绪: {llm_Ref.model_name}")

#----------------------------
# 创建转接工具
#----------------------------

from handoff_revise import create_handoff_tool

assign_to_ML_agent_with_description = create_handoff_tool(
    agent_name="ML_agent",
    name="assign_to_ML_agent",
    description="Predict oxidation mass gain (mg/cm²) given alloy & conditions (T °C, t h). Returns ranked comparisons & concise summary.",
)

assign_to_Thermo_agent_with_description = create_handoff_tool(
    agent_name="Thermo_agent",
    name="assign_to_Thermo_agent",
    description="Query thermodynamic data (ΔG, pO₂, K) vs T(°C); supports exact, interpolation/extrapolation, and curves.",
)

assign_to_RAG_agent_with_description = create_handoff_tool(
    agent_name="RAG_agent",
    name="assign_to_RAG_agent",
    description="Retrieve superalloy oxidation mechanisms & context; preserves inline citations; use for explanations, not numbers.",
)

assign_to_Ref_agent_with_description = create_handoff_tool(
    agent_name="Ref_agent",
    name="assign_to_Ref_agent",
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
2. If the query is **complex**, break it down step by step and assign subtasks to agents in sequence, collecting their outputs.

📌 **SUPERVISOR ROLE**:
1. Your role is **coordination only**. BUT Do not call multiple agents at once. Do not execute any tasks yourself.
2. Always ensure that final answers from RAG_agent and Ref_agent retain **original formatting, links, charts, references, and citation markers**. 
3. If you have the **final answer** that can be delivered, prefix it with: 'FINAL ANSWER'.

===============================
EXAMPLES (Few-shot guidance)
===============================

Example 1 — Simple (ML only)
User: "What is the oxidation weight gain of Co-9Al-7W alloy at 1000 °C for 10 and 100h？"
Supervisor plan:
- quantitative prediction → ML_agent only.
Action:
→ Call ML_agent: preidct the oxidation weight gain of Co-9Al-7W alloy at 1000 °C for 10h and 100h
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
  1. What is the role of Ti and Ta addition in Al2O3 scale formation in Co-based alloys? (If the answer is not clear, you can call RAG for further clarification.)
  2. What is the role of Ti addition in Al2O3 scale formation in Co-based alloys?
  3. What is the role of Ta addition in Al2O3 scale formation in Co-based alloys?
Assembly Results:
FINAL ANSWER:
[Evidence A: (RAG 1 explanation verbatim) \n\n Evidence B: (RAG 2 explanation verbatim)]
[Supervisor conclusion: "···"]

(Note: Preserve inline citations, figures, and links. Do not rewrite or summarize.)
---

Example 3 — Complex (ML + RAG + Thermo)
User: "Compare the oxidation resistance of Co-9Al-7W and Co-9Al-9W at 900 °C, and explain why."
Supervisor plan:
This is an abstract and complex issue.
1. First, I need to call ML_agent to make a prediction and obtain a rough conclusion. 
2. Then, I will call RAG_agent to explain the underlying mechanism. 
3. If necessary, I might also need to call Thermo_agent for further explanation and verification.
Action:
→ Call ML_agent ：preidct the oxidation weight gain of Co-9Al-7W and Co-9Al-9W alloy at 900 °C for 100h(Uniform default value)
→ Call RAG_agent: 
1. What is the role of W in Co-Al-W alloy system? Beneficial or harmful?
2. Why does the oxidation resistance of the Co-Al-W alloy increase (decrease) as the W content increases from 7at.% to 9at.%?
3. ···etc.(Until the answer is clear.)
→ Call Thermo_agent(if applicable)
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

from supervisor_revise import create_supervisor
from handoff_revise import create_forward_message_tool

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

    # 4) 当需要“手动转发”时再加上 转发工具：
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

display(Image(app.get_graph().draw_mermaid_png())) # 显示编译的框架图

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
def supervisor_alloy_chat(query: str, debug: bool = True) -> None:
    """
    debug=True  -> 流式打印全流程（与你现有 pretty_print_messages 输出一致）
    debug=False -> 只打印最后一条消息（直接从最后一个有效子图chunk中抽取）
    """
    if debug:
        for chunk in app.stream(
            {"messages": [{"role": "user", "content": query}]},
            subgraphs=True,
        ):
            pretty_print_messages(chunk, last_message=False)
        return

    # --- debug=False：只显示最后一条消息 ---
    last_any_update = None        # 记录最后一个（可能是父图）update
    last_subgraph_update = None   # 记录最后一个“非空namespace”的子图update（优先）

    for chunk in app.stream(
        {"messages": [{"role": "user", "content": query}]},
        subgraphs=True,
    ):
        if isinstance(chunk, tuple):
            ns, update = chunk
            last_any_update = update
            if len(ns) > 0:
                last_subgraph_update = update
        else:
            # 某些实现可能直接返回 dict
            last_any_update = chunk
            last_subgraph_update = chunk

    update = last_subgraph_update or last_any_update
    if not update:
        print("(no messages)")
        return

    # 取该 update 里“最后一个节点”的“最后一条消息”
    last_pair = None
    for node_name, node_update in update.items():
        last_pair = (node_name, node_update)
    if not last_pair:
        print("(no messages)")
        return

    _, node_update = last_pair
    messages = convert_to_messages(node_update.get("messages", []))
    if not messages:
        print("(no messages)")
        return

    # 打印最后一条消息
    pretty_print_message(messages[-1], indent=False)