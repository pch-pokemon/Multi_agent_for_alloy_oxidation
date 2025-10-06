# -*- coding: utf-8 -*-
"""
检索增强生成智能体 (高温合金氧化文献推荐系统)
============================
支持能力：
- 参考文献推荐，返回题目 + key finding + doi 以及总结
- 系统默认基础检索推荐返回文献数量 n <= 5，可变参数 top-k（粗略检索） 与 top-n（精确检索）

数据源：
- 默认从同目录下优先加载 metdadata_ref_documents 文件（代码运行需要与这些文件配合使用）
- 基座 Embedding 模型与 ReRank 模型: BGE_large_en_1.5v; BGE-rerank-large (可前往 Hugging Face 官网查看详细信息)
"""

from __future__ import annotations
import json
from langchain.schema import Document
import os
#-------------------------------------------------
# config 配置
#-------------------------------------------------
from dotenv import load_dotenv
from langchain_community.chat_models import ChatZhipuAI
# 导入 API Key
load_dotenv(r"C:\Users\12279\ZHIPU.env")
api_key = os.getenv('API_KEY')
llm_Ref = ChatZhipuAI(model="glm-4.5-air",api_key=api_key,temperature=0.6)
#--------------------------------------------------
# 以当前文件位置为基准
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent     # .../ref_REC
DATA_DIR = BASE_DIR / "data"                   # .../ref_REC/data

def p_data(*parts) -> Path:
    """拼出包内数据文件的绝对路径"""
    return DATA_DIR.joinpath(*parts)
#--------------------------------------------------
# config 配置
metadata_raw_path = p_data("metdadata_ref_documents.json")
faiss_raw_folder_path = p_data("")  # 向量文件夹地址（默认data文件夹下）
faiss_raw_name = "faiss_metadata_ref"  # 向量文件名称，保证 .faiss 与 .pkl 文件同时存在
embedding_model_raw_path = "D:/BGE_large_en_1.5v"
rerank_model_raw_path = "D:/BGE-rerank-large" 
env_file_raw_path = 'C:/Users/12279/ZHIPU.env' # 默认使用 ZHIPUAI 取'API_KEY'=值
#--------------------------------------------------

#--------- 加载 metdadata_ref 数据库 ---------

# Reading JSON Files
# 读取 JSON 文件

def load_documents_from_json(file_paths):
    documents = []
    
    # Traverse each file path
    # 遍历每个文件路径
    
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Parse JSON and convert back to Document object
        # 解析 JSON，转换回 Document 对象
        
        for item in data:
            doc = Document(
                metadata=item["metadata"], 
                page_content=json.dumps(item["page_content"], ensure_ascii=False)  # Maintain content structure # 保持内容结构
            )
            documents.append(doc)
    
    return documents


json_file_paths = [metadata_raw_path]

# Load all Documents
# 加载所有 Document

documents = load_documents_from_json(json_file_paths)

#--------- 加载 embedding 模型 ---------------

from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

model_path = embedding_model_raw_path
model_name =  model_path
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': True}  
embeddings = HuggingFaceEmbeddings(
    model_name = model_name,
    model_kwargs = model_kwargs,
    encode_kwargs = encode_kwargs
)

#--------- 加载向量文件 ------------

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load Index
# 加载索引

index_folder_path = faiss_raw_folder_path
index_name = faiss_raw_name  # 对应 .faiss 和 .pkl

try:
    vectordb = FAISS.load_local(
        folder_path=index_folder_path,
        embeddings=embeddings,
        index_name=index_name,
        allow_dangerous_deserialization=True
    )
    print("✅ Ref 索引加载成功/Ref Index loading successful!")
except Exception as e:
    print(f"加载失败/Load Fail: {e}")
#-----------------------------------

# Creating a basic vector indexer
# 创建基础向量检索器
faiss_retriever = vectordb.as_retriever(search_kwargs={"k": 10}) # Corresponding to top-k = 10 # 对应 top-k = 10

#-------------------

from langchain.retrievers import BM25Retriever, EnsembleRetriever

# Create a sparse retriever to find relevant documents based on keywords
# 创建稀疏检索器，根据关键词查找相关文档

bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k= 10

# Creating a Hybrid Searcher
# 创建混合检索器

hyb_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], 
                              weight=[0.5, 0.5]) # The weight distribution can be tested by the researcher themselves # 权重分配可由研究者自行测试

#-------- 加载 ReRank 模型 ----------------

from typing import Optional, Sequence
from typing import Tuple  # 如果你在 <=3.9 环境，改用 Tuple[List] 等旧式注解
from langchain_core.documents import Document
from langchain.callbacks.manager import Callbacks
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from sentence_transformers import CrossEncoder
from pydantic import ConfigDict, Field

class BgeRerank(BaseDocumentCompressor):
    """
    Document compressor that uses a locally stored 'BAAI/bge-reranker-large' model
    使用本地存储的 'BAAI/bge-reranker-large' 模型的文档压缩器
    """

    # -------- Pydantic v2 配置 --------
    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,  # 允许存放 CrossEncoder 这类任意对象
    )

    # -------- 字段定义 --------
    top_n: int = 5  # 对应 top-n = 5
    model_path: str = rerank_model_raw_path  # Specify the local path of the re-rank model #指定 re-rank model 本地路径
    model: Optional[CrossEncoder] = None  # 推迟初始化，避免 import 时就加载权重

    # v2: 用 model_post_init 在实例化后加载模型（更安全）
    def model_post_init(self, __context):
        if self.model is None:
            # 如需指定设备，可传 device=...（例如 'cuda' / 'cpu'）
            self.model = CrossEncoder(self.model_path)

    def bge_rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_n: Optional[int] = None,
    ) -> Sequence[tuple[int, float]]:
        """
        根据与查询的相关性对文档进行排序，返回 (索引, 得分)
        """
        if not documents:
            return []

        k = top_n if top_n is not None else self.top_n
        k = max(0, min(k, len(documents)))  # 防越界

        pairs = [[query, doc] for doc in documents]
        scores = self.model.predict(pairs)  # type: ignore[union-attr]
        results = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return results[:k]

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        使用本地 BAAI/bge-reranker-large 对文档进行压缩（重排+截取 top_n）
        """
        if not documents:
            return []

        doc_list = list(documents)
        doc_texts = [d.page_content for d in doc_list]

        results = self.bge_rerank(query=query, documents=doc_texts, top_n=self.top_n)

        final_results = []
        for idx, score in results:
            doc = doc_list[idx]
            # 写入相关性分数，便于后续链路使用
            doc.metadata["relevance_score"] = float(score)
            final_results.append(doc)

        return final_results

#----------- 创建 ReRank 压缩器 -----------------

compressor = BgeRerank()

from langchain.retrievers import ContextualCompressionRetriever

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever = faiss_retriever)

#------------- 加载 llm ----------------------

import os
from dotenv import load_dotenv

# Specify the path of the env file
# 指定 env文件路径

env_file_path = env_file_raw_path

# Load the specified env file
# 加载指定的env文件

load_dotenv(env_file_path)

# Load the API key
# 加载api-key

api_key = os.getenv('API_KEY')

#------------------------------------

from langchain_community.chat_models import ChatZhipuAI

llm =  ChatZhipuAI(
    temperature = 0, 
    model = "glm-4.5",
    api_key=api_key
)

# =========
# 主流程
# =========

import re
from typing import List
from langchain.prompts import PromptTemplate

# --------------- DOI 正则 ---------------

_DOI_URL_RE = re.compile(
    r"https?://(?:dx\.)?doi\.org/[^\s\)\]\}>\"'，。；、,]+",
    re.IGNORECASE,
)
_DOI_BARE_RE = re.compile(r"(10\.\d{4,9}/[^\s\)\]\}>\"'，。；、,]+)", re.IGNORECASE)

# ------------- Prompt 模板设计 ----------------------

_REF_PROMPT = PromptTemplate(
    input_variables=["paper_blocks", "allowed_dois", "question"],
    template=(
        "You are an expert recommending scientific papers on superalloy oxidation.\n"
        "Use ONLY the given papers. If a DOI is not listed under Allowed DOIs, DO NOT output it.\n\n"
        "=== Papers ===\n{paper_blocks}\n"
        "=== Allowed DOIs ===\n{allowed_dois}\n\n"
        "Question: {question}\n\n"
        "Write your answer in the EXACT format below for up to 5 papers:\n"
        "Based on current research findings, Here is about ··· <Your opening statement for the recommendation>\n"
        "1) Title: <paper title>\n"
        "   Key finding: <1–3 sentences focused on oxidation mechanisms/results>\n"
        "   DOI: <one URL from Allowed DOIs>\n"
        "2) Title: ...\n"
        "   Key finding: ...\n"
        "   DOI: ...\n"
        "(Include 2–5 items depending on relevance.)\n\n"
        "Then add a final paragraph:\n"
        "Conclusion: <Concise recommendation conclusion (1–5 sentences)>\n"
    ),
)

# --------------- 工具函数 ---------------

def _ensure_doi_url(s: str) -> str:
    """标准化为 https://doi.org/..."""
    if not s:
        return ""
    s = s.strip().rstrip(".,;")
    if _DOI_URL_RE.fullmatch(s):
        return s
    # 兼容 'doi: xxx' / 'https://dx.doi.org/xxx'
    s = re.sub(r"(?i)^(doi\s*[:=]\s*)", "", s)
    s = re.sub(r"(?i)^https?://dx\.doi\.org/", "https://doi.org/", s)
    if _DOI_URL_RE.fullmatch(s):
        return s
    m = _DOI_BARE_RE.fullmatch(s)
    if m:
        return f"https://doi.org/{m.group(1)}"
    m = _DOI_BARE_RE.search(s)
    return f"https://doi.org/{m.group(1)}" if m else s


def _extract_doi_urls(text: str) -> List[str]:
    """提取 DOI URL（包括裸 DOI 转换为 URL）"""
    seen, out = set(), []
    # 先收 URL
    for m in _DOI_URL_RE.finditer(text or ""):
        url = m.group(0).rstrip(".,;")
        if url not in seen:
            seen.add(url)
            out.append(url)
    # 再收裸 DOI
    for m in _DOI_BARE_RE.finditer(text or ""):
        start = m.start()
        prefix = (text or "")[max(0, start - 40):start].lower()
        if "doi.org/" in prefix:
            continue
        url = f"https://doi.org/{m.group(1).rstrip('.,;')}"
        if url not in seen:
            seen.add(url)
            out.append(url)
    return out


def _urlize_bare_doi_once(text: str) -> str:
    """只把真正裸露的 DOI 变成 URL，避免重复包裹"""
    if not text:
        return text

    def repl(m):
        start = m.start()
        prefix = text[max(0, start - 40):start].lower()
        if "doi.org/" in prefix:
            return m.group(1)
        return f"https://doi.org/{m.group(1)}"

    return _DOI_BARE_RE.sub(repl, text)


def _dedup_numbered_items(text: str) -> str:
    """去除重复的 1)/2)/3) 条目"""
    import hashlib, re as _re
    lines = text.splitlines()
    blocks, cur = [], []

    def _flush():
        if cur:
            blocks.append("\n".join(cur).strip())
            cur.clear()

    numbered_re = _re.compile(r"^\s*\d+\)", _re.IGNORECASE)
    head, tail = [], []
    phase = "pre"

    for ln in lines:
        if phase == "pre":
            if numbered_re.match(ln):
                phase = "items"
                cur.append(ln)
            else:
                head.append(ln)
        elif phase == "items":
            if numbered_re.match(ln):
                _flush()
                cur.append(ln)
            else:
                if ln.strip().lower().startswith("conclusion:") or ln.strip().lower().startswith("references"):
                    _flush()
                    phase = "post"
                    tail.append(ln)
                else:
                    cur.append(ln)
        else:
            tail.append(ln)
    _flush()

    seen, uniq_blocks = set(), []
    for b in blocks:
        h = hashlib.md5(b.encode("utf-8")).hexdigest()
        if h not in seen:
            seen.add(h)
            uniq_blocks.append(b)

    out = []
    if head and any(h.strip() for h in head):
        out.append("\n".join(head).strip())
    if uniq_blocks:
        out.append("\n\n".join(uniq_blocks))
    if tail and any(t.strip() for t in tail):
        out.append("\n".join(tail).strip())
    return "\n\n".join([p for p in out if p.strip()])


def _contains_any_allowed(answer_text: str, allowed: List[str]) -> bool:
    """宽松判断答案里是否出现过 allowed 里的 DOI"""
    low = answer_text.lower()
    return any(u.lower() in low for u in allowed)

def _build_paper_blocks(docs: List) -> tuple[str, List[str]]:
    """把检索到的文献整理为 Prompt 使用的块，并收集 Allowed DOIs"""
    blocks, allowed = [], []
    for i, d in enumerate(docs, 1):
        md = getattr(d, "metadata", {}) or {}
        title = md.get("title") or md.get("document_title") or "(untitled)"
        raw_doi = md.get("doi") or md.get("DOI") or ""
        doi_url = _ensure_doi_url(raw_doi) if raw_doi else ""
        if doi_url:
            allowed.append(doi_url)
        body = (getattr(d, "page_content", "") or "").strip()
        snippet = body[:800]
        blocks.append(
            f"[{i}]\nTitle: {title}\nDOI: {doi_url or '(none)'}\nAbstract/Snippet:\n{snippet}\n"
        )
    return "\n".join(blocks), allowed

# =========
# 主函数
# =========

def print_ref_answer(query: str, debug: bool = False, **kwargs):
    if "degug" in kwargs:  # 兼容手滑
        debug = kwargs["degug"]

    try:
        _ = compression_retriever  # noqa
        _ = llm  # noqa
    except NameError as e:
        raise RuntimeError("请先在全局初始化 compression_retriever 与 llm") from e

    docs = compression_retriever.invoke(query) or []
    if not isinstance(docs, list):
        docs = [docs] if docs else []
    if not docs:
        print("No relevant documents were retrieved, so I cannot provide DOI-backed recommendations.")
        return

    k = min(5, len(docs))
    paper_blocks, allowed = _build_paper_blocks(docs[:k])
    allowed_dois_str = "\n".join(f"- {u}" for u in allowed) if allowed else "(none)"

    chain = _REF_PROMPT | llm
    raw = chain.invoke({
        "paper_blocks": paper_blocks,
        "allowed_dois": allowed_dois_str,
        "question": query
    })
    answer_text = getattr(raw, "content", raw) or ""

    # 统一处理 DOI
    answer_text = _urlize_bare_doi_once(answer_text)
    # 去重
    answer_text = _dedup_numbered_items(answer_text)

    # 白名单过滤
    if allowed:
        found = _extract_doi_urls(answer_text)
        illegal = [u for u in found if u not in set(allowed)]
        for bad in illegal:
            answer_text = answer_text.replace(bad, "[DOI not in allowed list]")
    else:
        for u in _extract_doi_urls(answer_text):
            answer_text = answer_text.replace(u, "[DOI unavailable in context]")

    # References 追加逻辑
    present = set(_extract_doi_urls(answer_text))
    has_any_allowed = _contains_any_allowed(answer_text, allowed) if allowed else False
    if allowed and not present and not has_any_allowed:
        refs = "\n".join(f"- {u}" for u in allowed)
        answer_text = answer_text.rstrip() + "\n\nReferences (DOI URLs from context):\n" + refs

    # === 输出主体内容 ===
    print(answer_text.strip())

    # === DEBUG REPORT ===
    if debug:
        print("\n[DBG]-------DEBUG REPORT-------")
        print(f"[DBG] Retrieved={len(docs)}, used_topk={k}")
        print(f"[DBG] Allowed DOIs: {allowed}")

# ------------------------------------

from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
#(return_direct=True) 使用 return_direct=True，确保答案直接通过 tool 输出，不经过llm改写
@tool(return_direct=True)
def print_ref_answer_tool(query: str, debug: bool = False) -> str:
    """
    📌 Purpose:
        Call ref_REC.print_ref_answer and directly return its output without rewriting.

    ⚙️ Usage:
        - Input query: Recommend literature related to the oxidation of high-temperature alloys
        - Optional parameter debug: Whether to print debug logs

    ⚠️ Notes:
        - This tool is only for recommending literature on high-temperature superalloy oxidation.
        - If the question is outside this scope, the Agent should return "I don't know".
    """
    import io, contextlib

    buf = io.StringIO()
    result_str = None
    with contextlib.redirect_stdout(buf):
        result = print_ref_answer(query, debug=debug)
        if isinstance(result, str) and result.strip():
            result_str = result.strip()

    if result_str:
        return result_str
    return buf.getvalue().strip()


# ------------------ 提示词设定 ------------------

REF_REC_AGENT_PROMPT = """
You are a Ref_agent specialized in oxidation knowledge of superalloys.

- ALWAYS solve the user request by calling the tool `print_ref_answer_tool`.
- Do not rewrite or summarize the tool output. The tool result is the final answer.
- Handle questions related to the literature recommendations of superalloys oxidation.
- If the question is outside your scope, reply : "I don't know, other agents may help this question".
""".strip()

# 定义智能体和工具

def build_ref_agent(create_react_agent, llm_Ref):
    tools=[print_ref_answer_tool]
    agent = create_react_agent(
        model=llm_Ref,
        tools=tools,
        prompt=REF_REC_AGENT_PROMPT,
        name="Ref_agent",
    )
    return agent

Ref_agent = build_ref_agent(create_react_agent, llm_Ref)
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

def ref_alloy_chat(query: str, debug: bool = True) -> Optional[str]:
    """
    向 Ref_agent 发送一次性问题并按需打印：
      - debug=True : 完整打印全部 streaming 过程
      - debug=False: 仅返回最后一个 AI 答案
    """
    final_ai_message: Optional[BaseMessage] = None

    if debug:
        # 完整过程打印
        for chunk in Ref_agent.stream(
            {"messages": [{"role": "user", "content": query}]}
        ):
            pretty_print_messages(chunk, last_message=False)
            last_ai = _extract_last_ai_message(chunk)
            if last_ai is not None:
                final_ai_message = last_ai
    else:
        # 静默收集，直到结束只返回最终答案
        for chunk in Ref_agent.stream(
            {"messages": [{"role": "user", "content": query}]}
        ):
            last_ai = _extract_last_ai_message(chunk)
            if last_ai is not None:
                final_ai_message = last_ai

    return getattr(final_ai_message, "content", None)