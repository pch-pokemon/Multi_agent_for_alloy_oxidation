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

#--------------------------------------------------
# 以当前文件位置为基准
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent     # .../ref_REC
DATA_DIR = BASE_DIR / "data"                   # .../ref_REC/data

def p_data(*parts) -> Path:
    """拼出包内数据文件的绝对路径"""
    return DATA_DIR.joinpath(*parts)
#--------------------------------------------------

# =========
# config 配置
# =========

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


json_file_paths = [p_data("metdadata_ref_documents.json")]

# Load all Documents
# 加载所有 Document

documents = load_documents_from_json(json_file_paths)

#--------- 加载 embedding 模型 ---------------

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

model_path = "D:/BGE_large_en_1.5v"
model_name =  model_path
model_kwargs = {'device': 'cuda'}
encode_kwargs = {'normalize_embeddings': True}  
embeddings = HuggingFaceBgeEmbeddings(
    model_name = model_name,
    model_kwargs = model_kwargs,
    encode_kwargs = encode_kwargs
)

#--------- 加载向量文件 ------------

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load Index
# 加载索引

index_folder_path = p_data("")
index_name = "faiss_metadata_ref"  # 对应 .faiss 和 .pkl

try:
    vectordb = FAISS.load_local(
        folder_path=index_folder_path,
        embeddings=embeddings,
        index_name=index_name,
        allow_dangerous_deserialization=True
    )
    print("索引加载成功/Index loading successful!")
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

from typing import Dict, Optional, Sequence
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import Extra
from langchain.callbacks.manager import Callbacks
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from sentence_transformers import CrossEncoder

class BgeRerank(BaseDocumentCompressor):
    
    """Document compressor that uses a locally stored 'BAAI/bge-reranker-large' model
       使用本地存储的 'BAAI/bge-reranker-large' 模型的文档压缩器"""

    top_n: int = 5 # Corresponding to top-n = 5 #对应 top-n = 5

    model_path: str = "D:/BGE-rerank-large"  # Specify the local path of the re-rank model #指定 re-rank model 本地路径

    model: CrossEncoder = CrossEncoder(model_path)  # Initialize using the local model path #使用本地模型路径初始化
    
    class Config:
        
        """Configuration for this pydantic object
           Pydantic 对象的配置"""

        extra = Extra.forbid
        arbitrary_types_allowed = True
        
    def bge_rerank(self, query: str, documents: Sequence[str], top_n: int) -> Sequence[tuple[int, float]]:
        
        """
        Rank documents based on relevance to the query
        根据与查询的相关性对文档进行排序

        Args:
        参数：
        
            query: The input query
            查询：输入的查询
            
            documents: List of document contents
            文档：文档内容列表
            
            top_n: Number of top-ranked documents to return
            top_n：返回的排名靠前的文档数量

        Returns:
        返回：
        
            A sorted list of document indices and scores
            按顺序排列的文档索引及得分列表
            
        """
        
        pairs = [[query, doc] for doc in documents]
        scores = self.model.predict(pairs)
        results = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)  # Sort by score in descending order # 按得分降序排列
        return results[:top_n]
    
    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        
        """
        Compress documents using the locally stored BAAI/bge-reranker-large
        使用本地存储的 BAAI/bge-reranker-large 压缩文档

        Args:
        参数：
        
            documents: A sequence of documents to compress
            文档：要压缩的一系列文档
            
            query: The query to use for compressing the documents
            查询：用于压缩文档的查询语句
            
            callbacks: Callbacks to run during the compression process
            回调：压缩过程中要运行的回调

        Returns:
        返回：
        
            A sequence of compressed documents
            一系列压缩文档
            
        """
        
        if len(documents) == 0:
            return []
        
        doc_list = list(documents)
        doc_texts = [d.page_content for d in doc_list]

        results = self.bge_rerank(query=query, documents=doc_texts, top_n=self.top_n)

        final_results = []
        for idx, score in results:
            doc = doc_list[idx]
            doc.metadata["relevance_score"] = score
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

env_file_path = 'C:/Users/12279/ZHIPU.env'

# Load the specified env file
# 加载指定的env文件

load_dotenv(env_file_path)

# Load the API key
# 加载api-key

api_key = os.getenv('API_KEY')

#------------------------------------

from langchain_community.chat_models import ChatZhipuAI

llm =  ChatZhipuAI(
    temperature = 0.6, 
    model = "glm-4.5",
    api_key=api_key
)
print(llm.model_name)

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

    if debug:
        print(f"[DBG] Retrieved={len(docs)}, used_topk={k}")
        print(f"[DBG] Allowed DOIs: {allowed}")

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

    print(answer_text.strip())

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