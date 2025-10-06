# -*- coding: utf-8 -*-
"""
检索增强生成智能体 (高温合金氧化知识问答系统)
============================
支持能力：
- 基础能力回答，多模态图表返回，参考文献返回
- 可选参数支持 debug = True （打印调试过程）；render_markdown = True （图表链接渲染）

数据源：
- 默认从同目录下优先加载 metadata数据库、doc数据库、chart图表数据库、向量数据文件（代码运行需要与这些文件配合使用）
- 基座 Embedding 模型与 ReRank 模型: BGE_large_en_1.5v; BGE-rerank-large (可前往 Hugging Face 官网查看详细信息)
"""

from __future__ import annotations
import json
from langchain.schema import Document
#--------------------------------------------------
# config 配置
#--------------------------------------------------
import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatZhipuAI
# 导入 API Key
load_dotenv(r"C:\Users\12279\ZHIPU.env")
api_key = os.getenv('API_KEY')
llm_RAG = ChatZhipuAI(model="glm-4-plus",api_key=api_key,temperature=0.6)
#--------------------------------------------------
# 以当前文件位置为基准
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent     # .../rag_core
DATA_DIR = BASE_DIR / "data"                   # .../rag_core/data

def p_data(*parts) -> Path:
    """拼出包内数据文件的绝对路径"""
    return DATA_DIR.joinpath(*parts)
#--------------------------------------------------
metadata_raw_path = p_data("1_100_metadata.json")
doc_raw_path= p_data("1_100_doc.json")
chart_raw_path = p_data("1_100_chart.json")
faiss_raw_folder_path = p_data("")  # 向量文件夹地址（默认data文件夹下）
faiss_raw_name = "index"  # 向量文件名称，保证 .faiss 与 .pkl 文件同时存在
embedding_model_raw_path = "D:/BGE_large_en_1.5v"
rerank_model_raw_path = "D:/BGE-rerank-large" 
env_file_raw_path = 'C:/Users/12279/ZHIPU.env' # 默认使用 ZHIPUAI 取'API_KEY'=值
#--------------------------------------------------

#--------- 加载 DOC(bm25), metadata, 图表数据库 ---------

# Reading JSON Files
# 读取 JSON 文件

def load_documents_from_json(file_paths):
    data_db = []
    
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
            data_db.append(doc)
    
    return data_db

# The current code only performs on the specified demo JSON document
# 目前的代码只进行指定演示的 JOSN 文档

metdadata_json_file_paths = [metadata_raw_path]
doc_json_file_paths = [doc_raw_path]
chart_json_file_paths = [chart_raw_path]

# Load all Documents
# 加载所有 Document

metadata_db = load_documents_from_json(metdadata_json_file_paths)
documents = load_documents_from_json(doc_json_file_paths)
chart_db = load_documents_from_json(chart_json_file_paths)

# ------ 2) 加载 embedding 模型与 向量数据库 ---------

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

# ---------- 向量数据库地址 ----------

from langchain_community.vectorstores import FAISS

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
    print("✅ RAG 索引加载成功/RAG Index loading successful!")
except Exception as e:
    print(f"加载失败/Load Fail: {e}")

# --------------------------------

# Creating a basic vector indexer
# 创建基础向量检索器

faiss_retriever = vectordb.as_retriever(search_kwargs={"k": 10}) # Corresponding to top-k = 10 # 对应 top-k = 10

from langchain.retrievers import BM25Retriever, EnsembleRetriever

# Create a sparse retriever to find relevant documents based on keywords
# 创建稀疏检索器，根据关键词查找相关文档

bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k= 10

# Creating a Hybrid Searcher
# 创建混合检索器

hyb_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], 
                              weight=[0.5, 0.5]) # The weight distribution can be tested by the researcher themselves # 权重分配可由研究者自行测试

# ------------- 加载 ReRank 模型 --------------------

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

# -------------- 建立 ReRank 压缩器 -----------------

compressor = BgeRerank()

from langchain.retrievers import ContextualCompressionRetriever

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever = hyb_retriever)

# -------------- 加载 LLM -------------------------

import os
from dotenv import load_dotenv

# Specify the path of the env file
# 指定 env文件路径

env_file_path = env_file_raw_path
load_dotenv(env_file_path)

api_key = os.getenv('API_KEY')

from langchain_community.chat_models import ChatZhipuAI

llm =  ChatZhipuAI(
    temperature = 0, 
    model = "glm-4.5",
    api_key=api_key
)

# =========
# 主流程
# =========

import os, json, re
from typing import List, Dict, Tuple

# --- Pydantic v2 ---
from pydantic import BaseModel, PrivateAttr, model_validator

# --- LangChain (v0.2+ friendly imports；若环境未升级，保留旧路径) ---
try:
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
except Exception:
    # 回退以兼容旧版本
    from langchain.schema import Document, BaseRetriever

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Jupyter Markdown 渲染（可点击链接）
try:
    from IPython.display import display, Markdown
    _HAS_IPY = True
except Exception:
    _HAS_IPY = False
    def display(*args, **kwargs): pass
    def Markdown(x): return x

# -----------------------
# Utils
# -----------------------
def normalize_source(s: str) -> str:
    if not s:
        return ""
    return os.path.normpath(str(s)).replace("\\", "/").strip()

def _to_dict(obj):
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, str):
        try:
            return json.loads(obj)
        except Exception:
            return {}
    return {}

# 支持多级编号：1, 1.1, 1.2.3 ...
_num_pat = re.compile(r"(\d+(?:\.\d+)*)")

def _num_token(s: str) -> str:
    if not s:
        return ""
    m = _num_pat.search(str(s))
    return m.group(1) if m else ""

def _num_key(s: str) -> Tuple[int, ...]:
    tok = _num_token(s)
    if not tok:
        return tuple()
    return tuple(int(x) for x in tok.split("."))

def _normalize_labels(raw: str, kind: str) -> List[str]:
    """
    归一化标签：支持 'Fig. 7' / '7' / 'Fig. 1.1' / '1.2.3' / 'Table 2.3' ...
    生成可命中键（原样、标准化、纯编号）。
    kind in {'figure','table'}。
    """
    s = str(raw).strip()
    n = _num_token(s)
    labels = {s}
    if n:
        if kind == "figure":
            labels.add(f"Fig. {n}")
            labels.add(f"Figure {n}")
        elif kind == "table":
            labels.add(f"Table {n}")
        labels.add(n)
    return list(labels)

# 去掉 caption 前缀（Fig./Figure/Table + 编号）
_CAPTION_PREFIX_RE = re.compile(
    r"""
    ^\s*
    (?:
        # 1) Fig./Figure 变体（可选复数/点号），允许 S/A 等字母前缀 + 数字
        (?:Fig(?:ure)?s?\.?\s*)
        (?:[A-Za-z]{1,3}\s*)?            # 可选字母前缀：S / A / SI / Suppl 等
        \d+(?:\.\d+)*[A-Za-z]?           # 主编号，允许尾随字母（如 1a）
        (?:\s*\([a-z]\))?                # 可选 (a)/(b) 子图标记
        [\)\.]?
      |
        # 2) Table 变体（可选复数/点号），同理允许字母前缀 + 数字
        (?:Table(?:s)?\.?\s*)
        (?:[A-Za-z]{1,3}\s*)?
        \d+(?:\.\d+)*[A-Za-z]?
        (?:\s*\([a-z]\))?
        [\)\.]?
      |
        # 3) 裸的字母+数字作为开头：S1 / A1 / S1a / A1(b) / S1.2 等
        (?:[A-Za-z]{1,3}\s*\d+(?:\.\d+)*[A-Za-z]?(?:\s*\([a-z]\))?[\)\.]?)
    )
    \s*[:\-–—]?\s*
    """,
    re.IGNORECASE | re.VERBOSE,
)


def clean_caption(caption: str) -> str:
    if not caption:
        return caption
    return _CAPTION_PREFIX_RE.sub("", caption).strip()

def _truncate(s: str, max_len: int = 120) -> str:
    s = (s or "").strip()
    return (s[:max_len] + " …") if len(s) > max_len else s

def _short(s: str, n: int = 80) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[:n] + " …"

# =========
# 多模态检索逻辑
# =========
class ChartEnhancedRetriever(BaseRetriever, BaseModel):
    base_retriever: BaseRetriever
    chart_db: List[Document]
    chart_index: Dict[Tuple[str, str, str], Document] = {}
    debug: bool = False

    _logs: List[Dict] = PrivateAttr(default_factory=list)

    class Config:
        arbitrary_types_allowed = True

    def _log(self, stage: str, **data):
        if self.debug:
            self._logs.append({"stage": stage, **data})

    def get_logs(self) -> List[Dict]:
        return list(self._logs)

    @model_validator(mode="before")  # Pydantic v2
    def build_chart_index(cls, values):
        chart_db = values.get("chart_db", [])
        debug = values.get("debug", False)
        index = {}
        tmp_logs = []

        def _tlog(**d):
            if debug:
                tmp_logs.append({"stage": "build_index", **d})

        for doc in chart_db:
            src = normalize_source(doc.metadata.get("source"))
            if not src:
                _tlog(event="skip_chart_no_source"); continue
            md = doc.metadata

            # Tables
            if md.get("tables_num"):
                raw = str(md["tables_num"]).strip()
                for k in _normalize_labels(raw, "table"):
                    index[(src, k, "table")] = doc
                    _tlog(key=(src, k, "table"),
                          caption=_short(_to_dict(doc.page_content).get("caption") or md.get("caption","")),
                          url=_to_dict(doc.page_content).get("table_url") or md.get("table_url") or md.get("url"))

            # Figures
            if md.get("figures_num"):
                raw = str(md["figures_num"]).strip()
                for k in _normalize_labels(raw, "figure"):
                    index[(src, k, "figure")] = doc
                    _tlog(key=(src, k, "figure"),
                          caption=_short(_to_dict(doc.page_content).get("caption") or md.get("caption","")),
                          url=_to_dict(doc.page_content).get("image_url") or md.get("image_url") or md.get("url"))

        values["chart_index"] = index
        values["_tmp_build_logs"] = tmp_logs
        return values

    def __init__(self, **data):
        tmp_logs = data.pop("_tmp_build_logs", [])
        super().__init__(**data)
        for lg in tmp_logs:
            self._log(**lg)

    # 按首次出现顺序收集来源（供白名单与兜底用）
    def _collect_sources_in_order_for_docs(self, docs: List[Document]) -> List[str]:
        order, seen = [], set()
        for d in docs:
            s = normalize_source(d.metadata.get("source"))
            if s and s not in seen:
                seen.add(s); order.append(s)
        return order

    def _get_relevant_documents(self, query: str, **kwargs) -> List[Document]:
        base_docs = self.base_retriever.invoke(query, **kwargs)
        self._log("retrieve_base", query=query, base_docs=len(base_docs))

        seen_base   = set()  # 去重基础文档
        seen_charts = set()  # 去重 [CHART] 标记 (src, kind, raw_no)
        enhanced_docs: List[Document] = []

        for idx, doc in enumerate(base_docs, 1):
            src = normalize_source(doc.metadata.get("source"))
            self._log("base_doc", idx=idx, source=src, meta_keys=list(doc.metadata.keys()))

            # 基础文档注入
            base_key = (src, hash(doc.page_content))
            if base_key not in seen_base:
                seen_base.add(base_key)
                meta = dict(doc.metadata); meta["source"] = src
                enhanced_docs.append(Document(page_content=doc.page_content, metadata=meta))

            refs = (doc.metadata.get("ref_Fig/Table") or {})
            figs = refs.get("figures", []) or []
            tabs = refs.get("tables", []) or []
            self._log("refs_in_doc", idx=idx, source=src, figures=figs, tables=tabs)

            # ---- Tables ----
            for t in tabs:
                hit_doc = None; tried = []
                for label in _normalize_labels(t, "table"):
                    tried.append(label)
                    hit_doc = self.chart_index.get((src, label, "table"))
                    if hit_doc: break
                if not hit_doc:
                    self._log("miss_table_index", idx=idx, source=src, requested=t, tried=tried)
                    continue

                raw_no = str(hit_doc.metadata.get("tables_num", ""))
                chart_key = (src, "table", raw_no)
                if chart_key in seen_charts:
                    self._log("dup_table_injection", idx=idx, source=src, requested=t, raw_no=raw_no)
                    continue
                seen_charts.add(chart_key)

                payload = _to_dict(hit_doc.page_content)
                cap_raw = payload.get("caption", "") or hit_doc.metadata.get("caption", "")
                url_tbl = payload.get("table_url", "") or hit_doc.metadata.get("table_url", "") or hit_doc.metadata.get("url", "")
                cap_clean = clean_caption(cap_raw)

                enhanced_docs.append(Document(
                    page_content=(
                        "[CHART] type=table; "
                        f"no={raw_no}; "
                        f"caption={_truncate(cap_clean)}; "
                        f"url={url_tbl}; "
                        f"source={src}"
                    ),
                    metadata={
                        "source": src,
                        "__chart__": {
                            "kind": "table",
                            "raw_no": raw_no,
                            "caption": cap_raw,
                            "url": url_tbl,
                        },
                    },
                ))
                self._log("inject_table", idx=idx, source=src, label=t, raw_no=raw_no,
                          cap=_short(cap_raw), url=url_tbl)

            # ---- Figures ----
            for f in figs:
                hit_doc = None; tried = []
                for label in _normalize_labels(f, "figure"):
                    tried.append(label)
                    hit_doc = self.chart_index.get((src, label, "figure"))
                    if hit_doc: break
                if not hit_doc:
                    self._log("miss_figure_index", idx=idx, source=src, requested=f, tried=tried)
                    continue

                raw_no = str(hit_doc.metadata.get("figures_num", ""))
                chart_key = (src, "figure", raw_no)
                if chart_key in seen_charts:
                    self._log("dup_figure_injection", idx=idx, source=src, requested=f, raw_no=raw_no)
                    continue
                seen_charts.add(chart_key)

                payload = _to_dict(hit_doc.page_content)
                cap_raw = payload.get("caption", "") or hit_doc.metadata.get("caption", "")
                url_fig = payload.get("image_url", "") or hit_doc.metadata.get("image_url", "") or hit_doc.metadata.get("url", "")
                cap_clean = clean_caption(cap_raw)

                enhanced_docs.append(Document(
                    page_content=(
                        "[CHART] type=figure; "
                        f"no={raw_no}; "
                        f"caption={_truncate(cap_clean)}; "
                        f"url={url_fig}; "
                        f"source={src}"
                    ),
                    metadata={
                        "source": src,
                        "__chart__": {
                            "kind": "figure",
                            "raw_no": raw_no,
                            "caption": cap_raw,
                            "url": url_fig,
                        },
                    },
                ))
                self._log("inject_figure", idx=idx, source=src, label=f, raw_no=raw_no,
                          cap=_short(cap_raw), url=url_fig)

        # === 在返回前注入 "允许引用的来源白名单" 控制文档 ===
        ctx_sources_ordered = self._collect_sources_in_order_for_docs(base_docs)
        whitelist_doc = Document(
            page_content="[ALLOWED_SOURCES]" + json.dumps(ctx_sources_ordered, ensure_ascii=False),
            metadata={"source": "__control__"}
        )
        enhanced_docs.insert(0, whitelist_doc)
        self._log("inject_whitelist", allowed=ctx_sources_ordered)

        self._log("retrieve_done", enhanced_docs=len(enhanced_docs))
        return enhanced_docs

# =========
# 元数据索引 Metadata index: source -> {title, doi}
# =========

def build_metadata_index(metadata_db: List[Document]) -> Dict[str, Dict[str, str]]:
    idx: Dict[str, Dict[str, str]] = {}
    for d in metadata_db:
        src = normalize_source(d.metadata.get("source"))
        if not src: continue
        payload = _to_dict(d.page_content)
        if "document_title" in payload:
            title = (payload.get("document_title") or "").strip()
            if title: idx.setdefault(src, {})["title"] = title
        if "journal information" in payload:
            info = payload["journal information"]
            if isinstance(info, dict):
                doi = info.get("doi", "")
                if doi: idx.setdefault(src, {})["doi"] = str(doi).strip()
            elif isinstance(info, list):
                for item in info:
                    if isinstance(item, dict) and item.get("doi"):
                        idx.setdefault(src, {})["doi"] = str(item["doi"]).strip()
    return idx

# =========
# Prompt 设计 (白名单 + 禁止越权引用)
# =========

QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a precise materials-science assistant. Use the context to answer concisely and correctly.

Notes:
- There will be a single line starting with [ALLOWED_SOURCES] followed by a JSON array of sources present in this context. You MUST ONLY cite sources from that list. Citing anything else is forbidden.
- The context may contain numeric bracketed citations like [12], [12–14]. You MUST NOT copy those. Only use your own [1]-style markers tied to the <SOURCES> block order.
- Some context lines start with [CHART] and include type/no/caption/url/source of figures/tables. You may rely on them as evidence.
- Figures/Tables in the body: DO NOT use original numbers (e.g., "Fig. 7"). Instead, assign sequential numbers starting at 1 in the order you use them (Fig.1, Fig.2; Table 1, Table 2) and use those in the prose. The exact mapping will be provided in the final <CHARTS> block.
- References in the body: when you rely on a source, add a superscript-like marker using square brackets (e.g., [1], [2]) that corresponds to the order of first use in the body. The exact mapping will be provided in the final <SOURCES> block.
- At the very end, output exactly TWO machine-readable lines (and nothing after them), both in the order you used them in the body:
  1) <CHARTS>{{"figures":[{{"source":"D:/···","label":"Fig. X"}}],
               "tables":[{{"source":"D:/···","label":"Table X"}}]}}</CHARTS>
  2) <SOURCES>{{"sources":["D:/···","D:/···"]}}</SOURCES>
- If none were used, output empty arrays:
  <CHARTS>{{"figures":[],"tables":[]}}</CHARTS>
  <SOURCES>{{"sources":[]}}</SOURCES>
- Do not add any text after </SOURCES>.

[Context]
{context}

[Question]
{question}

Answer:
"""
)

# -------- 规范化 source（避免路径分隔符差异）----------

for d in chart_db:
    if "source" in d.metadata:
        d.metadata["source"] = normalize_source(d.metadata["source"])
for d in metadata_db:
    if "source" in d.metadata:
        d.metadata["source"] = normalize_source(d.metadata["source"])

chart_enhanced_retriever = ChartEnhancedRetriever(
    base_retriever=compression_retriever,
    chart_db=chart_db,
    debug=True,  # internal logs on
)

chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=chart_enhanced_retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": QA_PROMPT},
    return_source_documents=True
)

metadata_index = build_metadata_index(metadata_db)

# ---------- 解析图表和 source 区块 ------------

_CHARTS_BLOCK_RE  = re.compile(r"<CHARTS>\s*(\{.*?\})\s*</CHARTS>", re.DOTALL)
_SOURCES_BLOCK_RE = re.compile(r"<SOURCES>\s*(\{.*?\})\s*</SOURCES>", re.DOTALL)

def parse_chart_citations(answer_text: str) -> Dict[str, List[Dict[str, str]]]:
    m = _CHARTS_BLOCK_RE.search(answer_text)
    if not m:
        return {"figures": [], "tables": []}
    raw = m.group(1)
    try:
        obj = json.loads(raw)
        def norm_list(lst):
            out = []
            for it in lst or []:
                if isinstance(it, dict) and it.get("source") and it.get("label"):
                    out.append({"source": normalize_source(it["source"]), "label": str(it["label"]).strip()})
            return out
        return {"figures": norm_list(obj.get("figures")), "tables": norm_list(obj.get("tables"))}
    except Exception:
        return {"figures": [], "tables": []}

def parse_sources_block(answer_text: str) -> List[str]:
    m = _SOURCES_BLOCK_RE.search(answer_text)
    if not m:
        return []
    try:
        obj = json.loads(m.group(1))
        srcs = [normalize_source(s) for s in obj.get("sources", []) if s]
        # keep order, dedup
        seen, ordered = set(), []
        for s in srcs:
            if s not in seen:
                seen.add(s); ordered.append(s)
        return ordered
    except Exception:
        return []

def strip_blocks(answer_text: str) -> str:
    text = _CHARTS_BLOCK_RE.sub("", answer_text)
    text = _SOURCES_BLOCK_RE.sub("", text)
    return text.rstrip()

# -------- 根据上下文构建图表 map ---------------

def _collect_charts(src_docs: List[Document]):
    table_map: Dict[Tuple[str, str], Dict[str, str]] = {}
    fig_map: Dict[Tuple[str, str], Dict[str, str]] = {}
    all_items = []

    for d in src_docs:
        info = d.metadata.get("__chart__")
        s = normalize_source(d.metadata.get("source"))
        if not info or not s:
            continue
        raw_no = info.get("raw_no", "")
        item = {
            "raw_no": raw_no,
            "caption": clean_caption((info.get("caption") or "").strip()),
            "url": (info.get("url") or "").strip(),
            "source": s,
        }
        all_items.append({"kind": info.get("kind"), **item})
        if info.get("kind") == "table":
            for lab in _normalize_labels(raw_no, "table"):
                table_map[(s, lab)] = item
        elif info.get("kind") == "figure":
            for lab in _normalize_labels(raw_no, "figure"):
                fig_map[(s, lab)] = item
    return table_map, fig_map, all_items

def _collect_sources_in_order(src_docs: List[Document]) -> List[str]:
    order, seen = [], set()
    for d in src_docs:
        s = normalize_source(d.metadata.get("source"))
        if s and s not in seen:
            seen.add(s); order.append(s)
    return order

# ----------- 用于渲染 Markdown 的辅助工具 -------------

def _md_escape(s: str) -> str:
    # 简单转义，避免 caption 里的 []() 影响 Markdown
    return s.replace("[", r"\[").replace("]", r"\]").replace("(", r"\(").replace(")", r"\)")

def _mk_table_item(i: int, cap: str, url: str) -> str:
    cap = _md_escape(cap or "(no caption)")
    link = f"[🔗]({url})" if url else ""
    return f"**Table{i}.** {cap} {link}".strip()

def _mk_figure_item(i: int, cap: str, url: str) -> str:
    cap = _md_escape(cap or "(no caption)")
    link = f"[🔗]({url})" if url else ""
    return f"**Fig.{i}.** {cap} {link}".strip()

def _mk_refs_md(used_sources: List[str]) -> str:
    if not used_sources:
        return ""
    lines = ["**Reference:**"]
    for i, s in enumerate(used_sources, 1):
        meta = metadata_index.get(s, {})
        title = (meta.get("title") or "(title unavailable)").strip()
        doi = (meta.get("doi") or "").strip()
        tail = f", DOI: {doi}" if doi else ""
        lines.append(f"[{i}] {title}{tail}")
    return "\n".join(lines)

# ----------- 主要输入（两种模式：Markdown 或纯文本打印）---------------------

def print_answer(query: str, debug: bool = False, render_markdown: bool = True):
    """
    render_markdown=True → 用 Jupyter Markdown 渲染（链接可点击）
    render_markdown=False → 文本打印
    """
    # 清空 retriever 内部的 logs，避免累积（如果收集用户信息可以暴露保留）
    chart_enhanced_retriever._logs.clear()
    
    result = chain.invoke({"query": query})
    raw_answer = result["result"]
    src_docs: List[Document] = result.get("source_documents", [])
    retr_logs = chart_enhanced_retriever.get_logs()

    # parse contracts
    cites = parse_chart_citations(raw_answer)
    used_sources = parse_sources_block(raw_answer)
    answer = strip_blocks(raw_answer)

    # === 严格过滤：只保留出现在本轮 context 里的来源；为空则按上下文顺序兜底 ===
    ctx_sources = [normalize_source(d.metadata.get("source")) for d in src_docs if d.metadata.get("source")]
    ctx_sources_set = set(ctx_sources)
    filtered_used_sources = [s for s in used_sources if s in ctx_sources_set]
    if not filtered_used_sources:
        filtered_used_sources = _collect_sources_in_order(src_docs)
        if debug:
            print("[DBG] used_sources filled by context order (LLM empty or out-of-context).")
    used_sources = filtered_used_sources

    # charts: strict match (no fallback)
    table_map, fig_map, all_items = _collect_charts(src_docs)
    selected_tables, selected_figs = [], []
    miss = {"tables": [], "figures": []}

    for obj in cites.get("tables", []):
        src = obj["source"]; lab = obj["label"]
        hit = None
        tried = _normalize_labels(lab, "table")
        for k in tried:
            hit = table_map.get((src, k))
            if hit: break
        if hit and hit not in selected_tables:
            selected_tables.append(hit)
        else:
            miss["tables"].append({"source": src, "label": lab, "tried": tried})

    for obj in cites.get("figures", []):
        src = obj["source"]; lab = obj["label"]
        hit = None
        tried = _normalize_labels(lab, "figure")
        for k in tried:
            hit = fig_map.get((src, k))
            if hit: break
        if hit and hit not in selected_figs:
            selected_figs.append(hit)
        else:
            miss["figures"].append({"source": src, "label": lab, "tried": tried})

    # ----- Render -----
    if render_markdown and _HAS_IPY:
        # 1) 正文
        display(Markdown(answer))

        # 2) 图表（每条一行，可点击 🔗）
        if selected_tables:
            md_lines = [ _mk_table_item(i+1, t["caption"], t["url"]) for i, t in enumerate(selected_tables) ]
            display(Markdown("\n\n".join(md_lines)))
        if selected_figs:
            md_lines = [ _mk_figure_item(i+1, f["caption"], f["url"]) for i, f in enumerate(selected_figs) ]
            display(Markdown("\n\n".join(md_lines)))

        # 3) 参考文献（只按过滤后的 used_sources；DOI 前有逗号）
        refs_md = _mk_refs_md(used_sources)
        if refs_md:
            display(Markdown(refs_md))
    else:
        # 纯文本打印（不渲染 Markdown）
        print(answer)
        for i, t in enumerate(selected_tables, 1):
            print(f"Table{i}: {t['caption']}，url：{t['url']}")
        for i, f in enumerate(selected_figs, 1):
            print(f"Fig.{i}: {f['caption']}，url：{f['url']}")
        if used_sources:
            print("Reference:")
            for i, s in enumerate(used_sources, 1):
                meta = metadata_index.get(s, {})
                title = (meta.get("title") or "(title unavailable)").strip()
                doi = (meta.get("doi") or "").strip()
                tail = f", DOI: {doi}" if doi else ""
                print(f"[{i}] {title}{tail}")

    # ----- Debug report 打印 -----
    if debug:
        print("\n[DBG] ------- DEBUG REPORT -------")
        for lg in retr_logs:
            stage = lg.get("stage", "?")
            payload = {k: v for k, v in lg.items() if k != "stage"}
            print(f"[DBG][{stage}] {json.dumps(payload, ensure_ascii=False)}")

        if all_items:
            def _dbg_sort_key(it):
                return (it['kind'], _num_key(it['raw_no']))
            all_items_sorted = sorted(all_items, key=_dbg_sort_key)
            print("[DBG] Available charts in context (sorted by number):")
            for it in all_items_sorted:
                print(f"  - ({it['kind']}) src={it['source']} raw={it['raw_no']} cap={_short(it['caption'])} url={it['url']}")

        print("[DBG] LLM <CHARTS> used (ordered):")
        print("  tables:", json.dumps(cites.get("tables"), ensure_ascii=False))
        print("  figures:", json.dumps(cites.get("figures"), ensure_ascii=False))

        if used_sources:
            print("[DBG] LLM <SOURCES> used (after filtering, ordered):", json.dumps(used_sources, ensure_ascii=False))
            ctx_sources_set_dbg = set(ctx_sources)
            missing_srcs = [s for s in used_sources if s not in ctx_sources_set_dbg]
            if missing_srcs:
                print("[WARN] Unexpected: filtered used_sources still contain out-of-context:", json.dumps(missing_srcs, ensure_ascii=False))
            else:
                print("[DBG] All cited sources are present in context.")
        else:
            print("[DBG] LLM <SOURCES> is empty after filtering (no sources cited).")

        if selected_tables or selected_figs:
            print("[DBG] Matched charts (as printed):")
            for i, t in enumerate(selected_tables, 1):
                print(f"  Table{i} <= (src={t['source']}, raw={t['raw_no']}) cap={_short(t['caption'])}")
            for i, f in enumerate(selected_figs, 1):
                print(f"  Fig.{i} <= (src={f['source']}, raw={f['raw_no']}) cap={_short(f['caption'])}")

        print("[DBG] ------- END DEBUG -------")

#----------------------------------------------------

from langgraph.prebuilt import create_react_agent
from langchain.tools import tool

@tool(return_direct=True) # 设定为 True 可直接将 tool的答案直接传给用户，不经过 llm 的二次转写
def print_answer_tool(query: str, debug: bool = False, render_markdown: bool = False) -> str:
    """
    📌 Purpose:
        Call rag_core.print_answer and directly return its output without rewriting.

    ⚙️ Usage:
        - Input query: Questions related to alloy oxidation
        - Optional parameter debug: Whether to print debug logs
        - Optional parameter render_markdown: Whether to render the output in Markdown

    ⚠️ Notes:
        - Only for questions about the oxidation mechanisms and background knowledge of high-temperature alloys.
        - If the question is outside this scope, the Agent should return "I don't know".
    """
    import io, contextlib

    buf = io.StringIO()
    result_str = None
    with contextlib.redirect_stdout(buf):
        result = print_answer(query, debug=debug, render_markdown=render_markdown)
        if isinstance(result, str) and result.strip():
            result_str = result.strip()

    if result_str:
        return result_str
    return buf.getvalue().strip()


# ------------------ 提示词设定 ------------------

RAG_CORE_AGENT_PROMPT = """
You are a RAG_agent specialized in the knowledge of superalloys and their oxidation behavior.

- ALWAYS solve the user request by calling the tool `print_answer_tool`.
- Do not rewrite or summarize the tool output. The tool result is the final answer.
- Handle the questions related to superalloys (including nickel-based, cobalt-based, etc.) and their oxidation behavior.
- If the question is outside your scope, reply : "I don't know, other agents may help this question".
""".strip()

# 定义智能体和工具

def build_rag_agent(create_react_agent, llm_RAG):

    tools=[print_answer_tool]
    agent = create_react_agent(
        model=llm_RAG,
        tools=tools,
        prompt=RAG_CORE_AGENT_PROMPT,
        name="RAG_agent",
    )
    return agent

from langgraph.prebuilt import create_react_agent
RAG_agent = build_rag_agent(create_react_agent, llm_RAG)

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

def rag_alloy_chat(query: str, debug: bool = True) -> Optional[str]:
    """
    向 RAG_agent 发送一次性问题并按需打印：
      - debug=True : 完整打印全部 streaming 过程
      - debug=False: 仅返回最后一个 AI 答案
    """
    final_ai_message: Optional[BaseMessage] = None

    if debug:
        # 完整过程打印
        for chunk in RAG_agent.stream(
            {"messages": [{"role": "user", "content": query}]}
        ):
            pretty_print_messages(chunk, last_message=False)
            last_ai = _extract_last_ai_message(chunk)
            if last_ai is not None:
                final_ai_message = last_ai
    else:
        # 静默收集，直到结束只返回最终答案
        for chunk in RAG_agent.stream(
            {"messages": [{"role": "user", "content": query}]}
        ):
            last_ai = _extract_last_ai_message(chunk)
            if last_ai is not None:
                final_ai_message = last_ai

    return getattr(final_ai_message, "content", None)
