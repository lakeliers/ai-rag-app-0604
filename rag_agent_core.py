import base64
import html
import mimetypes
import os
import re
import shlex
import math
import time
import hashlib
from uuid import uuid4
from collections import Counter, defaultdict
from html.parser import HTMLParser
from urllib.parse import quote_plus, unquote, urlparse, parse_qs

import chromadb
import requests
from openai import OpenAI
from sentence_transformers import CrossEncoder, SentenceTransformer
from parsing_layer import ParsedSection
from chunking_layer import ChunkCandidate, chunk_section, format_section_text, split_table_section, split_text_fixed, split_text_recursive


EMBEDDING_MODEL_NAME = "shibing624/text2vec-base-chinese"
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
COLLECTION_NAME = "file_docs"
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
QWEN_VL_MODEL = os.getenv("QWEN_VL_MODEL", "qwen-vl-plus")
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "45"))
ANSWER_MAX_TOKENS = int(os.getenv("ANSWER_MAX_TOKENS", "1400"))
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-base")
ENABLE_RERANKER = os.getenv("ENABLE_RERANKER", "1") == "1"
RERANK_LIMIT = int(os.getenv("RERANK_LIMIT", "12"))
MAX_CHUNKS_PER_SOURCE = int(os.getenv("MAX_CHUNKS_PER_SOURCE", "2"))
CONTEXT_CHAR_BUDGET = int(os.getenv("CONTEXT_CHAR_BUDGET", "4500"))
ENABLE_SUMMARY_CHUNKS = os.getenv("ENABLE_SUMMARY_CHUNKS", "0") == "1"
SUMMARY_MIN_CHARS = int(os.getenv("SUMMARY_MIN_CHARS", "1200"))
PARENT_TEXT_CHAR_LIMIT = int(os.getenv("PARENT_TEXT_CHAR_LIMIT", "1800"))
INGEST_FAILED_SEARCH_RESULTS = os.getenv("INGEST_FAILED_SEARCH_RESULTS", "0") == "1"
WEB_SEARCH_CANDIDATE_MULTIPLIER = int(os.getenv("WEB_SEARCH_CANDIDATE_MULTIPLIER", "3"))
WEB_MIN_TEXT_CHARS = int(os.getenv("WEB_MIN_TEXT_CHARS", "100"))
WEB_SEARCH_TIMEOUT_SECONDS = float(os.getenv("WEB_SEARCH_TIMEOUT_SECONDS", "8"))
WEB_FETCH_TIMEOUT_SECONDS = float(os.getenv("WEB_FETCH_TIMEOUT_SECONDS", "8"))
WEB_COLLECT_MAX_SECONDS = float(os.getenv("WEB_COLLECT_MAX_SECONDS", "35"))
ENABLE_JINA_READER = os.getenv("ENABLE_JINA_READER", "1") == "1"
JINA_READER_TIMEOUT_SECONDS = float(os.getenv("JINA_READER_TIMEOUT_SECONDS", "5"))
RRF_K = 60
RETRIEVAL_VECTOR_ONLY = "vector_only"
RETRIEVAL_VECTOR_BM25 = "vector_bm25"
RETRIEVAL_VECTOR_BM25_RRF = "vector_bm25_rrf"
RETRIEVAL_STRATEGIES = {
    RETRIEVAL_VECTOR_ONLY,
    RETRIEVAL_VECTOR_BM25,
    RETRIEVAL_VECTOR_BM25_RRF,
}
CONTEXT_SIMPLE_TOPK = "simple_topk"
CONTEXT_SOURCE_PRIORITY = "source_priority"
CONTEXT_WEIGHTED = "weighted"
CONTEXT_STRICT_BUDGET = "strict_budget"
CONTEXT_PACKING_STRATEGIES = {
    CONTEXT_SIMPLE_TOPK,
    CONTEXT_SOURCE_PRIORITY,
    CONTEXT_WEIGHTED,
    CONTEXT_STRICT_BUDGET,
}
CHUNKING_PLAIN = "plain"
CHUNKING_PARENT_CHILD = "parent_child"
CHUNKING_TABLE = "table"
CHUNKING_SUMMARY = "summary"
CHUNKING_AUTO = "auto"
CHUNKING_STRATEGIES = {
    CHUNKING_PLAIN,
    CHUNKING_PARENT_CHILD,
    CHUNKING_TABLE,
    CHUNKING_SUMMARY,
    CHUNKING_AUTO,
}
SOURCE_WEIGHTS = {
    "upload": 1.35,
    "web": 1.0,
    "local": 0.75,
    "unknown": 0.65,
}
SOURCE_QUOTAS = {
    "upload": 3,
    "web": 2,
    "local": 1,
    "unknown": 1,
}
FAILED_WEB_CONTENT_TYPES = {"search_result", "search_result_failed"}
LOW_CONFIDENCE_WEB_CONTENT_TYPES = {"search_result_summary"}
FAILED_WEB_MARKERS = [
    "网页搜索结果摘要",
    "完整网页正文读取失败",
    "失败原因",
    "读取失败",
    "网页正文太少",
    "403 Client Error",
    "403 Forbidden",
    "forbidden",
    "access denied",
]
WEB_BLOCK_MARKERS = [
    "百度安全验证",
    "网络不给力",
    "请输入验证码",
    "访问受限",
    "访问异常",
    "安全验证",
    "403 forbidden",
    "access denied",
    "forbidden",
    "_waf_",
    "captcha",
    "verify you are human",
]
LOW_VALUE_SEARCH_DOMAINS = {
    "mail.yahoo.com",
    "login.yahoo.com",
    "www.instagram.com",
    "instagram.com",
    "www.facebook.com",
    "facebook.com",
    "twitter.com",
    "x.com",
}
LOW_READABILITY_DOMAIN_PARTS = [
    "weibo.com",
    "quanmin.baidu.com",
    "haokan.baidu.com",
    "douyin.com",
    "kuaishou.com",
    "bilibili.com",
]
OFFICIAL_FINANCE_DOMAIN_PARTS = [
    "ir.lixiang.com",
    "investor.lixiang.com",
    "sec.gov",
]
FINANCE_QUERY_WORDS = [
    "财报",
    "业绩",
    "营收",
    "利润",
    "季度",
    "一季报",
    "年报",
    "earnings",
    "financial",
    "results",
    "quarter",
    "q1",
    "q2",
    "q3",
    "q4",
]
DESKTOP_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36"
)
MOBILE_USER_AGENT = (
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1"
)
INTENT_WEIGHTS = {
    "latest_news": {
        "rerank": 0.45,
        "rrf": 0.10,
        "source": 0.15,
        "freshness": 0.30,
    },
    "definition": {
        "rerank": 0.75,
        "rrf": 0.15,
        "source": 0.10,
        "freshness": 0.00,
    },
    "policy_or_prd": {
        "rerank": 0.55,
        "rrf": 0.10,
        "source": 0.25,
        "freshness": 0.10,
    },
    "general": {
        "rerank": 0.65,
        "rrf": 0.15,
        "source": 0.10,
        "freshness": 0.10,
    },
}


LOCAL_FILES_ONLY = os.getenv("LOCAL_FILES_ONLY", "0") == "1"
_embedding_model = None
_collection_cache = {}

conversation_history = []
last_sources = []
reranker_model = None
reranker_load_failed = False


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        print("正在加载本地向量模型...")
        _embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME,
            local_files_only=LOCAL_FILES_ONLY,
        )
    return _embedding_model


def get_collection(chroma_path=CHROMA_PATH, collection_name=COLLECTION_NAME):
    key = (chroma_path, collection_name)
    if key not in _collection_cache:
        client = chromadb.PersistentClient(path=chroma_path)
        _collection_cache[key] = client.get_or_create_collection(name=collection_name)
    return _collection_cache[key]


def content_hash(text, length=12):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def make_document_key(source_type, source, url="", document_hash=""):
    raw = "|".join([source_type or "", source or "", url or "", document_hash or ""])
    return content_hash(raw, length=16)


def normalize_metadata_scope(metadata_scope=None):
    return {
        key: value
        for key, value in (metadata_scope or {}).items()
        if value not in (None, "")
    }


def metadata_scope_filter(metadata_scope=None):
    clean_scope = normalize_metadata_scope(metadata_scope)
    if len(clean_scope) <= 1:
        return clean_scope
    return {"$and": [{key: value} for key, value in clean_scope.items()]}


def combine_metadata_filters(*filters):
    valid_filters = []
    for item in filters:
        if not item:
            continue
        if isinstance(item, dict) and set(item.keys()) == {"$and"}:
            valid_filters.extend(item["$and"])
        else:
            valid_filters.append(item)
    if not valid_filters:
        return None
    if len(valid_filters) == 1:
        return valid_filters[0]
    return {"$and": valid_filters}


class SimpleTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.skip = False
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript"}:
            self.skip = True

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript"}:
            self.skip = False

    def handle_data(self, data):
        if not self.skip:
            text = data.strip()
            if text:
                self.parts.append(text)

    def get_text(self):
        text = "\n".join(self.parts)
        text = html.unescape(text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text.strip()


def get_deepseek_client():
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        return None
    return OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        timeout=LLM_TIMEOUT_SECONDS,
        max_retries=0,
    )


def get_qwen_client():
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")
    if not api_key:
        return None
    return OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        timeout=LLM_TIMEOUT_SECONDS,
        max_retries=0,
    )


def split_text(text, chunk_size=500, chunk_overlap=80):
    return split_text_fixed(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def split_section(section, chunk_size=500, chunk_overlap=80):
    return [
        chunk.text
        for chunk in chunk_section(section, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    ]


def embed_texts(texts):
    return get_embedding_model().encode(texts).tolist()


def summarize_section_for_retrieval(section, force=False):
    if not ENABLE_SUMMARY_CHUNKS and not force:
        return ""

    text = section.text.strip()
    if len(text) < SUMMARY_MIN_CHARS:
        return ""

    client = get_deepseek_client()
    if client is None:
        return ""

    prompt = f"""请把下面资料压缩成适合知识库检索的摘要。
要求：
1. 保留关键实体、数字、时间、规则、限制条件
2. 不要加入原文没有的信息
3. 控制在 180 字以内
4. 用陈述句输出

资料：
{text[:4000]}
"""

    try:
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=220,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"摘要 chunk 生成失败，跳过：{e}")
        return ""


def analyze_query(question):
    lowered = question.lower()
    latest_words = ["最近", "最新", "今天", "新闻", "动态", "趋势", "刚刚", "现在"]
    freshness_suppression_words = [
        "不需要最新",
        "不要最新",
        "不用最新",
        "无需最新",
        "不需要新闻",
        "不要新闻",
        "不用联网",
        "不要联网",
        "不需要联网",
        "只解释定义",
        "只要定义",
    ]
    definition_words = ["是什么", "定义", "概念", "解释一下", "介绍一下", "什么意思"]
    policy_words = ["prd", "需求", "规则", "政策", "制度", "上线时间", "价格", "方案", "口径"]

    if any(word in question for word in latest_words) and not any(word in lowered for word in freshness_suppression_words):
        intent = "latest_news"
    elif any(word in lowered for word in policy_words):
        intent = "policy_or_prd"
    elif any(word in question for word in definition_words):
        intent = "definition"
    else:
        intent = "general"

    return {
        "intent": intent,
        "weights": INTENT_WEIGHTS[intent],
        "time_sensitive": intent == "latest_news" or any(word in question for word in ["最近", "最新", "今天"]),
        "needs_exact_terms": bool(re.search(r"[A-Za-z0-9_+\-.]{3,}", question)),
    }


def get_ranking_weights(query_profile):
    return query_profile.get("weights", INTENT_WEIGHTS["general"])


def get_reranker_model():
    global reranker_model, reranker_load_failed

    if not ENABLE_RERANKER:
        return None

    if reranker_model is not None:
        return reranker_model

    if reranker_load_failed:
        return None

    try:
        print("正在加载 Reranker 精排模型...")
        reranker_model = CrossEncoder(
            RERANKER_MODEL_NAME,
            local_files_only=LOCAL_FILES_ONLY,
        )
        return reranker_model
    except Exception as e:
        reranker_load_failed = True
        print(f"Reranker 加载失败，继续使用 RRF 排序：{e}")
        return None


def extract_query_keywords(question):
    words = re.findall(r"[A-Za-z0-9_+\-.]{2,}|[\u4e00-\u9fff]{2,}", question)
    stop_words = {
        "什么",
        "是什么",
        "最近",
        "最新",
        "今天",
        "新闻",
        "一下",
        "介绍",
        "回答",
    }
    return [word.lower() for word in words if word not in stop_words]


def keyword_score(document, question_keywords):
    lowered_document = document.lower()
    score = 0

    for keyword in question_keywords:
        if keyword in lowered_document:
            score += 1

    return score


def tokenize_text(text):
    lowered = text.lower()
    tokens = re.findall(r"[a-z0-9_+\-.]{2,}", lowered)
    chinese_spans = re.findall(r"[\u4e00-\u9fff]+", lowered)

    for span in chinese_spans:
        if len(span) == 1:
            tokens.append(span)
        else:
            tokens.extend(span[index:index + 2] for index in range(len(span) - 1))

    return tokens


def bm25_retrieve(question, limit=20, metadata_filter=None, chroma_path=CHROMA_PATH):
    active_collection = get_collection(chroma_path=chroma_path)
    if metadata_filter:
        data = active_collection.get(include=["documents", "metadatas"], where=metadata_filter)
    else:
        data = active_collection.get(include=["documents", "metadatas"])
    ids = data.get("ids", [])
    documents = data.get("documents", [])
    metadatas = data.get("metadatas", [])

    if not documents:
        return []

    query_tokens = tokenize_text(question)
    if not query_tokens:
        return []

    corpus_tokens = [tokenize_text(document) for document in documents]
    doc_count = len(corpus_tokens)
    avg_doc_len = sum(len(tokens) for tokens in corpus_tokens) / max(doc_count, 1)
    doc_freq = Counter()

    for tokens in corpus_tokens:
        doc_freq.update(set(tokens))

    k1 = 1.5
    b = 0.75
    rows = []

    for item_id, document, metadata, tokens in zip(ids, documents, metadatas, corpus_tokens):
        if is_failed_web_result(make_search_item(
            item_id=item_id,
            document=document,
            metadata=metadata,
        )):
            continue

        token_counts = Counter(tokens)
        doc_len = len(tokens)
        score = 0.0

        for token in query_tokens:
            tf = token_counts.get(token, 0)
            if tf == 0:
                continue

            idf = math.log((doc_count - doc_freq[token] + 0.5) / (doc_freq[token] + 0.5) + 1)
            denominator = tf + k1 * (1 - b + b * doc_len / max(avg_doc_len, 1))
            score += idf * (tf * (k1 + 1)) / denominator

        if score <= 0:
            continue

        rows.append(make_search_item(
            item_id=item_id,
            document=document,
            metadata=metadata,
            bm25_score=score,
        ))

    rows.sort(key=lambda item: item["bm25_score"], reverse=True)
    return rows[:limit]


def get_rows_by_sources(sources, limit=20, metadata_filter=None, chroma_path=CHROMA_PATH):
    source_list = list(sources or [])
    if not source_list:
        return []

    source_filter = {"source": {"$in": source_list}}
    combined_filter = combine_metadata_filters(metadata_filter, source_filter)
    active_collection = get_collection(chroma_path=chroma_path)
    data = active_collection.get(
        include=["documents", "metadatas"],
        where=combined_filter,
        limit=limit,
    )

    rows = []
    for item_id, document, metadata in zip(
        data.get("ids", []),
        data.get("documents", []),
        data.get("metadatas", []),
    ):
        rows.append(make_search_item(
            item_id=item_id,
            document=document,
            metadata=metadata,
        ))

    return rows


def make_search_item(item_id, document, metadata, distance=None, bm25_score=0.0):
    source = metadata.get("source", "未知来源")
    return {
        "id": item_id,
        "document": document,
        "source": source,
        "source_type": metadata.get("source_type", "unknown"),
        "url": metadata.get("url", ""),
        "chunk_index": metadata.get("chunk_index", 0),
        "created_at": metadata.get("created_at", 0),
        "content_type": metadata.get("content_type", "text"),
        "document_key": metadata.get("document_key", source),
        "section_title": metadata.get("section_title", ""),
        "page": metadata.get("page", 0),
        "sheet": metadata.get("sheet", ""),
        "row_start": metadata.get("row_start", 0),
        "row_end": metadata.get("row_end", 0),
        "chunk_type": metadata.get("chunk_type", "child"),
        "parent_id": metadata.get("parent_id", ""),
        "parent_text": metadata.get("parent_text", ""),
        "distance": distance,
        "bm25_score": bm25_score,
    }


def is_failed_web_result(row):
    if row.get("source_type") != "web":
        return False

    document = row.get("document", "") or ""
    content_type = row.get("content_type", "")
    source = row.get("source", "") or ""
    lowered_document = document.lower()

    if content_type in FAILED_WEB_CONTENT_TYPES:
        return True
    if source.startswith("网页搜索结果："):
        return True

    failure_prefix = lowered_document[:260]
    for marker in FAILED_WEB_MARKERS:
        marker_lower = marker.lower()
        if failure_prefix.startswith(marker_lower):
            return True
        if marker_lower in failure_prefix and any(
            prefix in failure_prefix for prefix in ["说明：", "失败原因", "完整网页正文读取失败"]
        ):
            return True

    return False


def is_low_confidence_web_result(row):
    return (
        row.get("source_type") == "web"
        and row.get("content_type", "") in LOW_CONFIDENCE_WEB_CONTENT_TYPES
    )


def filter_answerable_rows(rows):
    answerable_rows = []
    low_confidence_rows = []

    for row in rows:
        if is_failed_web_result(row):
            row["context_skip_reason"] = "网页正文不可读，已排除"
            continue
        if is_low_confidence_web_result(row):
            row["context_skip_reason"] = "已有正文资料时，标题线索不进入最终上下文"
            row["final_score"] = row.get("final_score", 0) * 0.35
            low_confidence_rows.append(row)
            continue
        answerable_rows.append(row)

    if len(answerable_rows) >= 2:
        return answerable_rows

    return answerable_rows + low_confidence_rows


def safe_id(text):
    text = re.sub(r"[^a-zA-Z0-9_\-.]+", "_", text)
    return text[:120]


def normalize_chunking_strategies(chunking_strategy):
    if chunking_strategy is None:
        return {CHUNKING_PARENT_CHILD, CHUNKING_TABLE}

    if isinstance(chunking_strategy, str):
        strategies = {
            item.strip()
            for item in chunking_strategy.split(",")
            if item.strip()
        }
    else:
        strategies = set(chunking_strategy)

    strategies = {strategy for strategy in strategies if strategy in CHUNKING_STRATEGIES}

    if not strategies or CHUNKING_AUTO in strategies:
        strategies.discard(CHUNKING_AUTO)
        strategies.update({CHUNKING_PARENT_CHILD, CHUNKING_TABLE})

    return strategies


def choose_chunking_strategy(section, enabled_strategies):
    if section.content_type == "table" and CHUNKING_TABLE in enabled_strategies:
        return CHUNKING_TABLE
    if CHUNKING_PARENT_CHILD in enabled_strategies:
        return CHUNKING_PARENT_CHILD
    if CHUNKING_PLAIN in enabled_strategies:
        return CHUNKING_PLAIN
    return CHUNKING_PLAIN


def build_chunk_candidates(section, source, section_index, chunking_strategy):
    enabled_strategies = normalize_chunking_strategies(chunking_strategy)
    selected_strategy = choose_chunking_strategy(section, enabled_strategies)

    parent_text = format_section_text(section)
    parent_id = f"{source}:{section_index}:{section.section_title or section.content_type or 'section'}"

    if selected_strategy == CHUNKING_PLAIN:
        return [
            ChunkCandidate(text=text, chunk_type="plain")
            for text in split_text_fixed(parent_text)
        ]

    if selected_strategy == CHUNKING_TABLE:
        if section.content_type == "table":
            texts = split_table_section(section)
        else:
            texts = split_text_recursive(parent_text)
        return [
            ChunkCandidate(
                text=text,
                chunk_type="table" if section.content_type == "table" else "child",
                parent_id=parent_id,
                parent_text=parent_text,
            )
            for text in texts
        ]

    return chunk_section(section, source=source, section_index=section_index)


def add_text_to_chroma(
    text,
    source,
    source_type="local",
    url="",
    content_type="text",
    created_at=None,
    chunking_strategy=CHUNKING_PARENT_CHILD,
    chroma_path=CHROMA_PATH,
    metadata_scope=None,
):
    section = ParsedSection(text=text, content_type=content_type)
    return add_sections_to_chroma(
        [section],
        source=source,
        source_type=source_type,
        url=url,
        created_at=created_at,
        chunking_strategy=chunking_strategy,
        chroma_path=chroma_path,
        metadata_scope=metadata_scope,
    )


def add_sections_to_chroma(
    sections,
    source,
    source_type="local",
    url="",
    created_at=None,
    chunking_strategy=CHUNKING_PARENT_CHILD,
    chroma_path=CHROMA_PATH,
    metadata_scope=None,
):
    chunk_rows = []
    enabled_strategies = normalize_chunking_strategies(chunking_strategy)

    for section_index, section in enumerate(sections):
        chunk_candidates = build_chunk_candidates(section, source, section_index, enabled_strategies)
        summary = summarize_section_for_retrieval(section, force=True) if CHUNKING_SUMMARY in enabled_strategies else ""
        if summary:
            parent_id = chunk_candidates[0].parent_id if chunk_candidates else f"{source}:{section_index}:summary"
            parent_text = chunk_candidates[0].parent_text if chunk_candidates else section.text.strip()
            chunk_candidates.insert(0, ChunkCandidate(
                text=f"摘要：{summary}",
                chunk_type="summary",
                parent_id=parent_id,
                parent_text=parent_text,
            ))

        for chunk_index, chunk in enumerate(chunk_candidates):
            chunk_rows.append((section_index, chunk_index, section, chunk))

    if not chunk_rows:
        return 0

    now = int(created_at or time.time())
    batch_id = uuid4().hex[:12]
    ids = []
    metadatas = []
    chunks = []
    scope = normalize_metadata_scope(metadata_scope)
    source_hash = content_hash(f"{source_type}|{source}|{url or source}", length=12)
    document_hash = content_hash("\n".join(chunk.text for _, _, _, chunk in chunk_rows), length=12)
    document_key = f"{source_type}:{safe_id(source)}:{source_hash}:{document_hash}"

    for index, (section_index, chunk_index, section, chunk) in enumerate(chunk_rows):
        chunk_hash = content_hash(chunk.text, length=12)
        namespace = scope.get("eval_run_id") or scope.get("session_id") or scope.get("user_id") or "global"
        item_id = ":".join([
            safe_id(str(namespace)),
            safe_id(document_key),
            batch_id,
            str(section_index),
            str(chunk_index),
            chunk_hash,
        ])
        ids.append(item_id)
        chunks.append(chunk.text)
        metadata = {
            "source": source,
            "source_type": source_type,
            "url": url,
            "chunk_index": index,
            "created_at": now,
            "content_type": section.content_type,
            "document_key": document_key,
            "section_title": section.section_title,
            "section_index": section_index,
            "page": section.page or 0,
            "sheet": section.sheet,
            "row_start": section.row_start or 0,
            "row_end": section.row_end or 0,
            "chunk_type": chunk.chunk_type,
            "parent_id": chunk.parent_id,
            "parent_text": chunk.parent_text[:PARENT_TEXT_CHAR_LIMIT],
            "batch_id": batch_id,
            "chunk_hash": chunk_hash,
        }
        metadata.update(scope)
        metadatas.append(metadata)

    embeddings = embed_texts(chunks)

    get_collection(chroma_path=chroma_path).upsert(
        ids=ids,
        documents=chunks,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    return len(chunks)


def seed_local_note(file_path="my_note.md"):
    if get_collection().count() > 0:
        return 0

    if not os.path.exists(file_path):
        return 0

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    return add_text_to_chroma(
        text,
        source=os.path.basename(file_path),
        source_type="local",
        url=file_path,
    )


def source_priority_score(source, preferred_sources):
    if not preferred_sources:
        return 0

    return 1 if source in preferred_sources else 0


def is_allowed_source(row, preferred_sources, preferred_only=False):
    source_type = row.get("source_type", "unknown")
    if preferred_only:
        return source_type == "upload" and row.get("source") in preferred_sources

    if source_type != "upload":
        return True

    if not preferred_sources:
        return False

    return row.get("source") in preferred_sources


def source_label(item):
    source_type = item.get("source_type", "unknown")
    source = item.get("source", "")

    if source_type == "upload" and source.startswith("图片："):
        return "上传图片｜优先"
    if source_type == "upload":
        return "上传资料｜优先"
    if source_type == "web":
        return "网络资料｜补充"
    if source_type == "local":
        return "基础资料｜兜底"
    return "其他资料｜参考"


def build_metadata_filter(query_profile):
    if query_profile.get("intent") != "latest_news":
        return None

    recent_window_seconds = 90 * 24 * 60 * 60
    return {
        "created_at": {
            "$gte": int(time.time()) - recent_window_seconds,
        }
    }


def vector_retrieve(question, limit=20, metadata_filter=None, chroma_path=CHROMA_PATH):
    query_embedding = embed_texts([question])
    active_collection = get_collection(chroma_path=chroma_path)
    if metadata_filter:
        results = active_collection.query(
            query_embeddings=query_embedding,
            n_results=limit,
            where=metadata_filter,
        )
    else:
        results = active_collection.query(
            query_embeddings=query_embedding,
            n_results=limit,
        )

    rows = []
    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for item_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
        rows.append(make_search_item(
            item_id=item_id,
            document=document,
            metadata=metadata,
            distance=distance,
        ))

    return rows


def source_weight(source_type):
    return SOURCE_WEIGHTS.get(source_type, SOURCE_WEIGHTS["unknown"])


def freshness_score(row):
    created_at = row.get("created_at") or 0
    if not created_at:
        return 0.5

    age_days = max((int(time.time()) - int(created_at)) / 86400, 0)
    if age_days <= 1:
        return 1.0
    if age_days <= 7:
        return 0.85
    if age_days <= 30:
        return 0.65
    if age_days <= 90:
        return 0.4
    return 0.2


def normalize_rrf_score(score):
    return min(score * 30, 1.0)


def vector_similarity_score(row):
    distance = row.get("distance")
    if distance is None:
        return 0.0
    return 1 / (1 + max(float(distance), 0.0))


def normalize_bm25_score(score, max_score):
    if max_score <= 0:
        return 0.0
    return min(float(score) / max_score, 1.0)


def answerability_score(row, question_keywords, query_profile):
    document = row.get("document", "")
    lowered_document = document.lower()
    intent = query_profile.get("intent", "general")

    keyword_coverage = 0.0
    if question_keywords:
        hits = sum(1 for keyword in question_keywords if keyword in lowered_document)
        keyword_coverage = hits / len(question_keywords)

    info_markers = [
        "包括",
        "分别是",
        "主要",
        "原因",
        "目标",
        "明确",
        "显示",
        "为",
        "是",
        "定义",
        "趋势",
        "多智能体",
        "自动化",
        "工作流",
        "检索增强生成",
    ]
    noise_markers = [
        "导航",
        "入口",
        "榜单",
        "聚合",
        "热门产品列表",
        "工具榜单",
        "教程入口",
    ]

    marker_score = min(sum(1 for marker in info_markers if marker in document) / 4, 1.0)
    length_score = min(len(document) / 120, 1.0)
    noise_penalty = min(sum(1 for marker in noise_markers if marker in document) * 0.25, 0.6)

    score = keyword_coverage * 0.45 + marker_score * 0.35 + length_score * 0.20

    if intent == "definition" and ("是什么" in document or "定义" in document or "检索增强生成" in document):
        score += 0.25
    if intent == "latest_news" and ("趋势" in document or "新趋势" in document or "2026" in document):
        score += 0.20

    return max(min(score - noise_penalty, 1.0), 0.0)


def base_retrieval_score(row, source_priority, keyword_hits, query_profile):
    weights = get_ranking_weights(query_profile)
    source_component = source_weight(row.get("source_type", "unknown")) / max(SOURCE_WEIGHTS.values())
    freshness_component = freshness_score(row)
    question_keywords = query_profile.get("keywords", [])
    answerability_component = answerability_score(row, question_keywords, query_profile)
    exact_term_bonus = keyword_hits * 0.005

    score = (
        normalize_rrf_score(row.get("rrf_score", 0)) * weights["rrf"]
        + answerability_component * 0.35
        + source_component * weights["source"]
        + freshness_component * weights["freshness"]
        + exact_term_bonus
    )
    row["answerability_score"] = answerability_component

    if source_priority:
        score *= 1.2

    return score


def apply_source_quotas(rows, top_k):
    selected = []
    quota_counts = defaultdict(int)

    for row in rows:
        source_type = row.get("source_type", "unknown")
        quota = SOURCE_QUOTAS.get(source_type, SOURCE_QUOTAS["unknown"])

        if quota_counts[source_type] >= quota:
            continue

        selected.append(row)
        quota_counts[source_type] += 1

        if len(selected) >= top_k:
            return selected

    if len(selected) < top_k:
        selected_ids = {row["id"] for row in selected}
        for row in rows:
            if row["id"] in selected_ids:
                continue
            selected.append(row)
            if len(selected) >= top_k:
                break

    return selected


def dedupe_and_limit_chunks(rows, max_chunks_per_source=MAX_CHUNKS_PER_SOURCE):
    selected = []
    seen_texts = set()
    source_counts = defaultdict(int)

    for row in rows:
        document_key = row.get("document_key") or row.get("source")
        if source_counts[document_key] >= max_chunks_per_source:
            row["context_skip_reason"] = "同文档块数已达上限"
            continue

        normalized_text = re.sub(r"\s+", "", row["document"].lower())[:260]
        if normalized_text in seen_texts:
            row["context_skip_reason"] = "内容重复"
            continue

        seen_texts.add(normalized_text)
        source_counts[document_key] += 1
        row["context_skip_reason"] = ""
        selected.append(row)

    return selected


def pack_context_results(rows, top_k, char_budget=CONTEXT_CHAR_BUDGET, context_packing_strategy=CONTEXT_STRICT_BUDGET):
    rows = filter_answerable_rows(rows)

    if context_packing_strategy not in CONTEXT_PACKING_STRATEGIES:
        context_packing_strategy = CONTEXT_STRICT_BUDGET

    if context_packing_strategy == CONTEXT_SIMPLE_TOPK:
        packed = rows[:top_k]
        for index, row in enumerate(packed, start=1):
            row["context_order"] = index
            row["context_skip_reason"] = ""
        return packed

    if context_packing_strategy == CONTEXT_SOURCE_PRIORITY:
        packed = apply_source_quotas(rows, top_k)
        for index, row in enumerate(packed, start=1):
            row["context_order"] = index
            row["context_skip_reason"] = ""
        return packed

    if context_packing_strategy == CONTEXT_WEIGHTED:
        packed = dedupe_and_limit_chunks(apply_source_quotas(rows, top_k * 2))[:top_k]
        for index, row in enumerate(packed, start=1):
            row["context_order"] = index
        return packed

    quota_rows = apply_source_quotas(rows, top_k * 2)
    deduped_rows = dedupe_and_limit_chunks(quota_rows)

    packed = []
    used_chars = 0

    for row in deduped_rows:
        content_len = len(row.get("document", ""))
        if packed and used_chars + content_len > char_budget:
            row["context_skip_reason"] = "超过上下文预算"
            continue

        row["context_order"] = len(packed) + 1
        packed.append(row)
        used_chars += content_len

        if len(packed) >= top_k:
            break

    if len(packed) < top_k:
        packed_ids = {row["id"] for row in packed}
        for row in rows:
            if row["id"] in packed_ids:
                continue
            row["context_order"] = len(packed) + 1
            packed.append(row)
            if len(packed) >= top_k:
                break

    return packed


def rerank_results(question, rows, query_profile, limit=None):
    if not rows:
        return []

    model = get_reranker_model()
    if model is None:
        for row in rows:
            row["rerank_status"] = "未启用"
            row["final_score"] = row.get("pre_rerank_score", row.get("final_score", 0))
        return rows

    rerank_limit = limit or RERANK_LIMIT
    candidate_rows = rows[:rerank_limit]
    remaining_rows = rows[rerank_limit:]

    pairs = [
        [question, row["document"]]
        for row in candidate_rows
    ]

    scores = model.predict(pairs)
    min_score = min(float(score) for score in scores)
    max_score = max(float(score) for score in scores)
    score_range = max_score - min_score
    weights = get_ranking_weights(query_profile)

    for row, score in zip(candidate_rows, scores):
        raw_score = float(score)
        if score_range == 0:
            normalized_score = 0.5
        else:
            normalized_score = (raw_score - min_score) / score_range

        row["pre_rerank_score"] = row.get("final_score", 0)
        row["rerank_score"] = raw_score
        row["rerank_norm"] = normalized_score
        row["rerank_status"] = "已启用"
        source_component = source_weight(row.get("source_type", "unknown")) / max(SOURCE_WEIGHTS.values())
        freshness_component = freshness_score(row)
        row["final_score"] = (
            normalized_score * weights["rerank"]
            + normalize_rrf_score(row.get("rrf_score", 0)) * weights["rrf"]
            + source_component * weights["source"]
            + freshness_component * weights["freshness"]
        )
        if row.get("source_priority"):
            row["final_score"] *= 1.2

    candidate_rows.sort(key=lambda item: item["final_score"], reverse=True)
    return candidate_rows + remaining_rows


def search_chroma(
    question,
    top_k=3,
    preferred_sources=None,
    preferred_only=False,
    retrieval_strategy=RETRIEVAL_VECTOR_BM25_RRF,
    context_packing_strategy=CONTEXT_STRICT_BUDGET,
    chroma_path=CHROMA_PATH,
    metadata_scope=None,
):
    if retrieval_strategy not in RETRIEVAL_STRATEGIES:
        retrieval_strategy = RETRIEVAL_VECTOR_BM25_RRF
    query_profile = analyze_query(question)
    scope_filter = metadata_scope_filter(metadata_scope)
    metadata_filter = combine_metadata_filters(
        build_metadata_filter(query_profile),
        scope_filter,
    )
    preferred_sources = set(preferred_sources or [])
    if preferred_only and preferred_sources:
        metadata_filter = combine_metadata_filters(
            scope_filter,
            {"source": {"$in": list(preferred_sources)}},
        )
    recall_limit = max(top_k * 8, 24)
    vector_rows = vector_retrieve(
        question,
        limit=recall_limit,
        metadata_filter=metadata_filter,
        chroma_path=chroma_path,
    )
    bm25_rows = []
    if retrieval_strategy != RETRIEVAL_VECTOR_ONLY:
        bm25_rows = bm25_retrieve(
            question,
            limit=recall_limit,
            metadata_filter=metadata_filter,
            chroma_path=chroma_path,
        )
    preferred_rows = []
    if preferred_sources and not preferred_only:
        preferred_rows = get_rows_by_sources(
            preferred_sources,
            limit=max(len(preferred_sources) * MAX_CHUNKS_PER_SOURCE, top_k),
            metadata_filter=scope_filter,
            chroma_path=chroma_path,
        )
    if metadata_filter and not preferred_only and not vector_rows and not bm25_rows:
        metadata_filter = scope_filter or None
        vector_rows = vector_retrieve(
            question,
            limit=recall_limit,
            metadata_filter=metadata_filter,
            chroma_path=chroma_path,
        )
        if retrieval_strategy != RETRIEVAL_VECTOR_ONLY:
            bm25_rows = bm25_retrieve(
                question,
                limit=recall_limit,
                metadata_filter=metadata_filter,
                chroma_path=chroma_path,
            )
    question_keywords = extract_query_keywords(question)
    query_profile["keywords"] = question_keywords

    fused = {}

    for rank, row in enumerate(vector_rows, start=1):
        if is_failed_web_result(row):
            continue
        if not is_allowed_source(row, preferred_sources, preferred_only=preferred_only):
            continue
        item_id = row["id"]
        fused.setdefault(item_id, row.copy())
        fused[item_id]["vector_rank"] = rank
        fused[item_id]["rrf_score"] = fused[item_id].get("rrf_score", 0) + 1 / (RRF_K + rank)
        fused[item_id]["vector_score"] = vector_similarity_score(row)

    for rank, row in enumerate(bm25_rows, start=1):
        if is_failed_web_result(row):
            continue
        if not is_allowed_source(row, preferred_sources, preferred_only=preferred_only):
            continue
        item_id = row["id"]
        fused.setdefault(item_id, row.copy())
        fused[item_id]["bm25_rank"] = rank
        fused[item_id]["bm25_score"] = row.get("bm25_score", 0)
        fused[item_id]["rrf_score"] = fused[item_id].get("rrf_score", 0) + 1 / (RRF_K + rank)

    for rank, row in enumerate(preferred_rows, start=1):
        if is_failed_web_result(row):
            continue
        item_id = row["id"]
        fused.setdefault(item_id, row.copy())
        fused[item_id]["preferred_rank"] = rank
        fused[item_id]["rrf_score"] = fused[item_id].get("rrf_score", 0) + 1 / (RRF_K + rank)

    search_results = []
    max_bm25_score = max((row.get("bm25_score", 0) for row in fused.values()), default=0)

    for row in fused.values():
        source_type = row.get("source_type", "unknown")
        source_priority = source_priority_score(row["source"], preferred_sources)
        keyword_hits = keyword_score(row["document"], question_keywords)
        if retrieval_strategy == RETRIEVAL_VECTOR_ONLY:
            final_score = row.get("vector_score", vector_similarity_score(row))
        elif retrieval_strategy == RETRIEVAL_VECTOR_BM25:
            vector_component = row.get("vector_score", vector_similarity_score(row))
            bm25_component = normalize_bm25_score(row.get("bm25_score", 0), max_bm25_score)
            final_score = vector_component * 0.65 + bm25_component * 0.35
            if source_priority:
                final_score *= 1.1
        else:
            final_score = base_retrieval_score(row, source_priority, keyword_hits, query_profile)
        row["keyword_score"] = keyword_hits
        row["source_priority"] = source_priority
        row["source_weight"] = source_weight(source_type)
        row["freshness_score"] = freshness_score(row)
        row["answerability_score"] = row.get("answerability_score", 0)
        row["query_intent"] = query_profile["intent"]
        row["ranking_weights"] = str(query_profile["weights"])
        row["retrieval_strategy"] = retrieval_strategy
        row["context_packing_strategy"] = context_packing_strategy
        row["final_score"] = final_score
        row["pre_rerank_score"] = final_score
        search_results.append(row)

    search_results = filter_answerable_rows(search_results)
    search_results.sort(key=lambda item: item["final_score"], reverse=True)
    if retrieval_strategy == RETRIEVAL_VECTOR_BM25_RRF:
        search_results = rerank_results(
            question,
            search_results,
            query_profile,
            limit=max(top_k * 4, RERANK_LIMIT),
        )
    else:
        for row in search_results:
            row["rerank_status"] = "未启用"

    return pack_context_results(
        search_results,
        top_k,
        context_packing_strategy=context_packing_strategy,
    )


def build_context(search_results):
    search_results = filter_answerable_rows(search_results)

    if not search_results:
        return "没有检索到资料。"

    parts = []
    for index, item in enumerate(search_results, start=1):
        source_line = item["source"]
        if item.get("url"):
            source_line += f" | {item['url']}"
        content = item["document"]
        if item.get("chunk_type") == "summary" and item.get("parent_text"):
            content = f"{item['document']}\n\n原文依据：\n{item['parent_text']}"

        part = f"""资料 {index}：
来源：{source_line}
类型：{item['source_type']}
优先级：{source_label(item)}
查询意图：{item.get('query_intent', 'general')}
融合分：{item.get('final_score', 0):.4f}
原始融合分：{item.get('pre_rerank_score', item.get('final_score', 0)):.4f}
时间新鲜度：{item.get('freshness_score', 0):.2f}
答案性分数：{item.get('answerability_score', 0):.2f}
Rerank状态：{item.get('rerank_status', '未启用')}
Rerank分：{item.get('rerank_score', '无')}
向量排名：{item.get('vector_rank', '未召回')}
关键词排名：{item.get('bm25_rank', '未召回')}
上下文顺序：{item.get('context_order', index)}
块编号：{item['chunk_index']}
块类型：{item.get('chunk_type', 'child')}
小节：{item.get('section_title', '')}
页码：{item.get('page', 0)}
工作表：{item.get('sheet', '')}
行号：{item.get('row_start', 0)}-{item.get('row_end', 0)}
内容：{content}
"""
        parts.append(part)

    return "\n---\n".join(parts)


def build_history_text(max_turns=6):
    recent = conversation_history[-max_turns:]
    if not recent:
        return "暂无历史对话。"

    lines = []
    for message in recent:
        role = "用户" if message["role"] == "user" else "助手"
        lines.append(f"{role}：{message['content']}")

    return "\n".join(lines)


def build_answer_prompt(question, search_results, include_history=True):
    context = build_context(search_results)
    history_text = build_history_text() if include_history else "本轮为独立请求，不使用历史对话作为事实依据。"

    return f"""你是一个可以使用知识库和网页资料的 RAG Agent。

请根据【资料】回答【用户问题】。
如果资料不足，请明确说资料不足，不要编造。
资料使用规则：
1. 上传资料和上传图片属于用户主动提供的信息，可信优先级最高。
2. 网络资料只作为补充；当网络资料和上传资料冲突时，优先采用上传资料。
3. 基础资料只作为兜底，不能压过用户上传资料和当前联网资料。
4. 如果资料来自网页，请提醒用户网页信息可能会变化。
5. 除非用户明确要求通用概念解释，否则所有事实、案例、产品名、公司名、数据和时间点都必须来自【资料】。
6. 不要使用你自己的通用知识补充【资料】之外的案例或结论；如果【资料】里没有足够依据，请直接说明资料不足，并说明还缺什么资料。

【最近对话】
{history_text}

【资料】
{context}

【用户问题】
{question}

【回答要求】
1. 先给结论
2. 再用 2-4 条解释关键依据
3. 如果用户要求案例、类似案例、对标产品或产品例子，只能列出【资料】中可以支持的具体案例或产品名称；如果资料中没有案例，必须说明“当前资料没有可验证的类似案例”
4. 如果用户要求方案、报告、计划、清单、分析或建议，必须输出结构化交付物，至少包含：目标/结论、关键分析、具体建议或下一步计划、参考来源
5. 最后列出参考来源
"""


def ask_deepseek(question, search_results, include_history=True, model_name=""):
    client = get_deepseek_client()
    if client is None:
        print("没有找到 DEEPSEEK_API_KEY。")
        print("请在终端设置：export DEEPSEEK_API_KEY=\"sk-xxx\"")
        return None

    prompt = build_answer_prompt(question, search_results, include_history=include_history)

    response = client.chat.completions.create(
        model=model_name or DEEPSEEK_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=ANSWER_MAX_TOKENS,
        timeout=LLM_TIMEOUT_SECONDS,
    )

    return response.choices[0].message.content


def ask_deepseek_stream(question, search_results, on_delta=None, include_history=True, model_name=""):
    client = get_deepseek_client()
    if client is None:
        print("没有找到 DEEPSEEK_API_KEY。")
        print("请在终端设置：export DEEPSEEK_API_KEY=\"sk-xxx\"")
        return None

    prompt = build_answer_prompt(question, search_results, include_history=include_history)
    chunks = []

    stream = client.chat.completions.create(
        model=model_name or DEEPSEEK_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=ANSWER_MAX_TOKENS,
        timeout=LLM_TIMEOUT_SECONDS,
        stream=True,
    )

    for chunk in stream:
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            continue
        delta = getattr(choices[0], "delta", None)
        text = getattr(delta, "content", None) if delta else None
        if not text:
            continue
        chunks.append(text)
        if on_delta:
            on_delta(text, "".join(chunks))

    return "".join(chunks)


def normalize_search_url(raw_url):
    parsed_url = html.unescape(raw_url)

    if parsed_url.startswith("//"):
        parsed_url = "https:" + parsed_url

    if "uddg=" in parsed_url:
        query_part = parse_qs(urlparse(parsed_url).query)
        parsed_url = unquote(query_part.get("uddg", [parsed_url])[0])

    return parsed_url


def clean_title(raw_title):
    title = re.sub(r"<.*?>", "", raw_title)
    return html.unescape(title).strip()


def is_finance_query(query):
    lowered = query.lower()
    return any(word in lowered for word in FINANCE_QUERY_WORDS)


def unique_preserve_order(items):
    seen = set()
    unique_items = []
    for item in items:
        normalized = re.sub(r"\s+", " ", item).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique_items.append(normalized)
    return unique_items


def expand_web_queries(query):
    compact_query = query.strip()
    queries = [compact_query]
    lowered = compact_query.lower()
    is_ai_agent_query = "agent" in lowered or "智能体" in compact_query
    is_time_sensitive_query = any(word in compact_query for word in ["今天", "最近", "最新", "动态", "新闻", "发布"])
    is_rag_query = "rag" in lowered or "检索增强" in compact_query or "向量检索" in compact_query
    is_agent_memory_query = "memory" in lowered or "记忆" in compact_query

    if is_finance_query(compact_query):
        spaced_query = re.sub(r"([A-Za-z0-9]+|20\d{2}|19\d{2})", r" \1 ", compact_query)
        spaced_query = re.sub(r"\s+", " ", spaced_query).strip()
        queries.append(spaced_query)

        if "理想" in compact_query or "li auto" in lowered or "lixiang" in lowered:
            queries.extend([
                "理想汽车 2026 第一季度 财报",
                "理想汽车 2026 Q1 财报",
                "理想汽车 2026 年第一季度 业绩",
                "Li Auto Q1 2026 financial results",
                "Li Auto first quarter 2026 earnings",
                "Li Auto investor relations Q1 2026",
            ])

    if is_agent_memory_query and is_time_sensitive_query:
        queries.extend([
            "Agent Memory 长期记忆 实践 趋势",
            "AI Agent memory architecture long term memory",
            "AI Agent 记忆系统 用户记忆 任务记忆 实践",
        ])
    elif is_ai_agent_query and is_time_sensitive_query and not is_rag_query:
        queries.extend([
            "AI Agent 产品动态 2026 6月",
            "AI Agent 产品发布 2026 最新",
            "AI Agent 新产品 融资 发布 2026",
            "AI Agent product updates 2026 June",
        ])

    return unique_preserve_order(queries)[:4]


def domain_contains(domain, parts):
    return any(part in domain for part in parts)


def search_query_terms(query):
    terms = set(re.findall(r"[a-zA-Z][a-zA-Z0-9]+|20\d{2}|19\d{2}", query.lower()))
    for token in re.findall(r"[\u4e00-\u9fff]{2,}", query):
        if len(token) <= 4:
            terms.add(token)
        else:
            for index in range(max(len(token) - 1, 0)):
                terms.add(token[index:index + 2])
    return {term for term in terms if term not in {"什么", "情况", "一下", "了解"}}


def score_search_result(query, item):
    title = item.get("title", "") or ""
    url = item.get("url", "") or ""
    domain = urlparse(url).netloc.lower()
    lowered_text = f"{title} {url}".lower()
    score = 0.0

    if domain in LOW_VALUE_SEARCH_DOMAINS:
        score -= 12
    if domain_contains(domain, LOW_READABILITY_DOMAIN_PARTS):
        score -= 4

    if is_finance_query(query):
        if domain_contains(domain, OFFICIAL_FINANCE_DOMAIN_PARTS):
            score += 10
        if any(word in lowered_text for word in ["财报", "业绩", "营收", "利润", "financial", "earnings", "results", "quarterly"]):
            score += 5
        if any(word in lowered_text for word in ["第一季度", "一季度", "q1", "first quarter"]):
            score += 4
        if "2026" in lowered_text:
            score += 3
        if any(word in lowered_text for word in ["理想汽车", "li auto", "lixiang"]):
            score += 6
        if any(word in lowered_text for word in ["cctv", "autohome", "雪球", "xueqiu", "36kr", "财联社"]):
            score += 2
        if any(word in lowered_text for word in ["football", "fifa", "mundial", "corea", "mexico", "yahoo.com", "instagram"]):
            score -= 10

    terms = search_query_terms(query)
    if terms:
        matched = sum(1 for term in terms if term in lowered_text)
        score += min(matched, 8) * 0.8

    item["search_score"] = round(score, 3)
    return score


def is_high_value_search_result(query, item):
    return score_search_result(query, dict(item)) >= (8 if is_finance_query(query) else 4)


def add_search_result(results, seen, title, url, max_results):
    if not url.startswith("http"):
        return

    domain = urlparse(url).netloc.lower()
    blocked_domains = [
        "duckduckgo.com",
        "www.duckduckgo.com",
        "bing.com",
        "www.bing.com",
    ]

    if domain in blocked_domains or domain in LOW_VALUE_SEARCH_DOMAINS:
        return

    if url in seen:
        return

    seen.add(url)
    results.append({
        "title": title or url,
        "url": url,
    })


def duckduckgo_search(query, max_results=5):
    url = f"https://lite.duckduckgo.com/lite/?q={quote_plus(query)}"
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    response = requests.get(url, headers=headers, timeout=WEB_SEARCH_TIMEOUT_SECONDS)
    response.raise_for_status()

    results = []
    seen = set()

    links = re.findall(r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', response.text, re.S)

    for raw_url, raw_title in links:
        parsed_url = normalize_search_url(raw_url)
        title = clean_title(raw_title)
        add_search_result(results, seen, title, parsed_url, max_results)

        if len(results) >= max_results:
            break

    return results


def bing_search(query, max_results=5):
    url = f"https://www.bing.com/search?q={quote_plus(query)}"
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    response = requests.get(url, headers=headers, timeout=WEB_SEARCH_TIMEOUT_SECONDS)
    response.raise_for_status()

    results = []
    seen = set()

    blocks = re.findall(r'<li class="b_algo".*?</li>', response.text, re.S)

    for block in blocks:
        match = re.search(r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', block, re.S)
        if not match:
            continue

        parsed_url = normalize_search_url(match.group(1))
        title = clean_title(match.group(2))
        add_search_result(results, seen, title, parsed_url, max_results)

        if len(results) >= max_results:
            break

    return results


def baidu_search(query, max_results=5):
    url = f"https://www.baidu.com/s?ie=utf-8&wd={quote_plus(query)}"
    headers = {
        "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X)"
    }
    response = requests.get(url, headers=headers, timeout=WEB_SEARCH_TIMEOUT_SECONDS)
    response.raise_for_status()

    results = []
    seen = set()

    if "安全验证" in response.text:
        return results

    links = re.findall(
        r'data-log="[^"]*&quot;mu&quot;:&quot;([^"]+?)&quot;[^"]*".{0,3000}?<h3[^>]*>(.*?)</h3>',
        response.text,
        re.S,
    )

    for raw_url, raw_title in links:
        parsed_url = html.unescape(raw_url)
        title = clean_title(raw_title)
        add_search_result(results, seen, title, parsed_url, max_results)

        if len(results) >= max_results:
            break

    return results


def search_web(query, max_results=5):
    provider_results = []
    expanded_queries = expand_web_queries(query)
    per_query_limit = max(max_results, 8)

    if len(expanded_queries) > 1:
        print("扩展检索词：" + "｜".join(expanded_queries[:4]))

    for expanded_query in expanded_queries:
        for search_name, search_func in [
            ("百度", baidu_search),
            ("DuckDuckGo Lite", duckduckgo_search),
            ("Bing", bing_search),
        ]:
            try:
                results = search_func(expanded_query, max_results=per_query_limit)
                print(f"{search_name} 找到 {len(results)} 个结果。")
            except Exception as e:
                print(f"{search_name} 搜索失败：{e}")
                continue

            provider_results.append((search_name, expanded_query, results))

    candidates = []
    seen = set()

    def append_item(item, search_name, expanded_query, provider_rank):
        if item["url"] in seen:
            return False
        seen.add(item["url"])
        scored_item = dict(item)
        base_score = score_search_result(expanded_query, scored_item)
        provider_bonus = 1.5 if search_name == "百度" else 0.7
        scored_item["search_provider"] = search_name
        scored_item["search_query"] = expanded_query
        scored_item["search_rank"] = provider_rank
        scored_item["search_score"] = round(base_score + provider_bonus - provider_rank * 0.08, 3)
        candidates.append(scored_item)
        return True

    for search_name, expanded_query, results in provider_results:
        for index, item in enumerate(results):
            append_item(item, search_name, expanded_query, index)

    if not candidates:
        return []

    candidates.sort(key=lambda item: item.get("search_score", 0), reverse=True)

    if is_finance_query(query):
        strong_candidates = [item for item in candidates if item.get("search_score", 0) >= 5]
        if len(strong_candidates) >= max_results:
            return strong_candidates[:max_results]

    return candidates[:max_results]


def normalize_web_query(query):
    compact_query = query.strip()
    compact_query = re.sub(r"^(你知道|请问|帮我查一下|查一下|搜索一下|了解一下|介绍一下)\s*", "", compact_query)
    compact_query = re.sub(r"[吗嘛呢？?]+$", "", compact_query).strip()
    lowered_query = compact_query.lower()

    if "agent" in lowered_query and "ai" not in lowered_query:
        if lowered_query.startswith("agent"):
            return re.sub(r"(?i)^agent\s*", "AI Agent ", compact_query, count=1).strip()
        return f"AI Agent {compact_query}"

    return compact_query


class WebTextFetchError(RuntimeError):
    pass


def web_request_headers(user_agent):
    return {
        "User-Agent": user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,text/plain;q=0.8,*/*;q=0.7",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Cache-Control": "no-cache",
    }


def normalize_response_text(response):
    detected_encoding = response.apparent_encoding or "utf-8"
    if not response.encoding or response.encoding.lower() in {"iso-8859-1", "ascii"}:
        response.encoding = detected_encoding
    text = response.text or ""
    mojibake_markers = ["Ã", "Â", "æ", "ç", "è", "å"]
    if detected_encoding and sum(text.count(marker) for marker in mojibake_markers) >= 8:
        response.encoding = response.apparent_encoding or "utf-8"
        text = response.text or ""
    return text


def contains_block_marker(text):
    lowered = (text or "").lower()
    return any(marker.lower() in lowered for marker in WEB_BLOCK_MARKERS)


def validate_web_text(text, source_label):
    text = (text or "").strip()
    if not text:
        raise WebTextFetchError(f"{source_label} 没有抽取到正文")
    if contains_block_marker(text):
        raise WebTextFetchError(f"{source_label} 命中安全验证或访问限制")
    if len(text) < WEB_MIN_TEXT_CHARS:
        raise WebTextFetchError(f"{source_label} 正文太短：{len(text)} 字")
    return text[:8000]


def extract_text_from_html(html_text):
    parser = SimpleTextExtractor()
    parser.feed(html_text)
    return parser.get_text()


def remaining_timeout(deadline: float | None, fallback: float) -> float:
    if deadline is None:
        return fallback
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise WebTextFetchError("网页收集达到时间预算")
    return max(1.0, min(fallback, remaining))


def fetch_web_text_with_user_agent(url, user_agent, label, deadline: float | None = None):
    response = requests.get(
        url,
        headers=web_request_headers(user_agent),
        timeout=remaining_timeout(deadline, WEB_FETCH_TIMEOUT_SECONDS),
        allow_redirects=True,
    )
    response.raise_for_status()

    content_type = response.headers.get("content-type", "")
    if content_type and "text/html" not in content_type and "text/plain" not in content_type:
        raise WebTextFetchError(f"{label} 非文本页面：{content_type}")

    raw_text = normalize_response_text(response)
    if "text/plain" in content_type:
        text = raw_text
    else:
        text = extract_text_from_html(raw_text)

    return validate_web_text(text, label)


def jina_reader_url(url):
    return f"https://r.jina.ai/{url}"


def fetch_web_text_with_jina(url, deadline: float | None = None):
    if not ENABLE_JINA_READER:
        raise WebTextFetchError("Jina Reader 未启用")

    headers = {
        "User-Agent": DESKTOP_USER_AGENT,
        "Accept": "text/plain, text/markdown, */*",
    }
    jina_key = os.getenv("JINA_API_KEY")
    if jina_key:
        headers["Authorization"] = f"Bearer {jina_key}"

    response = requests.get(
        jina_reader_url(url),
        headers=headers,
        timeout=remaining_timeout(deadline, JINA_READER_TIMEOUT_SECONDS),
        allow_redirects=True,
    )
    response.raise_for_status()
    text = normalize_response_text(response)
    return validate_web_text(text, "Jina Reader")


def fetch_web_text(url, deadline: float | None = None):
    errors = []

    for label, user_agent in [
        ("桌面浏览器请求", DESKTOP_USER_AGENT),
        ("移动浏览器请求", MOBILE_USER_AGENT),
    ]:
        try:
            text = fetch_web_text_with_user_agent(url, user_agent, label, deadline=deadline)
            print(f"正文读取成功：{label}")
            return text
        except Exception as e:
            errors.append(f"{label}: {e}")
            print(f"正文读取失败：{label}｜{e}")

    try:
        text = fetch_web_text_with_jina(url, deadline=deadline)
        print("正文读取成功：Jina Reader")
        return text
    except Exception as e:
        errors.append(f"Jina Reader: {e}")
        print(f"正文读取失败：Jina Reader｜{e}")

    raise WebTextFetchError("；".join(errors))


def web_collect(query, max_results=3, chroma_path=CHROMA_PATH, metadata_scope=None, max_seconds=None):
    deadline = time.monotonic() + float(max_seconds or WEB_COLLECT_MAX_SECONDS)
    search_query = normalize_web_query(query)
    print("正在搜索网页...")
    if search_query != query:
        print(f"检索词改写：{search_query}")
    candidate_limit = min(
        max(max_results * WEB_SEARCH_CANDIDATE_MULTIPLIER, max_results + 3),
        max(max_results + 2, 5),
    )
    results = search_web(search_query, max_results=candidate_limit)

    if not results:
        print("没有搜索到网页结果。")
        return []

    ingested_sources = []
    successful_sources = []

    def ingest_search_result_fallback(item, reason):
        should_ingest_summary = (
            INGEST_FAILED_SEARCH_RESULTS
            or is_high_value_search_result(search_query, item)
        )
        if not should_ingest_summary:
            print("网页正文不可读，跳过写入知识库。")
            return

        fallback_text = (
            f"网页标题线索\n"
            f"查询：{search_query}\n"
            f"标题：{item['title']}\n"
            f"链接：{item['url']}\n"
            f"可信度：低。该资料只来自搜索结果标题和链接，未读取完整正文；只能作为线索，回答时需要提示用户核验原网页。"
        )
        source_name = f"网页标题线索：{item['title']}"
        chunk_count = add_text_to_chroma(
            fallback_text,
            source=source_name,
            source_type="web",
            url=item["url"],
            content_type="search_result_summary",
            chroma_path=chroma_path,
            metadata_scope=metadata_scope,
        )
        if chunk_count:
            print(f"已写入网页标题线索：{chunk_count} 块")
            ingested_sources.append(item)

    for item in results:
        if time.monotonic() >= deadline:
            print("联网收集达到时间预算，停止读取更多网页。")
            break

        title = item["title"]
        url = item["url"]
        print(f"正在读取网页：{title}")

        try:
            text = fetch_web_text(url, deadline=deadline)
        except Exception as e:
            print(f"读取失败：{e}")
            ingest_search_result_fallback(item, str(e))
            continue

        if len(text) < WEB_MIN_TEXT_CHARS:
            print("网页正文太少，跳过。")
            ingest_search_result_fallback(item, "网页正文太少")
            continue

        source_name = f"网页：{title}"
        chunk_count = add_text_to_chroma(
            text,
            source=source_name,
            source_type="web",
            url=url,
            chroma_path=chroma_path,
            metadata_scope=metadata_scope,
        )
        print(f"已写入网页资料：{chunk_count} 块")
        ingested_sources.append(item)
        successful_sources.append(item)

        if len(successful_sources) >= max_results:
            break

    return successful_sources


def image_to_data_url(image_path):
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type:
        mime_type = "image/jpeg"

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def image_bytes_to_data_url(image_bytes, mime_type):
    if not mime_type:
        mime_type = "image/jpeg"

    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def describe_image(image_path, question):
    client = get_qwen_client()
    if client is None:
        print("没有找到 DASHSCOPE_API_KEY。")
        print("请在终端设置：export DASHSCOPE_API_KEY=\"sk-xxx\"")
        return None

    if not os.path.exists(image_path):
        print(f"图片不存在：{image_path}")
        return None

    data_url = image_to_data_url(image_path)
    prompt = question or "请详细描述这张图片中和用户问题相关的信息。"

    response = client.chat.completions.create(
        model=QWEN_VL_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        temperature=0.3,
    )

    return response.choices[0].message.content


def describe_image_bytes(image_bytes, mime_type, question):
    client = get_qwen_client()
    if client is None:
        raise RuntimeError("没有找到 DASHSCOPE_API_KEY。")

    data_url = image_bytes_to_data_url(image_bytes, mime_type)
    prompt = question or "请详细描述这张图片中和用户问题相关的信息。"

    response = client.chat.completions.create(
        model=QWEN_VL_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        temperature=0.3,
        max_tokens=500,
    )

    return response.choices[0].message.content


def answer_with_rag(
    question,
    use_web=False,
    top_k=3,
    web_max_results=3,
    preferred_sources=None,
    chroma_path=CHROMA_PATH,
    metadata_scope=None,
):
    if use_web:
        web_collect(question, max_results=web_max_results, chroma_path=chroma_path, metadata_scope=metadata_scope)

    search_results = search_chroma(
        question,
        top_k=top_k,
        preferred_sources=preferred_sources,
        chroma_path=chroma_path,
        metadata_scope=metadata_scope,
    )
    answer = ask_deepseek(question, search_results)

    if answer is None:
        raise RuntimeError("没有找到 DEEPSEEK_API_KEY。")

    conversation_history.append({"role": "user", "content": question})
    conversation_history.append({"role": "assistant", "content": answer})

    return {
        "answer": answer,
        "sources": search_results,
    }


def answer_question(question, use_web=False):
    global last_sources

    if use_web:
        web_collect(question)

    search_results = search_chroma(question)
    last_sources = search_results

    print(f"检索到资料：{len(search_results)} 条")
    print("正在让 DeepSeek 生成回答...")

    answer = ask_deepseek(question, search_results)
    if answer is None:
        return

    print("\nAI 回答：")
    print(answer)
    print()

    conversation_history.append({"role": "user", "content": question})
    conversation_history.append({"role": "assistant", "content": answer})


def handle_image_command(command):
    try:
        parts = shlex.split(command)
    except ValueError as e:
        print(f"图片命令解析失败：{e}")
        print("用法：/image 图片路径 你的问题")
        return

    if len(parts) < 2:
        print("用法：/image 图片路径 你的问题")
        return

    image_path = parts[1]
    question = " ".join(parts[2:]).strip() if len(parts) >= 3 else "请描述这张图片。"

    print("正在理解图片...")
    image_summary = describe_image(image_path, question)
    if not image_summary:
        return

    source = f"图片：{os.path.basename(image_path)}"
    add_text_to_chroma(image_summary, source=source, source_type="image", url=image_path)

    print("图片理解结果已写入知识库。")
    answer_question(question)


def print_help():
    print("""
可用命令：
普通问题              直接走本地知识库 RAG
/web 你的问题         先联网搜索并写入知识库，再回答
/image 图片路径 问题   先理解图片并写入知识库，再回答
/sources              查看上一次回答用到的资料来源
/help                 查看帮助
/q                    退出
""")


def print_sources():
    if not last_sources:
        print("还没有来源记录。")
        return

    print("上一次检索来源：")
    for index, item in enumerate(last_sources, start=1):
        line = f"{index}. {item['source']}"
        if item.get("url"):
            line += f" | {item['url']}"
        print(line)


def main():
    print("RAG Agent Pro 已启动。输入 /help 查看命令，输入 /q 退出。")

    while True:
        try:
            user_input = input("\n你：").strip()
        except KeyboardInterrupt:
            print("\n已退出。")
            break

        if not user_input:
            continue

        if user_input in {"/q", "q", "quit", "exit"}:
            print("已退出。")
            break

        if user_input == "/help":
            print_help()
            continue

        if user_input == "/sources":
            print_sources()
            continue

        if user_input.startswith("/web "):
            question = user_input[len("/web "):].strip()
            answer_question(question, use_web=True)
            continue

        if user_input.startswith("/image "):
            handle_image_command(user_input)
            continue

        should_use_web = any(word in user_input for word in ["最新", "今天", "最近", "新闻", "网上", "互联网"])
        answer_question(user_input, use_web=should_use_web)


if __name__ == "__main__":
    main()
