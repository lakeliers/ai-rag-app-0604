import base64
import html
import mimetypes
import os
import re
import shlex
import math
import time
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
CHUNKING_STRATEGIES = {
    CHUNKING_PLAIN,
    CHUNKING_PARENT_CHILD,
    CHUNKING_TABLE,
    CHUNKING_SUMMARY,
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


print("正在加载本地向量模型...")
LOCAL_FILES_ONLY = os.getenv("LOCAL_FILES_ONLY", "0") == "1"
embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME,
    local_files_only=LOCAL_FILES_ONLY,
)

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

conversation_history = []
last_sources = []
reranker_model = None
reranker_load_failed = False


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
    return embedding_model.encode(texts).tolist()


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
    definition_words = ["是什么", "定义", "概念", "解释一下", "介绍一下", "什么意思"]
    policy_words = ["prd", "需求", "规则", "政策", "制度", "上线时间", "价格", "方案", "口径"]

    if any(word in question for word in latest_words):
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


def bm25_retrieve(question, limit=20, metadata_filter=None):
    if metadata_filter:
        data = collection.get(include=["documents", "metadatas"], where=metadata_filter)
    else:
        data = collection.get(include=["documents", "metadatas"])
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

    for marker in FAILED_WEB_MARKERS:
        if marker.lower() in lowered_document:
            return True

    return False


def filter_answerable_rows(rows):
    answerable_rows = []

    for row in rows:
        if is_failed_web_result(row):
            row["context_skip_reason"] = "网页正文不可读，已排除"
            continue
        answerable_rows.append(row)

    return answerable_rows


def safe_id(text):
    text = re.sub(r"[^a-zA-Z0-9_\-.]+", "_", text)
    return text[:120]


def build_chunk_candidates(section, source, section_index, chunking_strategy):
    if chunking_strategy not in CHUNKING_STRATEGIES:
        chunking_strategy = CHUNKING_PARENT_CHILD

    parent_text = format_section_text(section)
    parent_id = f"{source}:{section_index}:{section.section_title or section.content_type or 'section'}"

    if chunking_strategy == CHUNKING_PLAIN:
        return [
            ChunkCandidate(text=text, chunk_type="plain")
            for text in split_text_fixed(parent_text)
        ]

    if chunking_strategy == CHUNKING_TABLE:
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


def add_text_to_chroma(text, source, source_type="local", url="", content_type="text", created_at=None, chunking_strategy=CHUNKING_PARENT_CHILD):
    section = ParsedSection(text=text, content_type=content_type)
    return add_sections_to_chroma(
        [section],
        source=source,
        source_type=source_type,
        url=url,
        created_at=created_at,
        chunking_strategy=chunking_strategy,
    )


def add_sections_to_chroma(sections, source, source_type="local", url="", created_at=None, chunking_strategy=CHUNKING_PARENT_CHILD):
    chunk_rows = []

    for section_index, section in enumerate(sections):
        chunk_candidates = build_chunk_candidates(section, source, section_index, chunking_strategy)
        summary = summarize_section_for_retrieval(section, force=True) if chunking_strategy == CHUNKING_SUMMARY else ""
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
    ids = []
    metadatas = []
    chunks = []
    document_key = f"{source_type}:{source}:{url or source}"

    for index, (section_index, chunk_index, section, chunk) in enumerate(chunk_rows):
        item_id = f"{source_type}_{safe_id(source)}_{now}_{section_index}_{chunk_index}"
        ids.append(item_id)
        chunks.append(chunk.text)
        metadatas.append({
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
        })

    embeddings = embed_texts(chunks)

    collection.upsert(
        ids=ids,
        documents=chunks,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    return len(chunks)


def seed_local_note(file_path="my_note.md"):
    if collection.count() > 0:
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


def vector_retrieve(question, limit=20, metadata_filter=None):
    query_embedding = embed_texts([question])
    if metadata_filter:
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=limit,
            where=metadata_filter,
        )
    else:
        results = collection.query(
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
):
    if retrieval_strategy not in RETRIEVAL_STRATEGIES:
        retrieval_strategy = RETRIEVAL_VECTOR_BM25_RRF
    query_profile = analyze_query(question)
    metadata_filter = build_metadata_filter(query_profile)
    preferred_sources = set(preferred_sources or [])
    if preferred_only and preferred_sources:
        metadata_filter = {"source": {"$in": list(preferred_sources)}}
    recall_limit = max(top_k * 8, 24)
    vector_rows = vector_retrieve(question, limit=recall_limit, metadata_filter=metadata_filter)
    bm25_rows = []
    if retrieval_strategy != RETRIEVAL_VECTOR_ONLY:
        bm25_rows = bm25_retrieve(question, limit=recall_limit, metadata_filter=metadata_filter)
    if metadata_filter and not preferred_only and not vector_rows and not bm25_rows:
        metadata_filter = None
        vector_rows = vector_retrieve(question, limit=recall_limit)
        if retrieval_strategy != RETRIEVAL_VECTOR_ONLY:
            bm25_rows = bm25_retrieve(question, limit=recall_limit)
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


def build_answer_prompt(question, search_results):
    context = build_context(search_results)
    history_text = build_history_text()

    return f"""你是一个可以使用知识库和网页资料的 RAG Agent。

请根据【资料】回答【用户问题】。
如果资料不足，请明确说资料不足，不要编造。
资料使用规则：
1. 上传资料和上传图片属于用户主动提供的信息，可信优先级最高。
2. 网络资料只作为补充；当网络资料和上传资料冲突时，优先采用上传资料。
3. 基础资料只作为兜底，不能压过用户上传资料和当前联网资料。
4. 如果资料来自网页，请提醒用户网页信息可能会变化。

【最近对话】
{history_text}

【资料】
{context}

【用户问题】
{question}

【回答要求】
1. 先给结论
2. 再用 2-4 条解释关键依据
3. 如果用户要求案例、类似案例、对标产品或产品例子，必须列出资料中可以支持的具体案例或产品名称，不能只给抽象趋势
4. 如果用户要求方案、报告、计划、清单、分析或建议，必须输出结构化交付物，至少包含：目标/结论、关键分析、具体建议或下一步计划、参考来源
5. 最后列出参考来源
"""


def ask_deepseek(question, search_results):
    client = get_deepseek_client()
    if client is None:
        print("没有找到 DEEPSEEK_API_KEY。")
        print("请在终端设置：export DEEPSEEK_API_KEY=\"sk-xxx\"")
        return None

    prompt = build_answer_prompt(question, search_results)

    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=700,
        timeout=LLM_TIMEOUT_SECONDS,
    )

    return response.choices[0].message.content


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

    if domain in blocked_domains:
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
    response = requests.get(url, headers=headers, timeout=15)
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
    response = requests.get(url, headers=headers, timeout=15)
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
    response = requests.get(url, headers=headers, timeout=15)
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
    all_results = []
    seen = set()

    for search_name, search_func in [
        ("百度", baidu_search),
        ("DuckDuckGo Lite", duckduckgo_search),
        ("Bing", bing_search),
    ]:
        try:
            results = search_func(query, max_results=max_results)
            print(f"{search_name} 找到 {len(results)} 个结果。")
        except Exception as e:
            print(f"{search_name} 搜索失败：{e}")
            continue

        for item in results:
            if item["url"] in seen:
                continue
            seen.add(item["url"])
            all_results.append(item)

            if len(all_results) >= max_results:
                return all_results

    return all_results


def normalize_web_query(query):
    compact_query = query.strip()
    lowered_query = compact_query.lower()

    if "agent" in lowered_query and "ai" not in lowered_query:
        return f"AI Agent {compact_query}"

    return compact_query


def fetch_web_text(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }
    response = requests.get(url, headers=headers, timeout=20)
    response.raise_for_status()

    content_type = response.headers.get("content-type", "")
    if "text/html" not in content_type and "text/plain" not in content_type:
        return ""

    parser = SimpleTextExtractor()
    parser.feed(response.text)
    text = parser.get_text()
    return text[:8000]


def web_collect(query, max_results=3):
    search_query = normalize_web_query(query)
    print("正在搜索网页...")
    if search_query != query:
        print(f"检索词改写：{search_query}")
    candidate_limit = max(max_results * WEB_SEARCH_CANDIDATE_MULTIPLIER, max_results + 3)
    results = search_web(search_query, max_results=candidate_limit)

    if not results:
        print("没有搜索到网页结果。")
        return []

    ingested_sources = []
    successful_sources = []

    def ingest_search_result_fallback(item, reason):
        if not INGEST_FAILED_SEARCH_RESULTS:
            print("网页正文不可读，跳过写入知识库。")
            return

        fallback_text = (
            f"网页搜索结果摘要\n"
            f"查询：{search_query}\n"
            f"标题：{item['title']}\n"
            f"链接：{item['url']}\n"
            f"说明：完整网页正文读取失败或正文过短，当前仅保留搜索结果标题和链接作为低置信度网页资料。\n"
            f"失败原因：{reason}"
        )
        source_name = f"网页搜索结果：{item['title']}"
        chunk_count = add_text_to_chroma(
            fallback_text,
            source=source_name,
            source_type="web",
            url=item["url"],
            content_type="search_result_failed",
        )
        if chunk_count:
            print(f"已写入搜索结果摘要：{chunk_count} 块")
            ingested_sources.append(item)

    for item in results:
        title = item["title"]
        url = item["url"]
        print(f"正在读取网页：{title}")

        try:
            text = fetch_web_text(url)
        except Exception as e:
            print(f"读取失败：{e}")
            ingest_search_result_fallback(item, str(e))
            continue

        if len(text) < WEB_MIN_TEXT_CHARS:
            print("网页正文太少，跳过。")
            ingest_search_result_fallback(item, "网页正文太少")
            continue

        source_name = f"网页：{title}"
        chunk_count = add_text_to_chroma(text, source=source_name, source_type="web", url=url)
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
):
    if use_web:
        web_collect(question, max_results=web_max_results)

    search_results = search_chroma(
        question,
        top_k=top_k,
        preferred_sources=preferred_sources,
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
