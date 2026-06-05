import base64
import html
import mimetypes
import os
import re
import shlex
import time
from html.parser import HTMLParser
from urllib.parse import quote_plus, unquote, urlparse, parse_qs

import chromadb
import requests
from openai import OpenAI
from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL_NAME = "shibing624/text2vec-base-chinese"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "file_docs"
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
QWEN_VL_MODEL = os.getenv("QWEN_VL_MODEL", "qwen-vl-plus")


print("正在加载本地向量模型...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, local_files_only=True)

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

conversation_history = []
last_sources = []


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
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def get_qwen_client():
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")


def split_text(text, chunk_size=500, chunk_overlap=80):
    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = start + chunk_size - chunk_overlap

    return chunks


def embed_texts(texts):
    return embedding_model.encode(texts).tolist()


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


def safe_id(text):
    text = re.sub(r"[^a-zA-Z0-9_\-.]+", "_", text)
    return text[:120]


def add_text_to_chroma(text, source, source_type="local", url=""):
    chunks = split_text(text)
    if not chunks:
        return 0

    now = int(time.time())
    ids = []
    metadatas = []

    for index, chunk in enumerate(chunks):
        item_id = f"{source_type}_{safe_id(source)}_{now}_{index}"
        ids.append(item_id)
        metadatas.append({
            "source": source,
            "source_type": source_type,
            "url": url,
            "chunk_index": index,
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


def search_chroma(question, top_k=3):
    query_embedding = embed_texts([question])
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=max(top_k * 4, 20),
    )

    search_results = []
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    question_keywords = extract_query_keywords(question)

    for document, metadata, distance in zip(documents, metadatas, distances):
        search_results.append({
            "document": document,
            "source": metadata.get("source", "未知来源"),
            "source_type": metadata.get("source_type", "unknown"),
            "url": metadata.get("url", ""),
            "chunk_index": metadata.get("chunk_index", 0),
            "distance": distance,
            "keyword_score": keyword_score(document, question_keywords),
        })

    search_results.sort(
        key=lambda item: (
            -item["keyword_score"],
            item["distance"],
        )
    )

    return search_results[:top_k]


def build_context(search_results):
    if not search_results:
        return "没有检索到资料。"

    parts = []
    for index, item in enumerate(search_results, start=1):
        source_line = item["source"]
        if item.get("url"):
            source_line += f" | {item['url']}"

        part = f"""资料 {index}：
来源：{source_line}
类型：{item['source_type']}
块编号：{item['chunk_index']}
内容：{item['document']}
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
如果资料来自网页，请提醒用户网页信息可能会变化。

【最近对话】
{history_text}

【资料】
{context}

【用户问题】
{question}

【回答要求】
1. 先给结论
2. 再用 2-4 条解释关键依据
3. 最后列出参考来源
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
    print("正在搜索网页...")
    results = search_web(query, max_results=max_results)

    if not results:
        print("没有搜索到网页结果。")
        return []

    ingested_sources = []

    for item in results:
        title = item["title"]
        url = item["url"]
        print(f"正在读取网页：{title}")

        try:
            text = fetch_web_text(url)
        except Exception as e:
            print(f"读取失败：{e}")
            continue

        if len(text) < 100:
            print("网页正文太少，跳过。")
            continue

        source_name = f"网页：{title}"
        chunk_count = add_text_to_chroma(text, source=source_name, source_type="web", url=url)
        print(f"已写入网页资料：{chunk_count} 块")
        ingested_sources.append(item)

    return ingested_sources


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


def answer_with_rag(question, use_web=False, top_k=3, web_max_results=3):
    if use_web:
        web_collect(question, max_results=web_max_results)

    search_results = search_chroma(question, top_k=top_k)
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
