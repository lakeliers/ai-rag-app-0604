import hashlib
import mimetypes
import os
from dataclasses import dataclass
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import agent_runtime
import autonomous_agent
import badcase_manager
import memory_manager
import parsing_layer
import rag_agent_core as agent
import run_lifecycle


def apply_env_defaults():
    if os.getenv("SEED_TEACHING_MEMORY", "1") != "0":
        memory_manager.seed_default_memories_if_empty()
    if os.getenv("AUTO_SEED_LOCAL_NOTE", "0") == "1":
        agent.seed_local_note()


apply_env_defaults()

app = FastAPI(title="agent for train API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


SESSION_UPLOADS: dict[str, dict[str, str]] = {}
SESSION_UPLOAD_STATUS: dict[str, list[dict]] = {}


@dataclass
class UploadedFileAdapter:
    name: str
    content_type: str
    data: bytes

    @property
    def type(self):
        return self.content_type

    def getvalue(self):
        return self.data


class ChatConfig(BaseModel):
    run_mode: str = "normal"
    source_strategy: str = agent_runtime.SOURCE_STRATEGY_AUTO
    retrieval_strategy: str = agent.RETRIEVAL_VECTOR_BM25_RRF
    context_packing_strategy: str = agent.CONTEXT_STRICT_BUDGET
    chunking_strategy: list[str] = Field(default_factory=lambda: ["parent_child", "table"])
    router_mode: str = agent_runtime.ROUTER_MODE_RULES
    planner_type: str = agent_runtime.PLANNER_FALLBACK_MIXED
    evaluator_type: str = agent_runtime.EVALUATOR_RULES
    multi_agent_architecture: str = agent_runtime.MULTI_AGENT_AUTO
    memory_enabled: bool = True
    memory_route_strategy: str = agent_runtime.MEMORY_ROUTE_AUTO
    top_k: int = 3
    web_max_results: int = 2
    max_autonomous_steps: int = 3
    deepseek_model: str | None = None


class ChatRequest(BaseModel):
    session_id: str = ""
    question: str
    config: ChatConfig = Field(default_factory=ChatConfig)


class RunCreateRequest(BaseModel):
    session_id: str = ""
    question: str
    execution_mode: str = "async"
    config: dict = Field(default_factory=dict)


class RunResumeRequest(BaseModel):
    session_id: str
    user_input: str


class RunCancelRequest(BaseModel):
    session_id: str


class BadcaseRequest(BaseModel):
    trace_id: str = ""
    user_input: str
    actual_answer: str
    issue_summary: str
    expected_behavior: str = ""
    severity: str = "medium"
    save_target: str = "local"
    sources_used: list[str] = Field(default_factory=list)


def normalize_session_id(session_id: str | None):
    return session_id or f"session_{uuid4().hex[:12]}"


def upload_key(file_name: str, file_bytes: bytes, chunking_strategy: list[str]):
    content_hash = hashlib.sha256(file_bytes).hexdigest()[:16]
    strategy = ",".join(sorted(chunking_strategy or []))
    return f"{file_name}:{len(file_bytes)}:{content_hash}:{strategy}"


def is_image_file(file_name: str, content_type: str):
    guessed_type, _ = mimetypes.guess_type(file_name)
    return (content_type or guessed_type or "").startswith("image/")


def extract_source_types(sources):
    source_types = []
    for source in sources or []:
        source_type = source.get("source_type", "unknown")
        if source_type not in source_types:
            source_types.append(source_type)
    return source_types


def compact_source(source: dict):
    return {
        "source": source.get("source", ""),
        "source_type": source.get("source_type", ""),
        "url": source.get("url", ""),
        "page": source.get("page"),
        "section_title": source.get("section_title", ""),
        "content_type": source.get("content_type", ""),
        "text": (source.get("text", "") or "")[:420],
        "final_score": source.get("final_score"),
        "rerank_score": source.get("rerank_score"),
    }


def ingest_file(session_id: str, uploaded_file: UploadedFileAdapter, chunking_strategy: list[str]):
    file_bytes = uploaded_file.getvalue()
    key = upload_key(uploaded_file.name, file_bytes, chunking_strategy)
    session_cache = SESSION_UPLOADS.setdefault(session_id, {})
    if key in session_cache:
        return {
            "source": session_cache[key],
            "chunk_count": 0,
            "status": "already_ingested",
        }

    metadata_scope = {"session_id": session_id}
    if is_image_file(uploaded_file.name, uploaded_file.type):
        if not os.getenv("DASHSCOPE_API_KEY"):
            return {
                "source": f"图片：{uploaded_file.name}",
                "chunk_count": 0,
                "status": "skipped",
                "message": "图片需要配置 DASHSCOPE_API_KEY 才能解析。",
            }
        summary = agent.describe_image_bytes(
            file_bytes,
            uploaded_file.type,
            "请提取这张图片中的关键信息，整理成适合知识库检索的文字资料。",
        )
        source = f"图片：{uploaded_file.name}"
        chunk_count = agent.add_text_to_chroma(
            summary,
            source=source,
            source_type="upload",
            url=uploaded_file.name,
            content_type="image",
            chunking_strategy=chunking_strategy,
            metadata_scope=metadata_scope,
        )
    else:
        sections = parsing_layer.read_upload_as_sections(uploaded_file)
        source = f"上传：{uploaded_file.name}"
        chunk_count = agent.add_sections_to_chroma(
            sections,
            source=source,
            source_type="upload",
            url=uploaded_file.name,
            chunking_strategy=chunking_strategy,
            metadata_scope=metadata_scope,
        )

    session_cache[key] = source
    return {
        "source": source,
        "chunk_count": chunk_count,
        "status": "ingested",
    }


@app.get("/api/status")
def status():
    return {
        "deepseek_configured": bool(os.getenv("DEEPSEEK_API_KEY")),
        "dashscope_configured": bool(os.getenv("DASHSCOPE_API_KEY")),
        "reranker_enabled": agent.ENABLE_RERANKER,
        "default_model": agent.DEEPSEEK_MODEL,
    }


@app.post("/api/runs")
def create_product_run(request: RunCreateRequest):
    session_id = normalize_session_id(request.session_id)
    run = run_lifecycle.create_run(
        session_id=session_id,
        user_input=request.question,
        execution_mode=request.execution_mode,
        config=request.config,
    )
    if request.execution_mode == "async":
        run = run_lifecycle.transition_run(
            run["run_id"],
            run_lifecycle.STATUS_QUEUED,
            actor="backend_api",
            reason="异步任务已写入 Task Queue（任务队列）。",
            current_step="queued",
        )
    return run


@app.get("/api/runs/{run_id}")
def get_product_run(run_id: str, session_id: str):
    run = run_lifecycle.get_run(run_id, session_id=session_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run 不存在或不属于当前会话。")
    return run


@app.post("/api/runs/{run_id}/cancel")
def cancel_product_run(run_id: str, request: RunCancelRequest):
    run = run_lifecycle.get_run(run_id, session_id=request.session_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run 不存在或不属于当前会话。")
    run = run_lifecycle.request_cancel(run_id)
    return run_lifecycle.complete_cancel(run_id)


@app.post("/api/runs/{run_id}/resume")
def resume_product_run(run_id: str, request: RunResumeRequest):
    run = run_lifecycle.get_run(run_id, session_id=request.session_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run 不存在或不属于当前会话。")
    try:
        return run_lifecycle.resume_run(run_id, user_input=request.user_input)
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.post("/api/upload")
async def upload_files(
    session_id: str = Form(""),
    chunking_strategy: str = Form("parent_child,table"),
    files: list[UploadFile] = File(...),
):
    session_id = normalize_session_id(session_id)
    strategies = [item.strip() for item in chunking_strategy.split(",") if item.strip()]
    statuses = []
    for file in files:
        data = await file.read()
        adapted = UploadedFileAdapter(
            name=file.filename or "upload",
            content_type=file.content_type or "",
            data=data,
        )
        statuses.append(ingest_file(session_id, adapted, strategies))

    SESSION_UPLOAD_STATUS.setdefault(session_id, []).extend(statuses)
    return {
        "session_id": session_id,
        "uploaded_sources": list(SESSION_UPLOADS.get(session_id, {}).values()),
        "statuses": statuses,
    }


@app.post("/api/chat")
def chat(request: ChatRequest):
    if not os.getenv("DEEPSEEK_API_KEY"):
        raise HTTPException(status_code=400, detail="请先配置 DEEPSEEK_API_KEY。")

    session_id = normalize_session_id(request.session_id)
    config = request.config
    if config.deepseek_model:
        agent.DEEPSEEK_MODEL = config.deepseek_model
        agent_runtime.PLANNER_MODEL = config.deepseek_model

    preferred_sources = list(SESSION_UPLOADS.get(session_id, {}).values())
    memory_context = ""

    progress_events = []

    def progress_callback(event):
        progress_events.append(event)

    if config.run_mode == "autonomous":
        use_autonomous, reason = autonomous_agent.should_use_autonomous_mode(
            request.question,
            router_mode=config.router_mode,
        )
        if use_autonomous:
            result = autonomous_agent.run_autonomous_agent(
                request.question,
                top_k=config.top_k,
                web_max_results=config.web_max_results,
                max_steps=config.max_autonomous_steps,
                preferred_sources=preferred_sources,
                router_mode=config.router_mode,
                source_strategy=config.source_strategy,
                retrieval_strategy=config.retrieval_strategy,
                context_packing_strategy=config.context_packing_strategy,
                planner_type=config.planner_type,
                evaluator_type=config.evaluator_type,
                memory_context=memory_context,
                memory_enabled=config.memory_enabled,
                memory_route_strategy=config.memory_route_strategy,
                multi_agent_architecture=config.multi_agent_architecture,
                metadata_scope={"session_id": session_id},
                progress_callback=progress_callback,
            )
        else:
            result = agent_runtime.run_agent_pro(
                request.question,
                use_web=True,
                top_k=config.top_k,
                web_max_results=config.web_max_results,
                preferred_sources=preferred_sources,
                router_mode=config.router_mode,
                source_strategy=config.source_strategy,
                retrieval_strategy=config.retrieval_strategy,
                context_packing_strategy=config.context_packing_strategy,
                planner_type=config.planner_type,
                evaluator_type=config.evaluator_type,
                memory_context=memory_context,
                memory_enabled=config.memory_enabled,
                memory_route_strategy=config.memory_route_strategy,
                multi_agent_architecture=config.multi_agent_architecture,
                metadata_scope={"session_id": session_id},
                progress_callback=progress_callback,
            )
            result["planner_mode"] = "autonomous_fallback"
            result["autonomous_route_reason"] = reason
    else:
        result = agent_runtime.run_agent_pro(
            request.question,
            use_web=True,
            top_k=config.top_k,
            web_max_results=config.web_max_results,
            preferred_sources=preferred_sources,
            router_mode=config.router_mode,
            source_strategy=config.source_strategy,
            retrieval_strategy=config.retrieval_strategy,
            context_packing_strategy=config.context_packing_strategy,
            planner_type=config.planner_type,
            evaluator_type=config.evaluator_type,
            memory_context=memory_context,
            memory_enabled=config.memory_enabled,
            memory_route_strategy=config.memory_route_strategy,
            multi_agent_architecture=config.multi_agent_architecture,
            metadata_scope={"session_id": session_id},
            progress_callback=progress_callback,
        )

    trace_id = f"trace_{uuid4().hex[:12]}"
    sources = result.get("sources", [])
    return {
        "trace_id": trace_id,
        "session_id": session_id,
        "answer": result.get("answer", ""),
        "sources": [compact_source(source) for source in sources],
        "source_types": extract_source_types(sources),
        "steps": result.get("steps", []),
        "progress_events": progress_events,
        "planner_mode": result.get("planner_mode", ""),
        "teaching_config": result.get("teaching_config", {}),
        "memory": {
            "enabled": config.memory_enabled,
            "count": len(result.get("memory_used", [])),
        },
        "uploads": SESSION_UPLOAD_STATUS.get(session_id, []),
    }


@app.post("/api/badcase")
def save_badcase(request: BadcaseRequest):
    case = {
        "case_id": f"badcase_{uuid4().hex[:8]}",
        "suite": "regression",
        "category": "user_feedback",
        "input": request.user_input,
        "expected_behavior": request.expected_behavior or request.issue_summary,
        "rubric": {
            "must_address": [request.expected_behavior or request.issue_summary],
            "must_not": [],
        },
        "severity": request.severity,
    }
    validation = badcase_manager.validate_regression_case(case)
    if not validation["ok"]:
        return {
            "ok": False,
            "errors": validation["errors"],
        }

    result = badcase_manager.save_badcase(
        case=case,
        actual_answer=request.actual_answer,
        sources_used=request.sources_used,
        trace_id=request.trace_id,
        save_target=request.save_target,
    )
    return result
