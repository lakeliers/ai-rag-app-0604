const API_BASE = process.env.NEXT_PUBLIC_AGENT_API || "http://127.0.0.1:8000";

export async function getStatus() {
  const response = await fetch(`${API_BASE}/api/status`, { cache: "no-store" });
  if (!response.ok) throw new Error("状态读取失败");
  return response.json();
}

export async function uploadFiles({ sessionId, files, chunkingStrategy }) {
  const formData = new FormData();
  formData.append("session_id", sessionId);
  formData.append("chunking_strategy", chunkingStrategy.join(","));
  files.forEach((file) => formData.append("files", file));

  const response = await fetch(`${API_BASE}/api/upload`, {
    method: "POST",
    body: formData
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || "上传失败");
  return data;
}

export async function sendChat({ sessionId, question, config }) {
  const response = await fetch(`${API_BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, question, config })
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || "请求失败");
  return data;
}

export async function submitBadcase(payload) {
  const response = await fetch(`${API_BASE}/api/badcase`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || "反馈提交失败");
  return data;
}
