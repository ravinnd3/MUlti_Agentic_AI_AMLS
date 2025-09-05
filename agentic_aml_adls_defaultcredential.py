"""
Agentic Multi-Agent System for Azure ML Service (AMLS) — DefaultAzureCredential
-------------------------------------------------------------------------------

This file is an end-to-end Streamlit application designed to run inside Azure
Machine Learning compute or any environment with Azure AD identity access.
Key features:
- Uses DefaultAzureCredential (Managed Identity / az-login) for ADLS access
- Ingest PDFs from ADLS with primary parser + OCR fallback
- Chunk and build FAISS indexes (persisted back to ADLS under indexes/{doc_id})
- Agents: Coordinator, LanguageAnalysisAgent, NumericalAnalysisAgent, RiskAssessmentAgent
- Human-in-the-loop (HILP) approval surfaced via Streamlit

How to run:
1. Ensure compute has Managed Identity with Storage Blob Data Reader/Contributor
2. Set env vars: AZURE_OPENAI_*, AZURE_STORAGE_ACCOUNT_NAME, ADLS_FILESYSTEM
3. Install dependencies and run: streamlit run agentic_aml_adls_defaultcredential.py

"""

from __future__ import annotations
import os
import io
import json
import time
import uuid
import traceback
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# Azure storage (DefaultAzureCredential)
from azure.identity import DefaultAzureCredential
from azure.storage.filedatalake import DataLakeServiceClient

# PDF parsing / OCR
from pypdf import PdfReader
from pdf2image import convert_from_bytes
from PIL import Image
import pytesseract

# LangChain + Azure OpenAI
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai.chat_models.azure import AzureChatOpenAI
from langchain_openai.embeddings.azure import AzureOpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage

# Analysis
import numpy as np
from scipy import stats

# Streamlit for HILP UI
import streamlit as st

# -----------------------------
# Configuration / Env
# -----------------------------
AZ_OPENAI_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZ_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZ_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01")
AZ_CHAT_DEP = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
AZ_EMBED_DEP = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT")

STORAGE_ACCOUNT = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
ADLS_FILESYSTEM = os.getenv("ADLS_FILESYSTEM", "documents")
MODEL_TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", "0.1"))

if not (AZ_OPENAI_KEY and AZ_OPENAI_ENDPOINT and AZ_CHAT_DEP and AZ_EMBED_DEP and STORAGE_ACCOUNT):
    st.warning("Make sure Azure OpenAI and storage env vars are set. DefaultAzureCredential will be used for ADLS.")

# Local dirs
LOCAL_STORE_DIR = Path(".stores")
LOCAL_STORE_DIR.mkdir(parents=True, exist_ok=True)
CONTEXT_STORE_FILE = LOCAL_STORE_DIR / "context_store.json"

# -----------------------------
# ADLS helpers (DefaultAzureCredential)
# -----------------------------

def get_datalake_service_client() -> DataLakeServiceClient:
    account_url = f"https://{STORAGE_ACCOUNT}.dfs.core.windows.net"
    credential = DefaultAzureCredential()
    return DataLakeServiceClient(account_url=account_url, credential=credential)


def download_file_from_adls(path: str) -> bytes:
    svc = get_datalake_service_client()
    fs = svc.get_file_system_client(ADLS_FILESYSTEM)
    file_client = fs.get_file_client(path)
    downloader = file_client.download_file()
    return downloader.readall()


def upload_bytes_to_adls(data_bytes: bytes, remote_path: str):
    svc = get_datalake_service_client()
    fs = svc.get_file_system_client(ADLS_FILESYSTEM)
    file_client = fs.get_file_client(remote_path)
    try:
        file_client.create_file()
    except Exception:
        pass
    file_client.append_data(data_bytes, 0)
    file_client.flush_data(len(data_bytes))


def upload_directory_to_adls(local_dir: str, remote_prefix: str):
    svc = get_datalake_service_client()
    fs = svc.get_file_system_client(ADLS_FILESYSTEM)
    for root, _, files in os.walk(local_dir):
        for f in files:
            full = os.path.join(root, f)
            rel = os.path.relpath(full, local_dir)
            remote_path = os.path.join(remote_prefix, rel).replace("\\", "/")
            with open(full, "rb") as fh:
                data = fh.read()
            file_client = fs.get_file_client(remote_path)
            try:
                file_client.create_file()
            except Exception:
                pass
            file_client.append_data(data, 0)
            file_client.flush_data(len(data))

# -----------------------------
# Azure OpenAI helpers
# -----------------------------

def make_llm(deployment_name: Optional[str] = None, temperature: float = MODEL_TEMPERATURE):
    return AzureChatOpenAI(
        deployment_name=deployment_name or AZ_CHAT_DEP,
        openai_api_key=AZ_OPENAI_KEY,
        azure_openai_api_base=AZ_OPENAI_ENDPOINT,
        azure_openai_api_version=AZ_OPENAI_API_VERSION,
        temperature=temperature,
    )


def make_embeddings(deployment_name: Optional[str] = None):
    return AzureOpenAIEmbeddings(
        deployment=deployment_name or AZ_EMBED_DEP,
        openai_api_key=AZ_OPENAI_KEY,
        azure_openai_api_base=AZ_OPENAI_ENDPOINT,
        azure_openai_api_version=AZ_OPENAI_API_VERSION,
    )

# -----------------------------
# PDF parsing (primary + OCR fallback)
# -----------------------------

def parse_pdf_bytes_primary(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    texts = []
    for p in reader.pages:
        txt = p.extract_text() or ""
        texts.append(txt)
    combined = "\n\n".join(texts)
    if len(combined.strip()) < 50:
        raise ValueError("Primary parser returned very little text")
    return combined


def parse_pdf_bytes_ocr(pdf_bytes: bytes) -> str:
    pages = convert_from_bytes(pdf_bytes)
    texts = []
    for img in pages:
        txt = pytesseract.image_to_string(img)
        texts.append(txt)
    return "\n\n".join(texts)


def parse_pdf_with_fallback(adls_path: str) -> Tuple[str, str]:
    pdf_bytes = download_file_from_adls(adls_path)
    try:
        text = parse_pdf_bytes_primary(pdf_bytes)
        return text, "primary"
    except Exception:
        try:
            text = parse_pdf_bytes_ocr(pdf_bytes)
            return text, "ocr"
        except Exception as e:
            raise RuntimeError(f"Both parsers failed: {e}")

# -----------------------------
# Chunking + FAISS index building
# -----------------------------

def chunk_text_to_docs(text: str) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    docs = splitter.create_documents([text])
    return docs


def build_faiss_for_doc(doc_id: str, docs: List[Document], embeddings=None) -> str:
    embeddings = embeddings or make_embeddings()
    local_dir = LOCAL_STORE_DIR / f"faiss_{doc_id}"
    if local_dir.exists():
        for f in local_dir.glob("**/*"):
            if f.is_file():
                f.unlink()
    local_dir.mkdir(parents=True, exist_ok=True)
    vs = FAISS.from_documents(docs, embedding=embeddings)
    vs.save_local(str(local_dir))
    remote_prefix = f"indexes/{doc_id}"
    upload_directory_to_adls(str(local_dir), remote_prefix)
    return remote_prefix

# -----------------------------
# RAG retriever (download index from ADLS then load)
# -----------------------------

def make_rag_retriever_for_remote_index(remote_prefix: str, embeddings=None, k: int = 4):
    embeddings = embeddings or make_embeddings()
    tmp = tempfile.mkdtemp()
    svc = get_datalake_service_client()
    fs = svc.get_file_system_client(ADLS_FILESYSTEM)
    paths = list(fs.get_paths(remote_prefix))
    for p in paths:
        if getattr(p, 'is_directory', False):
            continue
        remote_path = p.name
        b = fs.get_file_client(remote_path).download_file().readall()
        rel = os.path.relpath(remote_path, remote_prefix).lstrip("/\\")
        local_path = os.path.join(tmp, rel)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "wb") as fh:
            fh.write(b)
    vs = FAISS.load_local(tmp, embeddings=embeddings, allow_dangerous_deserialization=True)

    def retriever(query: str) -> List[Document]:
        return vs.similarity_search(query, k=k)

    return retriever

# -----------------------------
# Agents & Coordinator
# -----------------------------

def simple_router_label(question: str) -> str:
    q = question.lower()
    if any(k in q for k in ["revenue", "growth", "eps", "income", "sales"]):
        return "growth"
    if any(k in q for k in ["risk", "debt", "liabilities", "default", "covenant"]):
        return "risk"
    if any(k in q for k in ["explain", "what", "who", "policy", "leave"]):
        return "language"
    if any(k in q for k in ["mean", "std", "median", "regression", "anova", "stat"]):
        return "numeric"
    return "language"


class Coordinator:
    def __init__(self):
        self.context_store = self._load_context()

    def _load_context(self) -> Dict[str, Any]:
        if CONTEXT_STORE_FILE.exists():
            return json.loads(CONTEXT_STORE_FILE.read_text())
        return {}

    def _save_context(self):
        CONTEXT_STORE_FILE.write_text(json.dumps(self.context_store, indent=2))
        try:
            upload_bytes_to_adls(CONTEXT_STORE_FILE.read_bytes(), "context/context_store.json")
        except Exception:
            pass

    def register_document_context(self, doc_id: str, metadata: Dict[str, Any]):
        self.context_store.setdefault("docs", {})[doc_id] = metadata
        self._save_context()

    def route(self, question: str) -> str:
        return simple_router_label(question)

    def add_agent_result(self, doc_id: str, agent_name: str, result: Dict[str, Any]):
        self.context_store.setdefault("results", {}).setdefault(doc_id, {}).setdefault(agent_name, []).append(result)
        self._save_context()


class LanguageAnalysisAgent:
    def __init__(self, retriever_callable):
        self.retriever = retriever_callable
        self.llm = make_llm()

    def extract_knowledge(self, query: str) -> Dict[str, Any]:
        snippets = self.retriever(query)
        context_text = "\n\n".join([d.page_content for d in snippets])
        prompt = f"Extract concise answers for the question and any structured facts.\nQuestion: {query}\nContext:\n{context_text}\nProvide JSON with keys: answer, facts(list of short facts)"
        resp = self.llm.generate([HumanMessage(content=prompt)])
        try:
            text = resp.generations[0][0].text
        except Exception:
            text = str(resp)
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = {"answer": text, "facts": []}
        return {"agent": "language", "query": query, "result": parsed}

    def chat(self, query: str) -> str:
        snippets = self.retriever(query)
        context_text = "\n\n".join([f"Source: {d.metadata.get('source', '')}\n{d.page_content}" for d in snippets])
        prompt = f"You are an assistant grounded in the following document context. Answer concisely and cite sources inline where relevant.\nContext:\n{context_text}\nQuestion: {query}"
        resp = self.llm.generate([HumanMessage(content=prompt)])
        try:
            text = resp.generations[0][0].text
        except Exception:
            text = str(resp)
        return text


class NumericalAnalysisAgent:
    def __init__(self, retriever_callable):
        self.retriever = retriever_callable
        self.llm = make_llm()

    def perform_statistical_analysis(self, numeric_series: List[float]) -> Dict[str, Any]:
        arr = np.array(numeric_series)
        res = {
            "count": int(arr.size),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if arr.size > 1 else None,
            "median": float(np.median(arr)),
            "skew": float(stats.skew(arr)) if arr.size > 2 else None,
        }
        return res

    def heavy_compute(self, query: str) -> Dict[str, Any]:
        docs = self.retriever(query)
        import re
        nums = []
        for d in docs:
            nums += [float(x.replace(',', '')) for x in re.findall(r"\d+[\d,]*\.?\d*", d.page_content)[:500]]
        if len(nums) < 5:
            return {"error": "Not enough numeric data found", "nums_found": len(nums)}
        x = np.arange(len(nums))
        y = np.array(nums)
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        return {"slope": float(slope), "intercept": float(intercept), "r2": float(r_value**2)}


class RiskAssessmentAgent:
    def __init__(self, retriever_callable):
        self.retriever = retriever_callable
        self.llm = make_llm()

    def assess_risk(self, query: str) -> Dict[str, Any]:
        docs = self.retriever(query)
        text = "\n\n".join([d.page_content for d in docs])
        if "debt" in text.lower() or "liabil" in text.lower():
            if "unusual" in text.lower() or "material" in text.lower():
                return {"flag": True, "reason": "Detected unusual debt language", "details": text[:1200]}
        return {"flag": False, "reason": "No immediate red flags detected"}


class Orchestrator:
    def __init__(self):
        self.coordinator = Coordinator()

    def ingest_and_index(self, adls_path: str, doc_id: Optional[str] = None) -> Dict[str, Any]:
        doc_id = doc_id or str(uuid.uuid4())
        text, method = parse_pdf_with_fallback(adls_path)
        docs = chunk_text_to_docs(text)
        remote_prefix = build_faiss_for_doc(doc_id, docs)
        self.coordinator.register_document_context(doc_id, {"adls_path": adls_path, "parser": method, "index_prefix": remote_prefix})
        return {"doc_id": doc_id, "index_prefix": remote_prefix, "parser": method}

    def analyze_document_multitask(self, doc_id: str, question: str) -> Dict[str, Any]:
        ctx = self.coordinator.context_store.get("docs", {}).get(doc_id)
        if not ctx:
            raise ValueError("doc_id not found in context")
        retriever = make_rag_retriever_for_remote_index(ctx["index_prefix"])

        lang_agent = LanguageAnalysisAgent(retriever)
        num_agent = NumericalAnalysisAgent(retriever)
        risk_agent = RiskAssessmentAgent(retriever)

        results = {}
        with ThreadPoolExecutor(max_workers=3) as ex:
            futures = {
                ex.submit(lang_agent.extract_knowledge, question): "language",
                ex.submit(num_agent.heavy_compute, question): "numeric",
                ex.submit(risk_agent.assess_risk, question): "risk",
            }
            for fut in as_completed(futures):
                key = futures[fut]
                try:
                    results[key] = fut.result()
                except Exception as e:
                    results[key] = {"error": str(e), "trace": traceback.format_exc()}

        risk_res = results.get("risk", {})
        if risk_res.get("flag"):
            approval_id = str(uuid.uuid4())
            pending = {"approval_id": approval_id, "doc_id": doc_id, "reason": risk_res.get("reason"), "details": risk_res.get("details"), "question": question}
            self.coordinator.context_store.setdefault("pending_approvals", {})[approval_id] = pending
            self.coordinator._save_context()
            results["human_approval_required"] = pending

        self.coordinator.add_agent_result(doc_id, "multi_analysis", {"question": question, "results": results, "timestamp": time.time()})
        return results

    def resume_after_approval(self, approval_id: str, approved: bool, approver_note: Optional[str] = None):
        pending = self.coordinator.context_store.get("pending_approvals", {}).get(approval_id)
        if not pending:
            return {"error": "approval id not found"}
        if not approved:
            pending["status"] = "rejected"
            pending["approver_note"] = approver_note
            self.coordinator._save_context()
            return {"status": "rejected"}
        doc_id = pending["doc_id"]
        retriever = make_rag_retriever_for_remote_index(self.coordinator.context_store["docs"][doc_id]["index_prefix"])
        risk_agent = RiskAssessmentAgent(retriever)
        summary = risk_agent.assess_risk(pending["question"])
        pending["status"] = "approved"
        pending["approver_note"] = approver_note
        pending["result_after_approval"] = summary
        self.coordinator._save_context()
        return {"status": "approved", "summary": summary}


# -----------------------------
# Streamlit UI (HILP + chat)
# -----------------------------

st.set_page_config(layout="wide", page_title="Agentic AI - Azure ADLS (HILP)")
orch = Orchestrator()

st.title("Agentic Multi-Agent AI — Azure ADLS + HILP (DefaultAzureCredential)")

with st.sidebar:
    st.header("Ingest / Management")
    adls_input = st.text_input("ADLS path to PDF (e.g. apple/10q.pdf)")
    if st.button("Ingest Document"):
        if not adls_input:
            st.error("Provide ADLS path")
        else:
            with st.spinner("Parsing & indexing... this may take a minute"):
                try:
                    res = orch.ingest_and_index(adls_input)
                    st.success(f"Ingested as {res['doc_id']} (parser={res['parser']})")
                except Exception as e:
                    st.error(f"Ingest failed: {e}")

    st.markdown("---")
    st.header("Pending Approvals")
    pending = orch.coordinator.context_store.get("pending_approvals", {})
    for pid, p in pending.items():
        with st.expander(f"Approval {pid} — doc {p['doc_id']}"):
            st.write(p.get("reason"))
            st.write(p.get("details")[:1000])
            col1, col2 = st.columns(2)
            if col1.button(f"Approve {pid}"):
                note = st.text_input(f"Approver note for {pid}")
                res = orch.resume_after_approval(pid, approved=True, approver_note=note)
                st.success("Approved")
            if col2.button(f"Reject {pid}"):
                note = st.text_input(f"Rejection note for {pid}")
                res = orch.resume_after_approval(pid, approved=False, approver_note=note)
                st.warning("Rejected")

# Main workspace
st.header("Multi-turn Conversation / Analysis")
col1, col2 = st.columns([2, 1])
with col1:
    question = st.text_area("Question or command (multi-turn)")
    doc_id_input = st.text_input("Context doc_id (optional)")
    if st.button("Run Analysis"):
        if not question:
            st.error("Enter a question")
        else:
            with st.spinner("Coordinating agents..."):
                try:
                    if doc_id_input:
                        res = orch.analyze_document_multitask(doc_id_input, question)
                    else:
                        st.info("Please provide a doc_id for robust RAG answers")
                        res = {"error": "doc_id missing"}
                    st.session_state["last_result"] = res
                    st.json(res)
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
with col2:
    st.subheader("Coordinator / Context")
    st.write("Registered documents:")
    st.json(orch.coordinator.context_store.get("docs", {}))
    st.write("Recent results:")
    st.json(orch.coordinator.context_store.get("results", {}))

st.markdown("---")
st.info("This demo app uses DefaultAzureCredential for ADLS access. Ensure your compute has appropriate RBAC permissions (Storage Blob Data Reader / Contributor) for the ADLS filesystem. For production, add authentication, logging, and error handling as appropriate.")
