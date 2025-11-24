import base64
import os
import shutil
from typing import List, Optional

from langchain.messages import HumanMessage, SystemMessage
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- CONFIG ---
UPLOAD_DIR = "./uploaded_evidence"
CHROMA_DB_DIR = "./chroma_db"
embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

# =============================================================================
# 1. DATABASE & FILE UTILITIES
# =============================================================================


def get_trial_vector_store(trial_id: int):
    """Returns the isolated vector collection for a specific trial."""
    return Chroma(
        collection_name=f"trial_{trial_id}",
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_DIR,
    )


def save_file_locally(file_obj, filename: str, trial_id: int) -> str:
    """Saves uploaded file to disk."""
    trial_folder = os.path.join(UPLOAD_DIR, str(trial_id))
    os.makedirs(trial_folder, exist_ok=True)
    file_path = os.path.join(trial_folder, filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file_obj, buffer)
    return file_path


async def process_file_to_text(file_path: str, file_type: str) -> str:
    """Extracts text from PDF or describes Image via GPT-4o."""
    filename = os.path.basename(file_path)

    if "pdf" in file_type.lower():
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()
        return "\n".join([p.page_content for p in pages])

    elif "image" in file_type.lower():
        with open(file_path, "rb") as image_file:
            base64_img = base64.b64encode(image_file.read()).decode("utf-8")

        llm = ChatOpenAI(model="gpt-5-mini", temperature=0.0)
        msg = [
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": "Describe this evidence image in strict, factual detail for a legal trial.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"},
                    },
                ]
            )
        ]
        response = await llm.ainvoke(msg)
        return f"[Image Description of {filename}]: {response.content}"

    return ""


def ingest_text_evidence(trial_id: int, text: str, source_name: str):
    """Ingests raw text (like Case Background) into Chroma."""
    vector_store = get_trial_vector_store(trial_id)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    doc = Document(
        page_content=text, metadata={"source": source_name, "trial_id": trial_id}
    )
    vector_store.add_documents(splitter.split_documents([doc]))


# =============================================================================
# 2. GENERATION LOGIC
# =============================================================================


async def batch_generate_initial_arguments(
    trial_id: int, case_background: str, user_arguments: List[str]
) -> List[str]:
    """
    PARALLEL: Generates responses for multiple starting arguments at once using .abatch()
    """
    prompt = ChatPromptTemplate.from_template("""
    You are a skilled AI Prosecutor/Defense Attorney.
    CASE BACKGROUND: {case_background}
    USER'S ARGUMENT: {user_argument}
    INSTRUCTIONS: Argue back logically. Be specific.
    """)

    chain = prompt | ChatOpenAI(model="gpt-4o", temperature=0.6)

    batch_inputs = [
        {"case_background": case_background, "user_argument": arg}
        for arg in user_arguments
    ]
    results = await chain.abatch(batch_inputs)
    return [res.content for res in results]


async def generate_reply_with_new_evidence(
    trial_id: int,
    case_background: str,
    user_text: str,
    chat_history: list,
    new_file_path: Optional[str] = None,
    new_file_type: Optional[str] = None,
) -> str:
    """
    SINGLE REPLY: Handles new evidence upload + RAG retrieval + Response generation.
    """
    # 1. Process Immediate New Evidence
    new_evidence_text = ""
    if new_file_path:
        raw_text = await process_file_to_text(new_file_path, new_file_type)
        new_evidence_text = f"\n--- NEWLY UPLOADED EVIDENCE ---\n{raw_text}\n"

    # 2. Retrieve Old Evidence
    vector_store = get_trial_vector_store(trial_id)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    search_query = f"{user_text} {new_evidence_text[:200]}"
    old_docs = await retriever.ainvoke(search_query)

    old_evidence_str = "\n".join(
        [
            f"[Archived Source: {d.metadata.get('source')}]\n{d.page_content}"
            for d in old_docs
        ]
    )

    # 3. Generate
    system_prompt = f"""
    You are a ruthless AI Lawyer.
    CASE BACKGROUND: {case_background}
    ARCHIVED EVIDENCE: {old_evidence_str}
    {new_evidence_text}
    INSTRUCTIONS: Reply to user. Check if new evidence contradicts old evidence. Cite sources.
    """

    messages = [SystemMessage(content=system_prompt)]
    for msg in chat_history[-4:]:
        role = HumanMessage if msg.sender == "user" else SystemMessage
        messages.append(role(content=msg.content))
    messages.append(HumanMessage(content=user_text))

    llm = ChatOpenAI(model="gpt-4o", temperature=0.6)
    response = await llm.ainvoke(messages)

    # 4. Background Ingest
    if new_file_path and new_evidence_text:
        doc = Document(
            page_content=new_evidence_text,
            metadata={"source": os.path.basename(new_file_path), "trial_id": trial_id},
        )
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        await vector_store.aadd_documents(splitter.split_documents([doc]))

    return response.content


# =============================================================================
# 3. CLEANUP & UPDATES
# =============================================================================


def delete_trial_data(trial_id: int):
    """Deletes vectors and files for a trial."""
    try:
        get_trial_vector_store(trial_id).delete_collection()
        path = os.path.join(UPLOAD_DIR, str(trial_id))
        if os.path.exists(path):
            shutil.rmtree(path)
    except Exception as e:
        print(f"Cleanup Error: {e}")


def delete_specific_file(trial_id: int, filename: str):
    get_trial_vector_store(trial_id).delete(where={"source": filename})
    path = os.path.join(UPLOAD_DIR, str(trial_id), filename)
    if os.path.exists(path):
        os.remove(path)


def update_case_background_vectors(trial_id: int, new_text: str):
    store = get_trial_vector_store(trial_id)
    store.delete(where={"source": "Case Background"})
    ingest_text_evidence(trial_id, new_text, "Case Background")
