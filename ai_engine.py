import asyncio
import base64
import os
import shutil
from typing import List, Optional

from langchain.messages import HumanMessage, SystemMessage
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- CONFIG ---
BASE_STORAGE_PATH = os.getenv("STORAGE_PATH", ".")

UPLOAD_DIR = os.path.join(BASE_STORAGE_PATH, "uploaded_evidence")
CHROMA_DB_DIR = os.path.join(BASE_STORAGE_PATH, "chroma_db")
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
    """Extracts text from PDF or describes Image via GPT-5-mini."""
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


async def ingest_new_file(trial_id: int, file_path: str, file_type: str) -> str:
    """
    1. Reads the file (PDF text or Image description).
    2. Saves it to the Vector Database immediately.
    3. Returns the text content (so we can use it in a prompt if needed).
    """
    vector_store = get_trial_vector_store(trial_id)
    filename = os.path.basename(file_path)

    # A. Extract Text based on file type
    text_content = ""
    if "pdf" in file_type.lower():
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()
        text_content = "\n".join([p.page_content for p in pages])

        # Add Metadata and Ingest
        for page in pages:
            page.metadata["source"] = filename
            page.metadata["trial_id"] = trial_id

        # Add to DB
        if pages:
            await vector_store.aadd_documents(pages)

    elif "image" in file_type.lower():
        # Generate description via GPT-4o
        text_content = await process_file_to_text(
            file_path, file_type
        )  # (Reusing your helper)

        # Create Document
        doc = Document(
            page_content=text_content,
            metadata={"source": filename, "trial_id": trial_id, "type": "image"},
        )
        # Add to DB
        await vector_store.aadd_documents([doc])

    print(f"✅ Ingested file: {filename} for Trial {trial_id}")
    return text_content


# =============================================================================
# 2. GENERATION LOGIC
# =============================================================================


async def retrieve_context_and_sources(trial_id: int, query: str):
    """
    Performs the Vector Search.
    Returns:
        - context_str: String formatted for the LLM
        - source_list: List of filenames found
    """
    vector_store = get_trial_vector_store(trial_id)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    # Async Search
    docs = await retriever.ainvoke(query)

    context_str = ""
    source_list = []

    for doc in docs:
        source_name = doc.metadata.get("source", "Unknown")
        # Avoid duplicates in source list
        if source_name not in source_list:
            source_list.append(source_name)

        context_str += f"--- [Source: {source_name}] ---\n{doc.page_content}\n\n"

    return context_str, source_list


async def _generate_single_argument_with_rag(
    trial_id: int, case_background: str, user_argument: str
):
    """
    Helper function to run RAG for ONE argument.
    """
    # 1. Retrieve specific evidence for this specific argument
    # (e.g., if argument is about 'Time', get chunks about 'Time')
    evidence_str, sources = await retrieve_context_and_sources(trial_id, user_argument)

    # 2. Build Prompt
    system_prompt = f"""
    You are a skilled AI Prosecutor/Defense Attorney.
    
    CASE BACKGROUND:
    {case_background}
    
    SPECIFIC EVIDENCE FOUND:
    {evidence_str}
    
    USER'S ARGUMENT:
    {user_argument}
    
    INSTRUCTIONS:
    - Argue back logically based on the evidence.
    - Cite sources using [Source: filename].
    """

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_argument),
    ]

    # 3. Call LLM
    llm = ChatOpenAI(model="gpt-5-mini", temperature=0.6)
    response = await llm.ainvoke(messages)

    return {
        "content": response.content,
        "sources": sources,  # Return the list of files used
    }


async def batch_generate_initial_arguments(
    trial_id: int, case_background: str, user_arguments: List[str]
) -> List[dict]:
    """
    Runs parallel RAG generation.
    Returns a list of dicts: [{"content": "...", "sources": ["file1.pdf"]}, ...]
    """
    tasks = []
    for arg in user_arguments:
        tasks.append(_generate_single_argument_with_rag(trial_id, case_background, arg))

    # Run all RAG searches and Generations in parallel
    results = await asyncio.gather(*tasks)
    return results


async def generate_reply_with_new_evidence(
    trial_id: int,
    case_background: str,
    user_text: str,
    chat_history: list,
    new_file_path: Optional[str] = None,
    new_file_type: Optional[str] = None,
    history_limit: int = 4,
) -> str:
    """
    SINGLE REPLY: Handles new evidence upload + RAG retrieval + Response generation.
    """
    # --- STEP 1: Process Immediate New Evidence (If any) ---
    new_evidence_text = ""
    if new_file_path:
        # Convert PDF or Image to text immediately so AI sees it NOW
        # (Even if it takes a moment to index into the DB later)
        raw_text = await process_file_to_text(new_file_path, new_file_type)
        new_evidence_text = (
            f"\n--- NEWLY UPLOADED EVIDENCE (Not yet in DB) ---\n{raw_text}\n"
        )

    # --- STEP 2: Retrieve Historical Evidence (RAG) ---
    # We search using the User's text + a snippet of the new file to find relevant old laws/facts
    search_query = f"{user_text} {new_evidence_text[:200]}"
    old_evidence_str, sources = await retrieve_context_and_sources(
        trial_id, search_query
    )

    # If we have a new file, add it to the sources list manually
    # (Because it wasn't in the DB during the search step above)
    if new_file_path:
        sources.append(os.path.basename(new_file_path))

    # --- STEP 3: Construct the System Prompt ---
    system_prompt = f"""
    You are a ruthless AI Lawyer.
    
    CASE BACKGROUND:
    {case_background}
    
    ARCHIVED EVIDENCE (From Database):
    {old_evidence_str}
    
    {new_evidence_text}
    
    INSTRUCTIONS:
    1. Reply to the user's latest argument: "{user_text}"
    2. If 'NEWLY UPLOADED EVIDENCE' is provided, analyze it immediately.
    3. Check if the new evidence contradicts the 'ARCHIVED EVIDENCE'.
    4. You MUST cite your sources using brackets, e.g., [Source: police_report.pdf].
    """

    messages = [SystemMessage(content=system_prompt)]

    # --- STEP 4: Handle Chat History (With Limit) ---
    # Apply the user's requested history limit
    relevant_history = chat_history[-history_limit:] if history_limit > 0 else []

    for msg in relevant_history:
        role = HumanMessage if msg.sender == "user" else SystemMessage
        messages.append(role(content=msg.content))

    # Append the current user input
    messages.append(HumanMessage(content=user_text))

    # --- STEP 5: Generate Response ---
    llm = ChatOpenAI(model="gpt-5-mini", temperature=0.6)
    response = await llm.ainvoke(messages)

    # --- STEP 6: Background Ingest (Save new file to DB) ---
    # We do this AFTER generating (or async) so the immediate reply is fast,
    # but the file is saved for the NEXT turn.
    if new_file_path and new_evidence_text:
        vector_store = get_trial_vector_store(trial_id)

        doc = Document(
            page_content=new_evidence_text,
            metadata={
                "source": os.path.basename(new_file_path),
                "trial_id": trial_id,
                "type": "new_upload",
            },
        )
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents([doc])

        await vector_store.aadd_documents(chunks)
        print(f"Ingested {len(chunks)} chunks from new upload into Trial {trial_id}")

    # --- STEP 7: Return Content + Sources ---
    return {"content": response.content, "sources": sources}


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
