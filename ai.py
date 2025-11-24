import base64
import os
import shutil
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from chromadb import PersistentClient
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

UPLOAD_DIR = "./uploaded_evidence"
CHROMA_DB_DIR = "./chroma_db"

embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))


def get_trial_vector_store(trial_id: int) -> Chroma:
    """Return a Chroma vector store scoped to a trial collection."""
    os.makedirs(CHROMA_DB_DIR, exist_ok=True)
    return Chroma(
        collection_name=f"trial_{trial_id}",
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_DIR,
    )


def _persist(vector_store: Chroma) -> None:
    """Persist vector store changes if supported."""
    try:
        vector_store.persist()
    except Exception:
        # Some LangChain/Chroma combos flush automatically; ignore if not supported.
        pass


def delete_trial_data(trial_id: int) -> bool:
    """Drop the trial collection and remove uploaded files."""
    try:
        client = PersistentClient(path=CHROMA_DB_DIR)
        coll_name = f"trial_{trial_id}"
        try:
            client.delete_collection(coll_name)
        except Exception:
            # Collection may not exist yet; ignore.
            pass

        trial_folder = os.path.join(UPLOAD_DIR, str(trial_id))
        if os.path.exists(trial_folder):
            shutil.rmtree(trial_folder)

        return True
    except Exception as e:
        print(f"Error deleting trial data: {e}")
        return False


def save_file_locally(file_obj, filename: str, trial_id: int) -> str:
    """Persist an uploaded file so it can be ingested."""
    trial_folder = os.path.join(UPLOAD_DIR, str(trial_id))
    os.makedirs(trial_folder, exist_ok=True)

    file_path = os.path.join(trial_folder, filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file_obj, buffer)

    return file_path


def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def generate_image_description(image_path: str) -> str:
    """Use GPT-4o vision to generate a factual description of an image."""
    base64_image = encode_image_to_base64(image_path)
    llm = ChatOpenAI(model="gpt-4o", temperature=0.0)

    msg = [
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": (
                        "Describe this evidence image in strict, factual detail. "
                        "Mention every object, text, injury, or condition visible. "
                        "Do not hallucinate details not present."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ]
        )
    ]

    response = llm.invoke(msg)
    return response.content


def _chunk_documents(
    docs: Sequence[Document], chunk_size: int = 800, chunk_overlap: int = 120
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "],
    )
    chunked: List[Document] = []
    for doc in docs:
        splits = splitter.split_documents([doc])
        for split in splits:
            # Preserve upstream metadata (e.g., source, page)
            split.metadata.update(doc.metadata)
        chunked.extend(splits)
    return chunked


def ingest_file(trial_id: int, file_path: str, file_type: str) -> None:
    """Ingest a PDF or image file into the trial vector store."""
    vector_store = get_trial_vector_store(trial_id)
    filename = os.path.basename(file_path)

    docs_to_add: List[Document] = []

    if "pdf" in file_type:
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        for page in pages:
            page.metadata.update(
                {
                    "source": filename,
                    "trial_id": trial_id,
                    "type": "document",
                    "page": page.metadata.get("page", None),
                }
            )
        docs_to_add.extend(_chunk_documents(pages))

    elif "image" in file_type:
        description = generate_image_description(file_path)
        doc = Document(
            page_content=description,
            metadata={
                "source": filename,
                "trial_id": trial_id,
                "type": "image_description",
                "original_path": file_path,
            },
        )
        docs_to_add.append(doc)

    if docs_to_add:
        vector_store.add_documents(docs_to_add)
        _persist(vector_store)
        print(f"Ingested {filename} into Trial {trial_id}")


def ingest_text_evidence(
    trial_id: int, text: str, source_name: str = "General Context"
) -> None:
    """Ingest raw text (case background or typed notes)."""
    vector_store = get_trial_vector_store(trial_id)
    base_doc = Document(
        page_content=text,
        metadata={"source": source_name, "trial_id": trial_id, "type": "text"},
    )
    chunks = _chunk_documents([base_doc])
    vector_store.add_documents(chunks)
    _persist(vector_store)


@dataclass
class ChatTurn:
    sender: str  # "user" or "ai"
    content: str


def generate_counter_argument(
    trial_id: int,
    case_background: str,
    user_argument: str,
    chat_history: Sequence[ChatTurn],
    *,
    top_k: int = 4,
) -> str:
    """Generate a counter-argument grounded in retrieved evidence."""
    vector_store = get_trial_vector_store(trial_id)
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    relevant_docs: List[Document] = retriever.invoke(user_argument)

    context_str = ""
    for doc in relevant_docs:
        source = doc.metadata.get("source", "Unknown Source")
        page = doc.metadata.get("page")
        page_tag = f" (page {page})" if page is not None else ""
        context_str += f"--- [Source: {source}{page_tag}] ---\n{doc.page_content}\n\n"

    system_prompt = f"""
    You are a skilled AI prosecutor/defense attorney in a mock trial.

    CASE BACKGROUND:
    {case_background}

    EVIDENCE RETRIEVED FROM DATABASE:
    {context_str}

    INSTRUCTIONS:
    1) Rely only on the provided evidence and background.
    2) Construct a sharp counter-argument with explicit citations in brackets, e.g., [Source: file.pdf (page 2)].
    3) If the user's claim lacks evidence support, say so and request specific exhibits.
    4) Keep the tone professional, adversarial, and concise.
    """

    messages: List = [SystemMessage(content=system_prompt)]

    for turn in list(chat_history)[-4:]:
        if turn.sender == "user":
            messages.append(HumanMessage(content=turn.content))
        else:
            messages.append(AIMessage(content=turn.content))

    messages.append(HumanMessage(content=user_argument))

    llm = ChatOpenAI(model="gpt-4o", temperature=0.4)
    response = llm.invoke(messages)
    return response.content

