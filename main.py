import json
from typing import List

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

import ai_engine
import database
import models

# 1. Setup Database Tables
models.Base.metadata.create_all(bind=database.engine)

# 2. Initialize App
app = FastAPI(title="Pressure Lab API", description="Backend for AI Mock Trials")

# 3. CORS (Allow Frontend to connect)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, change this to your Vercel URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Schemas (Input Validation) ---


class TrialCreate(BaseModel):
    title: str
    case_background: str
    initial_arguments: List[str] = []  # Optional batch arguments


class MessageEdit(BaseModel):
    new_content: str


class CaseUpdate(BaseModel):
    case_background: str


# =============================================================================
# 1. TRIAL MANAGEMENT API
# =============================================================================


@app.post("/trials/")
async def create_trial(trial: TrialCreate, db: Session = Depends(database.get_db)):
    """
    Creates a new trial.
    If 'initial_arguments' are provided, it runs a BATCH AI generation
    to create multiple threads instantly.
    """
    # A. Create SQL Record
    db_trial = models.Trial(
        title=trial.title,
        case_background=trial.case_background,
        user_id=1,  # Hardcoded for prototype
    )
    db.add(db_trial)
    db.commit()
    db.refresh(db_trial)

    # B. Ingest Background to AI (The "Brain" learns the case)
    # We do this synchronously as it's fast text
    ai_engine.ingest_text_evidence(
        db_trial.id, trial.case_background, "Case Background"
    )

    # C. Run Parallel Generation for Initial Arguments (The "Action")
    if trial.initial_arguments:
        # returns list of {"content": "...", "sources": [...]}
        ai_results = await ai_engine.batch_generate_initial_arguments(
            trial_id=db_trial.id,
            case_background=trial.case_background,
            user_arguments=trial.initial_arguments,
        )

        for user_arg, ai_res in zip(trial.initial_arguments, ai_results):
            # Create Thread
            thread = models.Thread(trial_id=db_trial.id, title=user_arg[:50])
            db.add(thread)
            db.commit()
            db.refresh(thread)

            # User Message
            db.add(models.Message(thread_id=thread.id, sender="user", content=user_arg))

            # AI Message (WITH SOURCES)
            db.add(
                models.Message(
                    thread_id=thread.id,
                    sender="ai",
                    content=ai_res["content"],
                    sources=json.dumps(ai_res["sources"]),  # Save list as JSON string
                )
            )

        db.commit()

    return {
        "trial_id": db_trial.id,
        "status": "created",
        "threads_generated": len(trial.initial_arguments),
    }


@app.get("/trials/")
def get_trials(db: Session = Depends(database.get_db)):
    """List all trials for the user."""
    return db.query(models.Trial).all()


@app.get("/trials/{trial_id}")
def get_trial_details(trial_id: int, db: Session = Depends(database.get_db)):
    """Get full dashboard: Case info + All threads."""
    trial = db.query(models.Trial).filter(models.Trial.id == trial_id).first()
    if not trial:
        raise HTTPException(404, detail="Trial not found")

    # SQLAlchemy relationship loads threads automatically
    return trial


@app.delete("/trials/{trial_id}")
def delete_trial(trial_id: int, db: Session = Depends(database.get_db)):
    """
    Deletes a trial.
    TRIGGERS:
    1. SQL Cascade Delete (Removes Threads/Messages)
    2. AI Engine Cleanup (Removes Vector Collection + Files)
    """
    trial = db.query(models.Trial).filter(models.Trial.id == trial_id).first()
    if not trial:
        raise HTTPException(404, detail="Trial not found")

    # 1. Clean AI Data
    ai_engine.delete_trial_data(trial_id)

    # 2. Delete from DB
    db.delete(trial)
    db.commit()
    return {"status": "deleted"}


@app.put("/trials/{trial_id}")
def update_trial_info(
    trial_id: int, payload: CaseUpdate, db: Session = Depends(database.get_db)
):
    """Updates case background and re-indexes it in the AI Brain."""
    trial = db.query(models.Trial).filter(models.Trial.id == trial_id).first()
    if not trial:
        raise HTTPException(404, detail="Not Found")

    # Update SQL
    trial.case_background = payload.case_background
    db.commit()

    # Update AI Vectors
    ai_engine.update_case_background_vectors(trial_id, payload.case_background)
    return {"status": "updated"}


# =============================================================================
# 2. THREAD & CONVERSATION API
# =============================================================================


@app.post("/trials/{trial_id}/threads/{thread_id}/reply")
async def reply_to_thread(
    trial_id: int,
    thread_id: int,
    content: str = Form(...),  # Handles Text
    history_limit: int = Form(4),
    file: UploadFile = File(None),  # Handles Optional PDF/Image
    db: Session = Depends(database.get_db),
):
    """
    The Main Loop: User replies (with optional file) -> AI Responds.
    """
    # A. Save User Message
    msg_content_display = content + (f" [Attached: {file.filename}]" if file else "")
    user_msg = models.Message(
        thread_id=thread_id, sender="user", content=msg_content_display
    )
    db.add(user_msg)

    # B. Handle File Upload (Save to disk)
    file_path, file_type = None, None
    if file:
        file_path = ai_engine.save_file_locally(file.file, file.filename, trial_id)
        file_type = file.content_type

    # C. Get Context (History + Background)
    trial = db.query(models.Trial).filter(models.Trial.id == trial_id).first()
    history = (
        db.query(models.Message).filter(models.Message.thread_id == thread_id).all()
    )

    # D. GENERATE AI RESPONSE (Async)
    ai_result = await ai_engine.generate_reply_with_new_evidence(
        trial_id=trial_id,
        case_background=trial.case_background,
        user_text=content,
        chat_history=history,
        new_file_path=file_path,
        new_file_type=file_type,
        history_limit=history_limit,
    )

    # E. Save AI Response
    db.add(
        models.Message(
            thread_id=thread_id,
            sender="ai",
            content=ai_result["content"],
            sources=json.dumps(ai_result["sources"]),  # Save sources
        )
    )
    db.commit()

    return {
        "user_content": content,
        "ai_response": ai_result["content"],
        "sources": ai_result["sources"],  # Return list to frontend
    }


@app.delete("/threads/{thread_id}")
def delete_thread(thread_id: int, db: Session = Depends(database.get_db)):
    """Deletes a single argument thread."""
    thread = db.query(models.Thread).filter(models.Thread.id == thread_id).first()
    if not thread:
        raise HTTPException(404, detail="Thread not found")

    db.delete(thread)
    db.commit()
    return {"status": "thread deleted"}


# =============================================================================
# 3. EDIT & REWIND API
# =============================================================================


@app.put("/messages/{message_id}")
async def edit_message_and_regenerate(
    message_id: int, payload: MessageEdit, db: Session = Depends(database.get_db)
):
    """
    Edits a user message and 'Rewinds Time':
    1. Deletes all messages that came AFTER the edited one.
    2. Updates the message content.
    3. Regenerates the AI response based on the new content.
    """
    target_msg = (
        db.query(models.Message).filter(models.Message.id == message_id).first()
    )
    if not target_msg:
        raise HTTPException(404, detail="Message not found")
    if target_msg.sender != "user":
        raise HTTPException(400, detail="Can only edit user messages")

    # A. Prune Future Messages
    db.query(models.Message).filter(
        models.Message.thread_id == target_msg.thread_id, models.Message.id > message_id
    ).delete()

    # B. Update Content
    target_msg.content = payload.new_content
    db.commit()

    # C. Regenerate AI Response
    # Re-fetch history (now pruned)
    trial = target_msg.thread.trial
    history = (
        db.query(models.Message)
        .filter(models.Message.thread_id == target_msg.thread_id)
        .all()
    )

    # Note: history[:-1] excludes the message we just edited, so we pass it as 'user_text'
    # Actually, simpler logic: Pass history UP TO the edited message, and pass edited msg as new input

    # history now contains [Msg1, Msg2, ..., EditedMsg]
    # We want history to be [Msg1, Msg2, ...] and user_text to be EditedMsg

    past_history = history[:-1]

    ai_res_text = await ai_engine.generate_reply_with_new_evidence(
        trial_id=trial.id,
        case_background=trial.case_background,
        user_text=payload.new_content,
        chat_history=past_history,
    )

    # D. Save New AI Response
    db.add(
        models.Message(thread_id=target_msg.thread_id, sender="ai", content=ai_res_text)
    )
    db.commit()

    return {"status": "edited and regenerated", "ai_response": ai_res_text}


@app.delete("/trials/{trial_id}/files/{filename}")
def delete_file(trial_id: int, filename: str):
    """Removes a specific file from evidence (both disk and vector store)."""
    ai_engine.delete_specific_file(trial_id, filename)
    return {"status": "file removed"}
