import json

from sqlalchemy import Column, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from database import Base


class Trial(Base):
    __tablename__ = "trials"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, index=True)  # Placeholder for auth
    title = Column(String)
    case_background = Column(Text)

    # Cascade: If Trial is deleted, delete all Threads
    threads = relationship(
        "Thread", back_populates="trial", cascade="all, delete-orphan"
    )


class Thread(Base):
    __tablename__ = "threads"
    id = Column(Integer, primary_key=True, index=True)
    trial_id = Column(Integer, ForeignKey("trials.id"))
    title = Column(String)

    trial = relationship("Trial", back_populates="threads")
    # Cascade: If Thread is deleted, delete all Messages
    messages = relationship(
        "Message", back_populates="thread", cascade="all, delete-orphan"
    )


class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    thread_id = Column(Integer, ForeignKey("threads.id"))
    sender = Column(String)  # "user" or "ai"
    content = Column(Text)
    sources = Column(Text, default="[]")

    thread = relationship("Thread", back_populates="messages")


def get_sources_list(self):
    return json.loads(self.sources) if self.sources else []
