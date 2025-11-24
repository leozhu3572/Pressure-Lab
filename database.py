import os

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# 1. Get the DB URL from Environment Variable (Cloud) OR use Local SQLite
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./pressure_lab.db")

# 2. Handle the "check_same_thread" rule (Only needed for SQLite)
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    # PostgreSQL (Cloud) doesn't need that argument
    engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
