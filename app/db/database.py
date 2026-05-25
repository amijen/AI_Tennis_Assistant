from sqlalchemy import create_engine 
from sqlalchemy.orm import sessionmaker
import os
import dotenv
dotenv.load_dotenv()

DATABASE_URL = os.getenv("DB_URL")
if not DATABASE_URL :
    raise ValueError("DB_URL environment variable is not set.")
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)