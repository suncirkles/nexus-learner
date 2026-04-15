"""
scripts/init_supabase.py
-------------------------
Utility script to initialize the Supabase PostgreSQL database schema.
Requires DATABASE_URL to be set in .env.

Usage:
    python scripts/init_supabase.py
"""

import sys
import os

# Add the project root to sys.path so we can import core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.database import engine, Base, logger

def init_db():
    print(f"Initializing database at: {engine.url}")
    try:
        # Enable pgvector extension if not present
        from sqlalchemy import text
        with engine.connect() as conn:
            print("Checking/Enabling 'vector' extension...")
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            conn.commit()
            
        # This will create all tables defined in core.database
        Base.metadata.create_all(bind=engine)
        print("[OK] Database schema initialized successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to initialize database: {e}")
        sys.exit(1)

if __name__ == "__main__":
    init_db()
