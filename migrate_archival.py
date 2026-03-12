import sqlite3
import os

DB_PATH = "d:/projects/Gen-AI/Nexus Learner/nexus_v3.db"

def migrate():
    if not os.path.exists(DB_PATH):
        print(f"Database {DB_PATH} not found. Skipping migration.")
        return
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Check if column exists
    cursor.execute("PRAGMA table_info(subjects)")
    columns = [col[1] for col in cursor.fetchall()]
    
    if "is_archived" not in columns:
        print("Adding is_archived column to subjects table...")
        cursor.execute("ALTER TABLE subjects ADD COLUMN is_archived BOOLEAN DEFAULT 0")
        conn.commit()
        print("Migration complete.")
    else:
        print("Column is_archived already exists.")
        
    conn.close()

if __name__ == "__main__":
    migrate()
