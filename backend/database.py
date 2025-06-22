import sqlite3
import json
import threading

DB_PATH = "bot_memory.db"

# Use a thread-local variable for the database connection
local = threading.local()

def get_db():
    """Gets a thread-safe database connection."""
    if not hasattr(local, "conn"):
        local.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return local.conn

def init_db():
    """Initializes the database and creates the 'conversations' table if it doesn't exist."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            chat_id INTEGER PRIMARY KEY,
            history TEXT NOT NULL
        )
    """)
    conn.commit()

def get_conversation_history(chat_id: int) -> list:
    """Retrieves the conversation history for a given chat_id."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT history FROM conversations WHERE chat_id = ?", (chat_id,))
    row = cursor.fetchone()
    if row:
        return json.loads(row[0])
    return []

def update_conversation_history(chat_id: int, history: list) -> None:
    """Updates or creates the conversation history for a given chat_id."""
    conn = get_db()
    cursor = conn.cursor()
    history_json = json.dumps(history)
    # Use INSERT OR REPLACE to either create a new row or update an existing one
    cursor.execute("INSERT OR REPLACE INTO conversations (chat_id, history) VALUES (?, ?)", (chat_id, history_json))
    conn.commit()

def clear_conversation_history(chat_id: int) -> None:
    """Deletes the conversation history for a given chat_id."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM conversations WHERE chat_id = ?", (chat_id,))
    conn.commit() 