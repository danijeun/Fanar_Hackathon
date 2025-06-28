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
    """Initializes the database and creates the 'conversations' and 'email_bodies' tables if they don't exist."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            chat_id INTEGER PRIMARY KEY,
            history TEXT NOT NULL
        )
    """)
    # New table for email bodies
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS email_bodies (
            chat_id INTEGER,
            message_id TEXT,
            body TEXT NOT NULL,
            PRIMARY KEY (chat_id, message_id)
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

def save_email_body(chat_id: int, message_id: str, body: str) -> None:
    """Saves the email body for a given chat_id and message_id."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT OR REPLACE INTO email_bodies (chat_id, message_id, body) VALUES (?, ?, ?)",
        (chat_id, message_id, body)
    )
    conn.commit()

def get_email_body(chat_id: int, message_id: str) -> str | None:
    """Retrieves the email body for a given chat_id and message_id."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT body FROM email_bodies WHERE chat_id = ? AND message_id = ?",
        (chat_id, message_id)
    )
    row = cursor.fetchone()
    if row:
        return row[0]
    return None

def ensure_short_id_table():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "CREATE TABLE IF NOT EXISTS short_id_mappings (short_id TEXT PRIMARY KEY, mapping_type TEXT, value TEXT)"
    )
    conn.commit()

def save_short_id_mapping(short_id: str, value: str) -> None:
    """Saves a mapping from short_id to value (as stringified message_id)."""
    ensure_short_id_table()
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT OR REPLACE INTO short_id_mappings (short_id, mapping_type, value) VALUES (?, ?, ?)",
        (short_id, 'summarize', value)
    )
    conn.commit()

def get_short_id_mapping(short_id: str) -> str | None:
    """Retrieves the value for a given short_id (mapping_type 'summarize')."""
    ensure_short_id_table()
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT value FROM short_id_mappings WHERE short_id = ? AND mapping_type = ?",
        (short_id, 'summarize')
    )
    row = cursor.fetchone()
    if row:
        return row[0]
    return None

def delete_short_id_mapping(short_id: str) -> None:
    """Deletes a mapping for a given short_id (mapping_type 'summarize')."""
    ensure_short_id_table()
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "DELETE FROM short_id_mappings WHERE short_id = ? AND mapping_type = ?",
        (short_id, 'summarize')
    )
    conn.commit()

def save_schedule_event_mapping(schedule_id: str, event_json: str) -> None:
    ensure_short_id_table()
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT OR REPLACE INTO short_id_mappings (short_id, mapping_type, value) VALUES (?, ?, ?)",
        (schedule_id, 'schedule_event', event_json)
    )
    conn.commit()

def get_schedule_event_mapping(schedule_id: str) -> str | None:
    ensure_short_id_table()
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT value FROM short_id_mappings WHERE short_id = ? AND mapping_type = ?",
        (schedule_id, 'schedule_event')
    )
    row = cursor.fetchone()
    if row:
        return row[0]
    return None

def delete_schedule_event_mapping(schedule_id: str) -> None:
    ensure_short_id_table()
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute(
        "DELETE FROM short_id_mappings WHERE short_id = ? AND mapping_type = ?",
        (schedule_id, 'schedule_event')
    )
    conn.commit() 