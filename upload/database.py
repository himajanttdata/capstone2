import sqlite3


DB_FILE = "user_data.db"
DB_FILE2 = "querydb1.db"
 
def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_name TEXT,
            email TEXT,
            password TEXT
        )
    """)
    conn.commit()
    conn.close()
 
def init_db2():
    conn = sqlite3.connect(DB_FILE2)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS querydb (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            query TEXT,
            lm_response TEXT,
            category TEXT,
            sentiment TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP      
        )
    """)
    conn.commit()
    conn.close()

init_db()
init_db2()

def log_user_data(user_name: str, email: str, password: str):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO user_data (user_name, email, password)
        VALUES (?, ?, ?)
    """, (user_name, email, password))
    conn.commit()
    conn.close()

def log_query(user_id: str, query: str, lm_response: str, category: str, sentiment: str):
    conn = sqlite3.connect(DB_FILE2)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO querydb (user_id, query, lm_response, category, sentiment)
        VALUES (?, ?, ?, ?)
    """, (user_id, query, lm_response, category, sentiment))
    conn.commit()
    conn.close()

def get_user_history(user_id, limit=5):
   conn = sqlite3.connect(DB_FILE2)
   cursor = conn.cursor()
   cursor.execute(
       "SELECT query, lm_response FROM querydb WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?",
       (user_id, limit)
   )
   history = cursor.fetchall()
   conn.close()
   messages = []
   for query, response in reversed(history):
       messages.append({"role": "user", "content": query})
       messages.append({"role": "assistant", "content": response})
   return messages

def get_user_id(email):
   conn = sqlite3.connect(DB_FILE)
   cursor = conn.cursor()
   cursor.execute("SELECT id FROM user_data WHERE email = ?", (email,))
   user = cursor.fetchone()
   conn.close()
   return user[0] if user else None


def check_db_for_query(query):
    print("Entering check db")
    conn = sqlite3.connect(DB_FILE2)
    cursor = conn.cursor()
    cursor.execute("SELECT lm_response FROM querydb WHERE query = ?", (query,))
    row = cursor.fetchone()
    conn.close()
    result = row[0] if row else None
    
    if result:
        print("Exiting check db")
        return result
    print("Exiting check db")
    return False

