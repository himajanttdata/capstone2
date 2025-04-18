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
            query TEXT,
            lm_response TEXT,
            category TEXT,
            sentiment TEXT
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

def log_query(query: str, lm_response: str, category: str, sentiment: str):
    conn = sqlite3.connect(DB_FILE2)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO querydb (query, lm_response, category, sentiment)
        VALUES (?, ?, ?, ?)
    """, (query, lm_response, category, sentiment))
    conn.commit()
    conn.close()


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

