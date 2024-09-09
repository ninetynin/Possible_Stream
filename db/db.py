import sqlite3
import pandas as pd

def setup_database():
    conn = sqlite3.connect('db/analysis_results.db')
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            age INTEGER,
            gender TEXT,
            emotion TEXT,
            face_confidence REAL
        )
    ''')

    conn.commit()
    conn.close()

def extract_data():
    conn = sqlite3.connect('db/analysis_results.db')
    query = 'SELECT timestamp, age, gender, emotion, face_confidence FROM analysis'
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

if __name__ == '__main__':
    setup_database()
    data = extract_data()
    print(data.head())
