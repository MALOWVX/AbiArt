"""
AbiArt — SQLite to PostgreSQL Migration Helper
==============================================
Transfers existing users and community presets from local SQLite (abiart.db)
to a PostgreSQL database (e.g. Railway Postgres).

Usage:
    python migrate_to_postgres.py
    # Or with a custom DATABASE_URL:
    python migrate_to_postgres.py postgresql://postgres:pass@host:5432/railway
"""

import os
import sys
import sqlite3
from dotenv import load_dotenv

load_dotenv()

def migrate():
    target_url = sys.argv[1] if len(sys.argv) > 1 else os.environ.get('DATABASE_URL')
    if not target_url:
        print("[ERROR] No DATABASE_URL found in environment or arguments.")
        print("Usage: python migrate_to_postgres.py <postgresql://...>")
        return

    if target_url.startswith('postgres://'):
        target_url = target_url.replace('postgres://', 'postgresql://', 1)

    sqlite_path = os.path.join(os.path.dirname(__file__), 'abiart.db')
    if not os.path.exists(sqlite_path):
        print(f"[WARN] Local SQLite file not found at {sqlite_path}. Nothing to migrate.")
        return

    try:
        import psycopg2
        import psycopg2.extras
    except ImportError:
        print("[ERROR] psycopg2-binary is required. Run: pip install psycopg2-binary")
        return

    print(f"Connecting to SQLite: {sqlite_path}...")
    s_conn = sqlite3.connect(sqlite_path)
    s_conn.row_factory = sqlite3.Row
    s_cur = s_conn.cursor()

    print(f"Connecting to PostgreSQL...")
    try:
        p_conn = psycopg2.connect(target_url)
        p_cur = p_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    except Exception as e:
        print(f"[ERROR] Failed to connect to PostgreSQL: {e}")
        return

    # Ensure tables exist in Postgres
    p_cur.execute('''CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        username VARCHAR(100) UNIQUE NOT NULL,
        password_hash VARCHAR(255) NOT NULL,
        images_remaining INTEGER DEFAULT 3,
        is_admin BOOLEAN DEFAULT FALSE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')

    p_cur.execute('''CREATE TABLE IF NOT EXISTS community_presets (
        id SERIAL PRIMARY KEY,
        username VARCHAR(100) NOT NULL,
        title VARCHAR(200) NOT NULL,
        image_thumb TEXT NOT NULL,
        preset_json TEXT NOT NULL,
        likes INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    p_conn.commit()

    # 1. Migrate users
    s_cur.execute("SELECT id, username, password_hash, images_remaining, is_admin, created_at FROM users")
    users = s_cur.fetchall()
    migrated_users = 0
    for u in users:
        p_cur.execute("SELECT id FROM users WHERE username = %s", (u['username'],))
        if not p_cur.fetchone():
            p_cur.execute(
                """INSERT INTO users (id, username, password_hash, images_remaining, is_admin, created_at)
                   VALUES (%s, %s, %s, %s, %s, %s)
                   ON CONFLICT (id) DO UPDATE SET
                       images_remaining = EXCLUDED.images_remaining,
                       is_admin = EXCLUDED.is_admin""",
                (u['id'], u['username'], u['password_hash'], u['images_remaining'], bool(u['is_admin']), u['created_at'])
            )
            migrated_users += 1
    p_conn.commit()

    # Reset users sequence
    p_cur.execute("SELECT setval(pg_get_serial_sequence('users', 'id'), COALESCE(MAX(id), 1)) FROM users")
    p_conn.commit()
    print(f"✓ Migrated {migrated_users} user(s).")

    # 2. Migrate community presets
    s_cur.execute("SELECT id, username, title, image_thumb, preset_json, likes, created_at FROM community_presets")
    presets = s_cur.fetchall()
    migrated_presets = 0
    for p in presets:
        p_cur.execute("SELECT id FROM community_presets WHERE id = %s", (p['id'],))
        if not p_cur.fetchone():
            p_cur.execute(
                """INSERT INTO community_presets (id, username, title, image_thumb, preset_json, likes, created_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (p['id'], p['username'], p['title'], p['image_thumb'], p['preset_json'], p['likes'], p['created_at'])
            )
            migrated_presets += 1
    p_conn.commit()

    # Reset presets sequence
    if presets:
        p_cur.execute("SELECT setval(pg_get_serial_sequence('community_presets', 'id'), COALESCE(MAX(id), 1)) FROM community_presets")
        p_conn.commit()
    print(f"✓ Migrated {migrated_presets} community preset(s).")

    s_conn.close()
    p_conn.close()
    print("Migration completed successfully!")

if __name__ == '__main__':
    migrate()
