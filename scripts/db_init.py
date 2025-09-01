#!/usr/bin/env python3
import os
from pathlib import Path


def main():
    sql_path = Path(__file__).resolve().parent.parent / 'migrations' / '001_init.sql'
    sql = sql_path.read_text()
    dsn = os.getenv('PG_DSN') or (
        f"dbname={os.getenv('PG_DATABASE','zen_mcp')} user={os.getenv('PG_USER','postgres')} "
        f"password={os.getenv('PG_PASSWORD','')} host={os.getenv('PG_HOST','localhost')} port={os.getenv('PG_PORT','5432')}"
    )
    try:
        import psycopg
    except Exception as e:
        raise SystemExit(f"psycopg not installed: {e}")
    try:
        with psycopg.connect(dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
        print('database initialized')
    except Exception as e:
        raise SystemExit(f"db init failed: {e}")

if __name__ == '__main__':
    main()

