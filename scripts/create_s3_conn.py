import os
import json
from airflow.models import Connection
from airflow import settings
from sqlalchemy.orm import Session


def main():
    # Сбор секретов из .env
    aws_access_key_id = os.getenv("S3_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("S3_SECRET_ACCESS_KEY")
    region = os.getenv("S3_REGION")
    endpoint_url = os.getenv("S3_ENDPOINT_URL")

    # Формирование extra
    extra = {
        "region_name": region,
        "endpoint_url": endpoint_url,
    }

    # Создание объекта Connection
    conn = Connection(
        conn_id="s3_connection",
        conn_type="aws",
        login=aws_access_key_id,
        password=aws_secret_access_key,
        extra=json.dumps(extra),
    )

    # Запись в metadata DB Airflow
    session = settings.Session()
    try:
        exists = session.query(Connection).filter(Connection.conn_id == conn.conn_id).first()
        if not exists:
            session.add(conn)
            session.commit()
            print(f"[+] Connection '{conn.conn_id}' успешно дбавлено!")
        else:
            print(f"[!] Connection '{conn.conn_id}' уже существует.")
    finally:
        session.close()


if __name__ == "__main__":
    main()
