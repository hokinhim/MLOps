import os
from airflow.models import Variable


# Секреты из .env
ENV_KEYS = [
    "S3_ACCESS_KEY_ID",
    "S3_REGION",
    "S3_ENDPOINT_URL",
    "S3_SECRET_ACCESS_KEY",
    "MLFLOW_BUCKET",
    "MLFLOW_TRACKING_URL",
]


def main():
    # Установка AWS-переменных из .env
    for key in ENV_KEYS:
        value = os.getenv(key)
        exists = Variable.get(key, default_var=None) is not None
        if exists:
            continue
        Variable.set(key, value)

    print("[+] Переменные Airflow заданы успешно!")


if __name__ == "__main__":
    main()
