"""
UNSW-NB15 IDS pipeline:
- Мультикласс: attack_cat (Normal + типы атак)
- Модели: LogisticRegression / RandomForest / XGBoost
- Тюнинг гиперпараметров: каждый trial = отдельный MLflow run
- Отдельный эксперимент для SMOTE vs baseline
- Версионирование датасета: sha256 (raw train/test + sha256(sample))
- Регистрация лучшей модели в MLflow Model Registry
"""

import os
import io
import json
import time
import hashlib
import random
import logging
from datetime import timedelta
from typing import Dict, Any, List
import pendulum
from airflow.decorators import dag, task
from airflow.models import Variable
from airflow.hooks.base import BaseHook
from airflow.providers.amazon.aws.hooks.s3 import S3Hook

# DAG config
DAG_ID = "UNSW_tuning"

# Timezone для Airflow DAG
TZ = pendulum.timezone("Europe/Moscow")

# S3 connection
S3_CONN_ID = "s3_connection"
S3_PREFIX = "datasets/ids/unsw_nb15"
S3_REPORT_PREFIX = "reports/ids_unsw_tuning"

# Логгер для DAG
_LOG = logging.getLogger(DAG_ID)
_LOG.setLevel(logging.INFO)


# Получение Airflow Variable
def get_var(name: str, default=None):
    try:
        return Variable.get(name)
    except Exception:
        return default


# Скачивание по URL
def download_to_file(url: str, out_path: str) -> None:
    import urllib.request
    _LOG.info(f"Скачивание по URL: {url}")

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=180) as resp, open(out_path, "wb") as f:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)


# Вычисление sha256 для файла
def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# Настройка MLflow из S3 connection
def configure_mlflow_from_s3_connection() -> None:
    import mlflow

    # Получение tracking uri из env или Airflow Variable
    tracking_uri = os.getenv("MLFLOW_TRACKING_URL")
    if not tracking_uri:
        tracking_uri = get_var("MLFLOW_TRACKING_URL")
    if not tracking_uri:
        raise RuntimeError("No MLFLOW_TRACKING_URL in env or Airflow Variables")
    
    # Настройка MLflow tracking uri
    mlflow.set_tracking_uri(tracking_uri)
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    
    # Получение S3 connection и настройка переменных окружения для MLflow S3 backend
    conn = BaseHook.get_connection(S3_CONN_ID)
    extra = conn.extra_dejson or {}
    endpoint = extra.get("endpoint_url") or os.getenv("S3_ENDPOINT_URL")
    region = extra.get("region_name") or os.getenv("S3_REGION")

    # MLflow S3 backend использует boto3, который читает AWS credentials из .env
    if conn.login and conn.password:
        os.environ["AWS_ACCESS_KEY_ID"] = conn.login
        os.environ["AWS_SECRET_ACCESS_KEY"] = conn.password
    os.environ["AWS_DEFAULT_REGION"] = region

    # MLflow S3 backend config
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = endpoint
    os.environ["MLFLOW_BOTO_CLIENT_ADDRESSING_STYLE"] = "path"

    # Логи для отладки
    _LOG.info(f"MLflow tracking uri = {tracking_uri}")
    _LOG.info(f"MLflow S3 endpoint = {endpoint}, region = {region}")


# Функция для случайного сэмплирования trial'ов из grid'а
def sample_trials(grid: List[Dict[str, Any]], max_trials: int, seed: int = 42) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    g = list(grid)
    rng.shuffle(g)
    return g[:max_trials]


# Функция для построения grid'ов гиперпараметров для разных моделей, с учетом использования SMOTE
def build_param_grids(num_classes: int, use_smote: bool) -> Dict[str, List[Dict[str, Any]]]:
    from sklearn.model_selection import ParameterGrid

    grids = {}

    # Logistic Regression
    lr_grid = list(ParameterGrid({
        "clf__C": [0.05, 0.1, 0.3, 1.0, 3.0],
        "clf__solver": ["lbfgs", "saga"],
        "clf__class_weight": [None, "balanced"],
    }))

    # Random Forest
    rf_grid = list(ParameterGrid({
        "clf__n_estimators": [200, 400, 800],
        "clf__max_depth": [None, 12, 20, 35],
        "clf__min_samples_split": [2, 10],
        "clf__max_features": ["sqrt", 0.5],
        "clf__class_weight": [None, "balanced_subsample"],
    }))

    # XGBoost
    xgb_grid = list(ParameterGrid({
        "clf__n_estimators": [200, 400, 800],
        "clf__max_depth": [3, 5, 8],
        "clf__learning_rate": [0.03, 0.07, 0.1],
        "clf__subsample": [0.8, 1.0],
        "clf__colsample_bytree": [0.8, 1.0],
        "clf__min_child_weight": [1, 5],
        "clf__reg_lambda": [1.0, 5.0, 10.0],
    }))

    # Если SMOTE включен - добавить параметры sampler'а во все grids
    if use_smote:
        smote_grid = list(ParameterGrid({
            "smote__k_neighbors": [3, 5, 7],
            "smote__sampling_strategy": ["not majority", "auto"],
        }))
        def product(a, b):
            out = []
            for x in a:
                for y in b:
                    z = dict(x)
                    z.update(y)
                    out.append(z)
            return out

        lr_grid = product(lr_grid, smote_grid)
        rf_grid = product(rf_grid, smote_grid)
        xgb_grid = product(xgb_grid, smote_grid)

    grids["logreg"] = lr_grid
    grids["rf"] = rf_grid
    grids["xgb"] = xgb_grid
    return grids


@dag(
    dag_id=DAG_ID,
    start_date=pendulum.now(TZ).subtract(days=1),
    schedule="0 4 * * *",
    catchup=False,
    max_active_runs=1,
    default_args={
        "owner": "hokinhim",
        "retries": 1,
        "retry_delay": timedelta(minutes=3),
        "email_on_failure": False,
        "email_on_retry": False,
    },
    tags=["mlops", "ids", "tuning", "smote"],
)
def pipeline():
    @task
    def download_version_and_upload_samples() -> Dict[str, Any]:
        """
        Скачивание train/test UNSW-NB15, подсчет sha256,
        создание сбалансированного sample из train, загрузка sample + meta в S3.
        """
        import tempfile
        import pandas as pd

        bucket = get_var("MLFLOW_BUCKET")
        if not bucket:
            raise RuntimeError("Airflow Variable MLFLOW_BUCKET is required")

        # Зеркала
        default_train = "https://raw.githubusercontent.com/Nir-J/ML-Projects/master/UNSW-Network_Packet_Classification/UNSW_NB15_training-set.csv"
        default_test = "https://raw.githubusercontent.com/Nir-J/ML-Projects/master/UNSW-Network_Packet_Classification/UNSW_NB15_testing-set.csv"

        train_url = get_var("IDS_DATASET_TRAIN_URL", default_train)
        test_url = get_var("IDS_DATASET_TEST_URL", default_test)

        # target по умолчанию — attack_cat
        target_col = get_var("IDS_TARGET_COLUMN", "attack_cat")

        # Сэмплирование
        max_total_rows = int(get_var("IDS_MAX_TOTAL_ROWS", "180000"))
        max_per_class = int(get_var("IDS_MAX_PER_CLASS", "25000"))
        
        # Инициализация S3-hook для загрузки в S3 и настройки MLflow
        s3 = S3Hook(aws_conn_id=S3_CONN_ID)

        # Скачивание train/test, подсчет sha256
        with tempfile.TemporaryDirectory() as tmp:
            train_path = os.path.join(tmp, "train.csv")
            test_path = os.path.join(tmp, "test.csv")

            download_to_file(train_url, train_path)
            download_to_file(test_url, test_path)

            train_sha = sha256_file(train_path)
            test_sha = sha256_file(test_path)

            # Сбалансированный sample из train
            chunksize = 80_000
            counts = {}
            parts = []
            total = 0
            for chunk in pd.read_csv(train_path, chunksize=chunksize, low_memory=False, on_bad_lines="skip"):
                if target_col not in chunk.columns:
                    raise RuntimeError(
                        f"Target column '{target_col}' not found. "
                        f"Columns sample: {list(chunk.columns)[:20]}"
                    )
                
                chunk[target_col] = chunk[target_col].astype(str)
                
                for cls, sub in chunk.groupby(target_col):
                    have = counts.get(cls, 0)
                    need = max_per_class - have
                    if need <= 0:
                        continue

                    take = min(need, len(sub), max_total_rows - total)
                    if take <= 0:
                        continue

                    parts.append(sub.sample(n=take, random_state=42))
                    counts[cls] = have + take
                    total += take

                    if total >= max_total_rows:
                        break

                if total >= max_total_rows:
                    break

            if not parts:
                raise RuntimeError("Sampling failed: got 0 rows")

            # Финальный sample
            df_sample = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)

            # sha256 для sample (по CSV без индекса, чтобы было одинаково при любом порядке колонок)
            sample_hash = hashlib.sha256(df_sample.to_csv(index=False).encode("utf-8")).hexdigest()

            # Организация ключей в S3: datasets/ids/unsw_nb15/{train_sha}_{test_sha}/{sample_hash}/meta.json
            base = f"{S3_PREFIX}/{train_sha[:12]}_{test_sha[:12]}/{sample_hash[:12]}"
            meta_key = f"{base}/meta.json"
            sample_key = f"{base}/train_sample.pkl"
            meta = {
                "train_url": train_url,
                "test_url": test_url,
                "train_sha256": train_sha,
                "test_sha256": test_sha,
                "sample_sha256": sample_hash,
                "target_col": target_col,
                "rows_sample": int(df_sample.shape[0]),
                "cols_sample": int(df_sample.shape[1]),
                "class_counts_sample": {str(k): int(v) for k, v in counts.items()},
                "notes": "sample from training-set; evaluation will use official testing-set",
            }

            # Загрузка meta и sample в S3
            meta_buf = io.BytesIO(json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8"))
            s3.load_file_obj(meta_buf, bucket_name=bucket, key=meta_key, replace=True)

            # Загрузка sample в S3 в виде pickle
            buf = io.BytesIO()
            df_sample.to_pickle(buf)
            buf.seek(0)
            s3.load_file_obj(buf, bucket_name=bucket, key=sample_key, replace=True)

            # Вернуть meta для дальнейшего использования в тюнинге
            return {
                "bucket": bucket,
                "meta_key": meta_key,
                "sample_key": sample_key,
                "target_col": target_col,
                "train_url": train_url,
                "test_url": test_url,
                "train_sha256": train_sha,
                "test_sha256": test_sha,
                "sample_sha256": sample_hash,
            }

    @task
    def run_tuning_experiment(dataset_meta: Dict[str, Any], use_smote: bool = False) -> Dict[str, Any]:
        import tempfile
        import pandas as pd
        import mlflow
        import mlflow.sklearn
        from mlflow.models import infer_signature
        from mlflow.tracking import MlflowClient
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import OrdinalEncoder, StandardScaler
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
        from xgboost import XGBClassifier

        # Настройка MLflow из S3 connection
        configure_mlflow_from_s3_connection()

        # Построение grid'ов гиперпараметров для моделей
        base_exp = get_var("MLFLOW_EXPERIMENT_NAME", "UNSW_IDS_TUNING")
        exp_name = f"{base_exp}_{'smote' if use_smote else 'baseline'}"
        mlflow.set_experiment(exp_name)

        # Лимиты trial'ов
        max_lr = int(get_var("TUNE_MAX_TRIALS_LR", "3"))
        max_rf = int(get_var("TUNE_MAX_TRIALS_RF", "3"))
        max_xgb = int(get_var("TUNE_MAX_TRIALS_XGB", "3"))
        limits = {"logreg": max_lr, "rf": max_rf, "xgb": max_xgb}

        # Загрузка sample из S3
        s3 = S3Hook(aws_conn_id=S3_CONN_ID)
        local_sample = s3.download_file(key=dataset_meta["sample_key"], bucket_name=dataset_meta["bucket"])
        df_train = pd.read_pickle(local_sample)

        # Проверка наличия target_col в sample
        target_col = dataset_meta["target_col"]
        if target_col not in df_train.columns:
            raise RuntimeError(f"Target col '{target_col}' not in sample columns")

        # Загрузка официального test
        import tempfile as _tf
        with _tf.TemporaryDirectory() as tmp:
            test_path = os.path.join(tmp, "test.csv")
            download_to_file(dataset_meta["test_url"], test_path)
            df_test = pd.read_csv(test_path, low_memory=False, on_bad_lines="skip")

        # Базовая очистка
        def drop_leaks(df):
            cols = []
            for c in df.columns:
                lc = c.lower()
                if c == target_col:
                    continue
                if any(p in lc for p in ["id", "flow", "ip", "srcip", "dstip", "port", "time", "timestamp", "date"]):
                    cols.append(c)
            return df.drop(columns=cols, errors="ignore")
        df_train = drop_leaks(df_train)
        df_test = drop_leaks(df_test)

        # Подготовка данных:
        # - target: label encoding (OrdinalEncoder)
        # - фичи: разделение на cat/num, OrdinalEncoder для cat и StandardScaler для num
        y_train_raw = df_train[target_col].astype(str)
        y_test_raw = df_test[target_col].astype(str)

        classes = sorted(set(y_train_raw.unique().tolist()) | set(y_test_raw.unique().tolist()))
        class_to_id = {c: i for i, c in enumerate(classes)}

        y_train = y_train_raw.map(class_to_id).astype(int)
        y_test = y_test_raw.map(class_to_id).astype(int)

        X_train_full = df_train.drop(columns=[target_col], errors="ignore")
        X_test = df_test.drop(columns=[target_col], errors="ignore")

        num_classes = len(classes)

        # Разделить train на train/val для тюнинга (75/25)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train_full, y_train, test_size=0.25, random_state=42, stratify=y_train
        )

        # Определить cat/num колонки для препроцессинга
        cat_cols = [c for c in X_tr.columns if X_tr[c].dtype == object]
        num_cols = [c for c in X_tr.columns if c not in cat_cols]

        # Препроцессор под SMOTENC:
        # - cat: OrdinalEncoder -> числа
        # - num: StandardScaler
        # на выходе плотная матрица, где первые len(cat_cols) признаков - категориальные индексы
        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
                ("num", StandardScaler(), num_cols),
            ],
            remainder="drop",
            sparse_threshold=0.0,
        )

        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        models = {
            "logreg": LogisticRegression(max_iter=400),
            "rf": RandomForestClassifier(random_state=42, n_jobs=-1),
            "xgb": XGBClassifier(
                objective="multi:softprob",
                num_class=num_classes,
                tree_method="hist",
                eval_metric="mlogloss",
                random_state=42,
                n_jobs=4,
            ),
        }

        # SMOTE
        if use_smote:
            from imblearn.over_sampling import SMOTENC
            from imblearn.pipeline import Pipeline as ImbPipeline
            cat_idx = list(range(len(cat_cols)))
            sampler = SMOTENC(
                categorical_features=cat_idx,
                sampling_strategy="not majority",
                k_neighbors=5,
                random_state=42,
            )
            PipelineCls = ImbPipeline
        else:
            sampler = None
            PipelineCls = Pipeline

        grids = build_param_grids(num_classes=num_classes, use_smote=use_smote)

        # Метрики для оценки
        def macro_f1(y_true, y_pred) -> float:
            return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        def acc(y_true, y_pred) -> float:
            return float(accuracy_score(y_true, y_pred))

        parent_run_name = f"{'smote' if use_smote else 'baseline'}_{dataset_meta['train_sha256'][:8]}_{dataset_meta['sample_sha256'][:8]}"

        best_overall = None  # (val_score, family, params, model_uri, test_score, run_id)

        with mlflow.start_run(run_name=parent_run_name) as parent:
            # Теги и параметры для всего эксперимента
            mlflow.set_tag("sampling", "smote" if use_smote else "baseline")
            mlflow.set_tag("dataset_train_sha256", dataset_meta["train_sha256"])
            mlflow.set_tag("dataset_test_sha256", dataset_meta["test_sha256"])
            mlflow.set_tag("dataset_sample_sha256", dataset_meta["sample_sha256"])
            mlflow.set_tag("target_col", target_col)
            mlflow.set_tag("problem", "UNSW-NB15_multiclass_attack_cat")
            mlflow.log_param("num_classes", num_classes)
            mlflow.log_param("rows_train_sample", int(len(X_train_full)))
            mlflow.log_param("rows_test", int(len(X_test)))
            mlflow.log_param("n_cat_cols", int(len(cat_cols)))
            mlflow.log_param("n_num_cols", int(len(num_cols)))

            # Логирование class mapping как артефакт
            with tempfile.TemporaryDirectory() as tmp:
                p = os.path.join(tmp, "class_mapping.json")
                with open(p, "w", encoding="utf-8") as f:
                    json.dump({"classes": classes, "class_to_id": class_to_id}, f, ensure_ascii=False, indent=2)
                mlflow.log_artifact(p, artifact_path="dataset")

            # Тюнинг моделей
            for family, clf in models.items():
                max_trials = limits[family]
                trials = sample_trials(grids[family], max_trials=max_trials, seed=42)
                
                # Каждый trial - отдельный MLflow run внутри родительского
                with mlflow.start_run(run_name=f"{family}_tuning", nested=True):
                    mlflow.set_tag("model_family", family)
                    mlflow.log_param("trials", len(trials))

                    best_family = None  # (val_score, params, fitted_pipeline)

                    for i, params in enumerate(trials):
                        with mlflow.start_run(run_name=f"{family}_trial_{i:03d}", nested=True) as trial_run:
                            # Построить пайплайн с препроцессором, опциональным SMOTE и классификатором, установить параметры из grid'а
                            steps = [("prep", preprocessor)]
                            if sampler is not None:
                                steps.append(("smote", sampler))
                            steps.append(("clf", clf))
                            pipe = PipelineCls(steps=steps)
                            pipe.set_params(**params)

                            # Логирование параметров trial'а
                            mlflow.log_params(params)

                            t0 = time.time()
                            pipe.fit(X_tr, y_tr)
                            mlflow.log_metric("fit_seconds", float(time.time() - t0))

                            y_val_pred = pipe.predict(X_val)
                            val_f1 = macro_f1(y_val, y_val_pred)
                            val_acc = acc(y_val, y_val_pred)

                            mlflow.log_metric("val_f1_macro", val_f1)
                            mlflow.log_metric("val_accuracy", val_acc)

                            if best_family is None or val_f1 > best_family[0]:
                                best_family = (val_f1, dict(params), pipe)

                    if best_family is None:
                        raise RuntimeError(f"No successful trials for {family}")

                    # После тюнинга - обучить лучшую модель на объединении train+val и оценить на тесте
                    best_val_f1, best_params, best_pipe = best_family

                    X_tv = pd.concat([X_tr, X_val], axis=0)
                    y_tv = pd.concat([y_tr, y_val], axis=0)

                    t0 = time.time()
                    best_pipe.fit(X_tv, y_tv)
                    mlflow.log_metric("retrain_seconds", float(time.time() - t0))
                    mlflow.log_metric("best_val_f1_macro", float(best_val_f1))

                    y_test_pred = best_pipe.predict(X_test)
                    test_f1 = macro_f1(y_test, y_test_pred)
                    test_acc = acc(y_test, y_test_pred)

                    mlflow.log_metric("test_f1_macro", test_f1)
                    mlflow.log_metric("test_accuracy", test_acc)

                    # model signature для лучшей модели в семействе, логирование модели в MLflow
                    signature = infer_signature(X_test, y_test_pred)
                    model_info = mlflow.sklearn.log_model(
                        sk_model=best_pipe,
                        name="model",
                        signature=signature,
                    )

                    # Логирование classification report и confusion matrix как артефакты
                    with tempfile.TemporaryDirectory() as tmp:
                        rep = os.path.join(tmp, "classification_report.txt")
                        with open(rep, "w", encoding="utf-8") as f:
                            f.write(classification_report(y_test, y_test_pred, zero_division=0))
                        mlflow.log_artifact(rep, artifact_path="reports")

                        cm = confusion_matrix(y_test, y_test_pred)
                        cmj = os.path.join(tmp, "confusion_matrix.json")
                        with open(cmj, "w", encoding="utf-8") as f:
                            json.dump(cm.tolist(), f, ensure_ascii=False, indent=2)
                        mlflow.log_artifact(cmj, artifact_path="reports")

                    # Обновить лучшую модель среди всех семейств, сравнивая по тестовому macro F1
                    if best_overall is None or test_f1 > best_overall[0]:
                        best_overall = (test_f1, family, best_params, model_info.model_uri, test_acc, trial_run.info.run_id)

            if best_overall is None:
                raise RuntimeError("best_overall is None")

            # Логирование лучшей модели среди всех семейств как тегов и метрик в родительском run'е
            best_f1, best_family, best_params, best_uri, best_acc, best_run_id = best_overall
            mlflow.set_tag("best_family", best_family)
            mlflow.set_tag("best_model_uri", best_uri)
            mlflow.set_tag("best_model_run_id", best_run_id)
            mlflow.log_metric("best_test_f1_macro", float(best_f1))

            summary = {
                "experiment": exp_name,
                "sampling": "smote" if use_smote else "baseline",
                "best_family": best_family,
                "best_test_f1_macro": float(best_f1),
                "best_test_accuracy": float(best_acc),
                "best_params": best_params,
                "best_model_uri": best_uri,
                "dataset": {
                    "train_sha256": dataset_meta["train_sha256"],
                    "test_sha256": dataset_meta["test_sha256"],
                    "sample_sha256": dataset_meta["sample_sha256"],
                    "target_col": target_col,
                    "num_classes": num_classes,
                },
            }

            # Регистрация лучшей модели в MLflow Model Registry с тегами и алиасом
            reg_name_base = get_var("MLFLOW_REGISTERED_MODEL_NAME", "UNSW_IDS_Champion")
            alias = "smote_champion" if use_smote else "baseline_champion"
            client = MlflowClient()

            registry = {"status": "skipped"}
            try:
                mv = mlflow.register_model(model_uri=best_uri, name=reg_name_base)
                client.set_registered_model_alias(reg_name_base, alias, mv.version)
                client.set_model_version_tag(reg_name_base, mv.version, "sampling", summary["sampling"])
                client.set_model_version_tag(reg_name_base, mv.version, "best_family", best_family)
                client.set_model_version_tag(reg_name_base, mv.version, "dataset_sample_sha256", dataset_meta["sample_sha256"])
                registry = {"status": "ok", "name": reg_name_base, "version": mv.version, "alias": alias}
            except Exception as e:
                registry = {"status": "failed", "error": str(e)}

            # Добавить информацию о регистрации модели в summary
            summary["registry"] = registry
            with tempfile.TemporaryDirectory() as tmp:
                p = os.path.join(tmp, "summary.json")
                with open(p, "w", encoding="utf-8") as f:
                    json.dump(summary, f, ensure_ascii=False, indent=2)
                mlflow.log_artifact(p, artifact_path="reports")

            # ПРодублировать summary в S3
            out_key = f"{S3_REPORT_PREFIX}/{summary['sampling']}_summary_{dataset_meta['train_sha256'][:8]}_{dataset_meta['sample_sha256'][:8]}.json"
            buf = io.BytesIO(json.dumps(summary, ensure_ascii=False, indent=2).encode("utf-8"))
            s3.load_file_obj(buf, bucket_name=dataset_meta["bucket"], key=out_key, replace=True)

            return summary

    ds_meta = download_version_and_upload_samples()
    baseline_summary = run_tuning_experiment.override(task_id="tune_baseline")(dataset_meta=ds_meta, use_smote=False)
    smote_summary = run_tuning_experiment.override(task_id="tune_smote")(dataset_meta=ds_meta, use_smote=True)


dag = pipeline()
