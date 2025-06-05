import pandas as pd
from typing import Any
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
import mlflow


def entrenar_kmeans(df: pd.DataFrame) -> Any:
    """
    Nodo que entrena un modelo KMeans y lo devuelve. Kedro-MLflow capturará
    automáticamente el objeto retornado y lo guardará en el dataset 'kmeans_model'.

    Recibe:
        df: DataFrame con las columnas necesarias para clustering.

    Retorna:
        pipeline: sklearn.Pipeline con ColumnTransformer + KMeans entrenado.
    """
    # Definir mapeo de tipos
    dtype_map = {
        "h_num_adu_cat": "object",
        "hay_menores": "bool",
        "h_num_noc_cat": "object",
        "Estado_cve": "object",
        "Tipo_Habitacion_Nombre": "object",
        "Tipo_Habitacion_Detalles": "object"
    }

    # Asegurar tipos correctos
    df_temp = df.copy()
    for col, dtype in dtype_map.items():
        if col in df_temp.columns:
            try:
                if dtype == "bool":
                    df_temp[col] = df_temp[col].astype(bool)
                else:
                    df_temp[col] = df_temp[col].astype(dtype)
            except Exception:
                # Podemos loguear un warning, pero no interrumpir la ejecución
                print(f"Warning: no se pudo convertir {col} a {dtype}")

    # Columnas a usar
    num_cols = []  # Si en el futuro agregas columnas numéricas, inclúyelas aquí
    cat_cols = [
        "h_num_adu_cat",
        "hay_menores",
        "h_num_noc_cat",
        "Estado_cve",
        "Tipo_Habitacion_Nombre",
        "Tipo_Habitacion_Detalles"
    ]

    # Preprocesador: solo OneHotEncoder para las categóricas
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
    ])

    # Definir número de clusters
    num_clusters = 5

    # Construir pipeline completo
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("cluster", KMeans(n_clusters=num_clusters, random_state=42))
    ])

    # Entrenar pipeline
    X = df_temp[num_cols + cat_cols]
    pipeline.fit(X)

    # Registrar el parámetro en MLflow
    mlflow.log_param("n_clusters", num_clusters)

    # Devolver el objeto pipeline: Kedro-MLflow se encargará de guardarlo
    return pipeline


def entrenar_kmeans_v2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nodo alternativo que entrena KMeans y, además de devolver el DataFrame con
    la columna 'cluster', salva los artefactos de preprocesador y modelo en archivos
    locales (si así lo deseas). No está integrado con Kedro-MLflow para el modelo,
    pero retorna el DataFrame con la columna 'cluster'.
    """
    # Mapeo de tipos (igual que en el nodo anterior)
    dtype_map = {
        "h_num_adu_cat": "object",
        "hay_menores": "bool",
        "h_num_noc_cat": "object",
        "Estado_cve": "object",
        "Tipo_Habitacion_Nombre": "object",
        "Tipo_Habitacion_Detalles": "object"
    }

    df_temp = df.copy()
    for col, dtype in dtype_map.items():
        if col in df_temp.columns:
            try:
                if dtype == "bool":
                    df_temp[col] = df_temp[col].astype(bool)
                else:
                    df_temp[col] = df_temp[col].astype(dtype)
            except Exception:
                print(f"Warning: no se pudo convertir {col} a {dtype}")

    num_cols = []
    cat_cols = [
        "h_num_adu_cat",
        "hay_menores",
        "h_num_noc_cat",
        "Estado_cve",
        "Tipo_Habitacion_Nombre",
        "Tipo_Habitacion_Detalles"
    ]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
    ])

    kmeans = KMeans(n_clusters=5, random_state=42)

    # Entrenar
    X = df_temp[num_cols + cat_cols]
    df_temp["cluster"] = kmeans.fit_predict(X)

    # Si quieres exportar localmente:
    import joblib
    joblib.dump(preprocessor, "data/06_models/kmeans_v2_preprocessor.pkl")
    joblib.dump(kmeans, "data/06_models/kmeans_v2_model.pkl")

    return df_temp
