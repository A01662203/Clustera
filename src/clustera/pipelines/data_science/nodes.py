import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
import mlflow

# Establecer la URI de seguimiento de MLflow
mlflow.set_tracking_uri("http://127.0.1:5000")
# Nombre del experimento
mlflow.set_experiment("kmeans")

# Versión 1: clustering sin guardar artefactos
# def entrenar_kmeans(df: pd.DataFrame) -> pd.DataFrame:
#     dtype_map = {
#         "h_num_adu_cat": "object",
#         "hay_menores": "bool",
#         "h_num_noc_cat": "object",
#         "Estado_cve": "object",
#         "Tipo_Habitacion_Nombre": "object",
#         "Tipo_Habitacion_Detalles": "object"
#     }

#     df_temp = df.copy()
#     for col, dtype in dtype_map.items():
#         if col in df_temp.columns:
#             try:
#                 if dtype == "bool":
#                     df_temp[col] = df_temp[col].astype(bool)
#                 else:
#                     df_temp[col] = df_temp[col].astype(dtype)
#             except Exception as e:
#                 print(f"Error al convertir {col} a {dtype}: {e}")

#     num_cols = []
#     cat_cols = ["h_num_adu_cat", "hay_menores", "h_num_noc_cat", "Estado_cve",
#                 "Tipo_Habitacion_Nombre", "Tipo_Habitacion_Detalles"]

#     preprocessor = ColumnTransformer([
#         ("num", StandardScaler(), num_cols),
#         ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
#     ])

#     pipeline = Pipeline(steps=[
#         ("preprocessor", preprocessor),
#         ("cluster", KMeans(n_clusters=5, random_state=42))
#     ])

#     X = df_temp[num_cols + cat_cols]
#     cluster_labels = pipeline.fit_predict(X)

#     df_temp["cluster"] = cluster_labels

#     return df_temp

def entrenar_kmeans(df: pd.DataFrame) -> Pipeline:
    """
    Entrena un modelo de clustering KMeans y devuelve el pipeline entrenado.

    Args:
        df: DataFrame de entrada con las características para clustering.

    Returns:
        Pipeline entrenado con el preprocesador y el modelo KMeans.
    """
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
            except Exception as e:
                print(f"Error al convertir {col} a {dtype}: {e}")

    num_cols = []
    cat_cols = ["h_num_adu_cat", "hay_menores", "h_num_noc_cat", "Estado_cve",
                 "Tipo_Habitacion_Nombre", "Tipo_Habitacion_Detalles"]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
    ])
    num_clusters = 5  # Número de clusters

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("cluster", KMeans(n_clusters=num_clusters, random_state=42))
    ])

    X = df_temp[num_cols + cat_cols]
    pipeline.fit(X)

    with mlflow.start_run():
        mlflow.log_param("n_clusters", num_clusters)
        mlflow.log_metric

    # Devuelve el pipeline directamente, para que lo maneje Kedro-MLflow
    return pipeline


# Versión 2: clustering + exportación de artefactos para predicción
def entrenar_kmeans_v2(df: pd.DataFrame) -> pd.DataFrame:
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
            except Exception as e:
                print(f"Error al convertir {col} a {dtype}: {e}")

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

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("cluster", KMeans(n_clusters=5, random_state=42))
    ])

    X = df_temp[num_cols + cat_cols]
    cluster_labels = pipeline.fit_predict(X)
    df_temp["cluster"] = cluster_labels

    # Guardar artefactos en el directorio del modelo
    joblib.dump(preprocessor, "data/06_models/kmeans_v2_preprocessor.pkl")
    joblib.dump(pipeline.named_steps["cluster"], "data/06_models/kmeans_v2_model.pkl")

    return df_temp
