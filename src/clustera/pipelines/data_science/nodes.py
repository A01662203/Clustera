import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline

# Versión 1: clustering sin guardar artefactos
def entrenar_kmeans(df: pd.DataFrame) -> pd.DataFrame:
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

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("cluster", KMeans(n_clusters=5, random_state=42))
    ])

    X = df_temp[num_cols + cat_cols]
    cluster_labels = pipeline.fit_predict(X)

    df_temp["cluster"] = cluster_labels

    return df_temp

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
