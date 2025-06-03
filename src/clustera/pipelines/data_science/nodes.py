import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline

def entrenar_kmeans(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Forzar los tipos de datos
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

    # 2. Seleccionar columnas
    num_cols = []  # Coloca aquí tus columnas numéricas si tienes
    cat_cols = ["h_num_adu_cat", "hay_menores", "h_num_noc_cat", "Estado_cve", "Tipo_Habitacion_Nombre", "Tipo_Habitacion_Detalles"]

    # 3. Preprocesador
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
    ])

    # 4. Pipeline con KMeans (k=5 como ejemplo)
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("cluster", KMeans(n_clusters=5, random_state=42))
    ])

    # 5. Entrenamiento y predicción
    X = df_temp[num_cols + cat_cols]
    cluster_labels = pipeline.fit_predict(X)

    # 6. Añadir la etiqueta al DataFrame original
    df_temp["cluster"] = cluster_labels

    return df_temp
