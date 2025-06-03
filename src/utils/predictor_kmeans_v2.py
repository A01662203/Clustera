import pandas as pd
import joblib
from pathlib import Path

# Ruta base del proyecto Kedro
BASE_PATH = Path(__file__).resolve().parents[2]

# Cargar artefactos entrenados previamente (ajusta si los guardas con joblib.dump)
preprocessor = joblib.load(BASE_PATH / "data/06_models/kmeans_v2_preprocessor.pkl")
kmeans_model = joblib.load(BASE_PATH / "data/06_models/kmeans_v2_model.pkl")

# DataFrame ya clusterizado como referencia
df_clustered = pd.read_csv(BASE_PATH / "data/05_model_input/clustered_reservaciones_segmentadas.csv")

# Ejemplo de nueva entrada (cámbiala desde Streamlit u otro input)
new_input = pd.DataFrame([{
    "h_num_adu_cat": "",
    "hay_menores": True,
    "h_num_noc_cat": "",
    "Estado_cve": "",
    "Tipo_Habitacion_Nombre": "",   # Se predicen
    "Tipo_Habitacion_Detalles": ""  # Se predicen
}])

# Transformar la entrada
X_new = preprocessor.transform(new_input)

# Predecir clúster
predicted_cluster = kmeans_model.predict(X_new)[0]

# Filtrar registros del mismo clúster
matching = df_clustered[df_clustered["cluster"] == predicted_cluster]

# Habitaciones sugeridas
recommended_rooms = matching["Tipo_Habitacion_Nombre"].value_counts().index.tolist()

# Detalles sugeridos
recommended_details = matching["Tipo_Habitacion_Detalles"].value_counts().index.tolist()

# Paquete sugerido
valid_packages = matching[
    ~matching["Paquete_nombre"].isin(["WALK IN", "SIN DEFINIR"])
]

if not valid_packages.empty:
    recommended_package = valid_packages["Paquete_nombre"].mode()[0]
else:
    recommended_package = "Sin recomendación disponible"

# Resultados

print(f"Habitaciones sugeridas: {recommended_rooms}")
print(f"Detalles sugeridos: {recommended_details}")
