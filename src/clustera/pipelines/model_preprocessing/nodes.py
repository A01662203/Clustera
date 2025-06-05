import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data(raw_data: pd.DataFrame, params: dict) -> dict:
    df = raw_data.copy()

    # Convertir fechas
    df["Fecha_hoy"] = pd.to_datetime(df["Fecha_hoy"])
    df["Fecha_llegada"] = pd.to_datetime(df["Fecha_llegada"])
    
    # Cálculos temporales
    df["mes_llegada_sin"] = np.sin(2 * np.pi * (df["mes_llegada"] - 1) / 12)
    df["mes_llegada_cos"] = np.cos(2 * np.pi * (df["mes_llegada"] - 1) / 12)
    df["mes_rsv_sin"] = np.sin(2 * np.pi * (df["mes_rsv"] - 1) / 12)
    df["mes_rsv_cos"] = np.cos(2 * np.pi * (df["mes_rsv"] - 1) / 12)

    # Fin de semana (booleano)
    df["is_weekend_reserva"] = df["Fecha_hoy"].dt.weekday >= 5
    df["is_weekend_llegada"] = df["Fecha_llegada"].dt.weekday >= 5

    # Día de la semana
    df["dow_reserva"] = df["Fecha_hoy"].dt.weekday
    df["dow_llegada"] = df["Fecha_llegada"].dt.weekday

    # Lead time en días
    df["dias_anticipacion"] = (df["Fecha_llegada"] - df["Fecha_hoy"]).dt.days

    # Drop rows where 'Tipo_Habitacion_Nombre' is HANDICAP, HONEY and SUITE PRES
    df = df[~df["Tipo_Habitacion_Nombre"].isin(["HANDICAP", "HONEY", "SUITE PRES"])]
    
    # Selección de features finales
    columns_to_keep = [
        "h_num_adu_cat",
        "hay_menores",
        "h_num_noc_cat",
        "meses_anticipacion",
        "Estado_cve",
        "num_sem_llegada",
        "num_sem_rsv",
        "mes_llegada_sin",
        "mes_llegada_cos",
        "mes_rsv_sin",
        "mes_rsv_cos",
        "is_weekend_reserva",
        "is_weekend_llegada",
        "dias_anticipacion",
        "dow_reserva",
        "dow_llegada"
    ]

    target = "Tipo_Habitacion_Nombre"
    X = df[columns_to_keep]
    y = df[target]

    # Train-test split
    test_size = 1 - params["data"]["train_split"]
    random_state = params["data"]["random_seed"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }
