import pandas as pd
import numpy as np

def calcular_tabla_anticipacion(df: pd.DataFrame) -> pd.DataFrame:
    # Convertir fechas a datetime
    df["Fecha_hoy"] = pd.to_datetime(df["Fecha_hoy"])
    df["h_fec_lld_ok"] = pd.to_datetime(df["h_fec_lld_ok"])

    # Calcular anticipación y otras variables temporales
    df["meses_anticipacion"] = (df["h_fec_lld_ok"].dt.year - df["Fecha_hoy"].dt.year) * 12 + (df["h_fec_lld_ok"].dt.month - df["Fecha_hoy"].dt.month)
    df["año_llegada"] = df["h_fec_lld_ok"].dt.year
    df["mes_llegada"] = df["h_fec_lld_ok"].dt.month
    df["año_rsv"] = df["Fecha_hoy"].dt.year
    df["mes_rsv"] = df["Fecha_hoy"].dt.month
    df["num_sem_rsv"] = ((df["Fecha_hoy"].dt.day - 1) // 7) + 1
    df["num_sem_llegada"] = ((df["h_fec_lld_ok"].dt.day - 1) // 7) + 1
    df["num_noc_rgo"] = df["h_num_noc_cat"]

    # Agrupación
    tabla_resumen = df.groupby([
        "Estado_cve",
        "Tipo_Habitacion_Nombre",
        "meses_anticipacion",
        "año_llegada",
        "mes_llegada",
        "año_rsv",
        "mes_rsv",
        "num_sem_rsv",
        "num_sem_llegada",
        "num_noc_rgo"
    ]).agg(
        lista_ids=("ID_Reserva", list),
        conteo=("h_tfa_total", "count"),
        max_tfa_total=("h_tfa_total", "max"),
        min_tfa_total=("h_tfa_total", "min"),
        avg_tfa_total=("h_tfa_total", "mean"),
        mda_tfa_total=("h_tfa_total", lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
    ).reset_index()

    # Expandir lista de IDs
    tabla_expandida = tabla_resumen.explode("lista_ids").rename(columns={"lista_ids": "ID_Reserva"})

    # Merge para recuperar columnas adicionales (solo de esas reservas)
    columnas_extra = ["ID_Reserva", "Fecha_hoy", "h_num_adu_cat", "hay_menores", "h_num_noc", "h_tfa_total"]
    tabla_expandida = tabla_expandida.merge(df[columnas_extra], on="ID_Reserva", how="left")

    # Reordenar columnas 
    columnas_principales = [
        "ID_Reserva", "Fecha_hoy", "Estado_cve", "Tipo_Habitacion_Nombre",
        "meses_anticipacion", "año_llegada", "mes_llegada", "num_sem_llegada",
        "año_rsv", "mes_rsv", "num_sem_rsv", "num_noc_rgo", "conteo",
        "max_tfa_total", "min_tfa_total", "avg_tfa_total", "mda_tfa_total",
        "h_num_adu_cat", "hay_menores", "h_num_noc", "h_tfa_total"
    ]
    tabla_expandida = tabla_expandida[columnas_principales]

    tabla_expandida.to_csv("data/03_primary/tabla_desglosada_anticipacion.csv", index=False)
    return tabla_expandida
