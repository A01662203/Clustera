import pandas as pd
import numpy as np
import json

def calcular_tabla_anticipacion(df: pd.DataFrame) -> pd.DataFrame:
    # Convertir las columnas de fecha a datetime
    df["Fecha_hoy"] = pd.to_datetime(df["Fecha_hoy"])
    df["h_fec_lld_ok"] = pd.to_datetime(df["h_fec_lld_ok"])

    # Calcular columnas auxiliares
    df["meses_anticipacion"] = (df["h_fec_lld_ok"].dt.year - df["Fecha_hoy"].dt.year) * 12 + (df["h_fec_lld_ok"].dt.month - df["Fecha_hoy"].dt.month)
    df["año_llegada"] = df["h_fec_lld_ok"].dt.year
    df["mes_llegada"] = df["h_fec_lld_ok"].dt.month
    df["año_rsv"] = df["Fecha_hoy"].dt.year
    df["mes_rsv"] = df["Fecha_hoy"].dt.month
    df["num_noc_rgo"] = df["h_num_noc_cat"]

    # Agrupar
    tabla_resumen = df.groupby([
        "Estado_cve",
        "Tipo_Habitacion_Nombre",
        "meses_anticipacion",
        "año_llegada",
        "mes_llegada",
        "año_rsv",
        "mes_rsv",
        "num_noc_rgo"
    ]).agg(
        lista_ids=("ID_Reserva", lambda x: x.tolist()),
        conteo=("h_tfa_total", "count"),
        max_tfa_total=("h_tfa_total", "max"),
        min_tfa_total=("h_tfa_total", "min"),
        avg_tfa_total=("h_tfa_total", "mean"),
        mda_tfa_total=("h_tfa_total", lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
    ).reset_index()

    # Expandir lista_ids
    tabla_expandida = tabla_resumen.explode("lista_ids").rename(columns={"lista_ids": "ID_Reserva"})

    # Reordenar columnas
    cols = ['ID_Reserva', 'conteo'] + [col for col in tabla_expandida.columns if col not in ['ID_Reserva', 'conteo']]
    tabla_expandida = tabla_expandida[cols]

    tabla_expandida.to_csv("data/03_primary/tabla_desglosada_anticipacion.csv", index=False)
    return tabla_expandida


