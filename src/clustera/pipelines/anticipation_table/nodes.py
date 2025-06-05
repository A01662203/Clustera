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
        "h_num_noc_cat"
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

    # Merge para recuperar columnas adicionales
    columnas_extra = ["ID_Reserva", "Fecha_hoy", "h_num_adu_cat", "hay_menores", "h_num_noc", "h_tfa_total"]
    tabla_expandida = tabla_expandida.merge(df[columnas_extra], on="ID_Reserva", how="left")

    # Diccionario de claves a nombres de estado
    diccionario_estados = {
        "EAGU": "AGUASCALIENTES",
        "ECA": "CALIFORNIA",
        "ECH": "CHIHUAHUA",
        "ECRI": "COSTA RICA",
        "EDF": "DISTRITO FEDERAL",
        "EFL": "FLORIDA",
        "EGA": "GEORGIA",
        "EGR": "GUERRERO",
        "EGT": "GUANAJUATO",
        "EIL": "ILLINOIS",
        "EJA": "JALISCO",
        "EMC": "MICHOACÁN",
        "EMN": "MINNESOTA",
        "EMR": "MORELOS",
        "EMX": "MÉXICO",
        "ENL": "NUEVO LEÓN",
        "EON": "ONTARIO",
        "EPB": "PUEBLA",
        "EQE": "QUERÉTARO",
        "EQR": "QUINTANA ROO",
        "ESL": "SAN LUIS POTOSÍ",
        "ETX": "TEXAS",
        "AZU": "Azul"
    }

    # Limpiar espacios en Estado_cve antes de mapear
    tabla_expandida["nombre_estado"] = tabla_expandida["Estado_cve"].str.strip().map(diccionario_estados)

    # Reordenar columnas
    columnas_principales = [
        "ID_Reserva", "Fecha_hoy", "Estado_cve", "nombre_estado", "Tipo_Habitacion_Nombre",
        "meses_anticipacion", "año_llegada", "mes_llegada", "num_sem_llegada",
        "año_rsv", "mes_rsv", "num_sem_rsv", "h_num_noc_cat", "conteo",
        "max_tfa_total", "min_tfa_total", "avg_tfa_total", "mda_tfa_total",
        "h_num_adu_cat", "hay_menores", "h_num_noc", "h_tfa_total"
    ]
    tabla_expandida = tabla_expandida[columnas_principales]

    # Renombrar si la columna venía mal nombrada (opcional)
    tabla_expandida = tabla_expandida.rename(columns={"h_num_rgo": "h_num_noc_cat"})

    # Exportar
    tabla_expandida.to_csv("data/03_primary/tabla_desglosada_anticipacion.csv", index=False)
    return tabla_expandida
