import pandas as pd
import matplotlib.pyplot as plt
from nixtla import NixtlaClient
from matplotlib.figure import Figure

def detectar_anomalias_y_retornar_figura(df: pd.DataFrame, api_key: str) -> Figure:
    nixtla_client = NixtlaClient(api_key=api_key)

    df['Fecha_hoy'] = pd.to_datetime(df['Fecha_hoy'])
    df = df[df['h_num_noc'] != 0]
    df['Tarifa_promedio'] = df['h_tfa_total'] / df['h_num_noc']

    serie = df.groupby(['Fecha_hoy', 'Tipo_Habitacion_Nombre'])['Tarifa_promedio'].mean().reset_index()
    serie.columns = ['ds', 'unique_id', 'y']
    serie = serie.sort_values(['unique_id', 'ds'])

    min_date = serie['ds'].min()
    max_date = serie['ds'].max()
    full_dates = pd.date_range(min_date, max_date, freq='D')
    habitaciones = serie['unique_id'].unique()
    full_index = pd.MultiIndex.from_product([habitaciones, full_dates], names=['unique_id', 'ds'])

    serie_full = serie.set_index(['unique_id', 'ds']).reindex(full_index).reset_index()
    serie_full['y'] = serie_full['y'].interpolate()
    serie_full = serie_full.dropna()

    anomalies_df = nixtla_client.detect_anomalies(serie_full, freq='D')
    
    fig = nixtla_client.plot(serie_full, anomalies_df, max_ids=9)
    
    return fig  # 👈 este es el truco clave
