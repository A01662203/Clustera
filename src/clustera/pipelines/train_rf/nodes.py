import pandas as pd
import re
import logging

logger = logging.getLogger(__name__)

def merge_reservaciones(iar_Reservaciones, iar_paquetes, iar_Agencias, iar_Tipos_Habitaciones):
    merged = iar_Reservaciones.merge(
        iar_Agencias[['ID_Agencia', 'Hotel_cve', 'Agencia_nombre', 'Estado_cve']],
        on='ID_Agencia',
        how='left'
    ).merge(
        iar_Tipos_Habitaciones[['ID_Tipo_Habitacion', 'Tipo_Habitacion_nombre', 'Clasificacion']],
        on='ID_Tipo_Habitacion',
        how='left'
    ).merge(
        iar_paquetes[['ID_paquete', 'Paquete_nombre']],
        left_on='ID_Paquete',
        right_on='ID_paquete',
        how='left'
    )
    return merged

def filtrar_reservaciones(df: pd.DataFrame):
    mascara = (
        (df['Reservacion'] != 0) &
        (df['ID_Programa'] != 0) &
        (df['ID_Paquete'].isin([1, 2])) &
        (df['h_num_per'] <= 10) &
        (df['h_num_noc'] <= 30) &
        (df['h_tfa_total'] > 0) &
        (df['h_num_noc'] > 0)
    )
    return df.loc[mascara].copy()

def limpieza_basica(df: pd.DataFrame):
    columnas_a_eliminar = [
        'h_res_fec', 'h_res_fec_okt', 'aa_h_num_per', 'aa_h_num_adu', 'aa_h_num_men', 'aa_h_num_noc', 'aa_h_tot_hab',
        'ID_empresa', 'h_fec_lld', 'h_fec_lld_okt', 'h_fec_reg', 'h_fec_reg_okt', 'h_fec_sda', 'h_fec_sda_okt',
        'aa_Reservacion', 'h_correo_e', 'h_nom', 'aa_h_tfa_total', 'moneda_cve', 'h_ult_cam_fec', 'h_ult_cam_fec_okt',
        'Hotel_cve', 'ID_Programa', 'ID_paquete', 'Paquete_nombre', 'aa_Cliente_Disp', 'Reservacion', 'h_fec_reg_ok',
        'ID_Paquete', 'h_res_fec_ok', 'h_ult_cam_fec_ok', 'h_codigop', 'h_cod_reserva', 'ID_estatus_reservaciones',
        'h_num_per', 'h_edo'
    ]
    columnas_tentativas = [
        'ID_Agencia', 'ID_Segmento_Comp', 'ID_Tipo_Habitacion', 'ID_canal', 'ID_Pais_Origen', 'Cliente_Disp',
        'h_can_res', 'Agencia_nombre'
    ]
    df.drop(columns=columnas_a_eliminar + columnas_tentativas, inplace=True, errors='ignore')

    for col in df.columns:
        if 'fec' in col.lower():
            df[col] = pd.to_datetime(df[col], errors='coerce')

    df['hay_menores'] = (df['h_num_men'] > 0).astype(bool)
    df.drop(columns=['h_num_men'], inplace=True, errors='ignore')

    df['dif_dias'] = (df['h_fec_sda_ok'] - df['h_fec_lld_ok']).dt.days
    df['dif_dias_ajustada'] = df['dif_dias'].replace(0, 1)
    df['diferencia_noc'] = (df['h_num_noc'] - df['dif_dias_ajustada']).abs()

    return df

def eliminar_outliers(df: pd.DataFrame):
    ID_Eliminar = [97229]
    ids_diferencias = df.loc[df['diferencia_noc'] > 1, 'ID_Reserva'].tolist()
    ID_Eliminar = list(set(ID_Eliminar + ids_diferencias))

    df = df[~df['ID_Reserva'].isin(ID_Eliminar)].copy()
    df.drop(columns=['dif_dias', 'dif_dias_ajustada', 'diferencia_noc'], inplace=True, errors='ignore')
    return df

def crear_categorias_y_separar(df: pd.DataFrame):
    df['h_num_noc_cat'] = pd.cut(
        df['h_num_noc'], bins=[0, 2, 4, 6, 8, float('inf')],
        labels=['0-2', '2-4', '4-6', '6-8', '+8'], right=False
    )
    # df.drop(columns=['h_num_noc'], inplace=True, errors='ignore')

    # Conversión a numérico (por si vienen de CSV)
    df['h_tot_hab'] = pd.to_numeric(df['h_tot_hab'], errors='coerce')
    df['h_num_adu'] = pd.to_numeric(df['h_num_adu'], errors='coerce')

    # Categorización usando pd.cut
    df['h_tot_hab_cat'] = pd.cut(
        df['h_tot_hab'],
        bins=[0, 1, float('inf')],
        labels=['1', '+1'],
        right=True
    )

    df['h_num_adu_cat'] = pd.cut(
        df['h_num_adu'],
        bins=[0, 1, 2, 3, 4, 5, float('inf')],
        labels=['1', '2', '3', '4', '5', '+5'],
        right=True
    )

    # Elimina las columnas originales
    df.drop(columns=['h_tot_hab', 'h_num_adu'], inplace=True, errors='ignore')


    # df.drop(columns=['h_num_adu'], inplace=True, errors='ignore')

    def separar_componentes(valor):
        if pd.isna(valor):
            return pd.Series([None, None, None])
        valor = re.sub(r'\s+', ' ', valor.strip())
        dobles = ['JR SUITE', 'MASTER SUITE', 'MV LUXURY', 'SUP LUJ', 'SUITE PRES', 'SUITE FAMILIAR']
        doble_pattern = r'^(?:' + '|'.join(re.escape(d) for d in dobles) + r')\b'
        match_doble = re.match(doble_pattern, valor)
        if match_doble:
            doble = match_doble.group(0)
            resto = valor[len(doble):].strip()
            if not resto:
                return pd.Series([doble, None, None])
            resto_partes = resto.split(' ', 1)
            if len(resto_partes) == 1 and ('C/' in resto_partes[0] or 'S/' in resto_partes[0]):
                return pd.Series([doble, None, resto_partes[0]])
            componente2 = resto_partes[0] if len(resto_partes) > 0 else None
            componente3 = resto_partes[1] if len(resto_partes) > 1 else None
            return pd.Series([doble, componente2, componente3])
        partes = valor.split()
        if len(partes) == 2 and ('C/' in partes[1] or 'S/' in partes[1]):
            return pd.Series([partes[0], None, partes[1]])
        elif len(partes) == 3 and partes[2] == 'SB':
            return pd.Series([partes[0], f"{partes[1]} {partes[2]}", None])
        else:
            return pd.Series([
                partes[0],
                partes[1] if len(partes) > 1 else None,
                partes[2] if len(partes) > 2 else None
            ])

    df['Tipo_Habitacion_nombre'] = df['Tipo_Habitacion_nombre'].str.strip().str[:-3].str.strip()
    df['Tipo_Habitacion_nombre'] = (
        df['Tipo_Habitacion_nombre']
        .str.replace('SN12', '', regex=False)
        .str.replace('SIN DEFI', '', regex=False)
        .str.replace('AZUL', '', regex=False)
        .str.strip()
        .replace('', None)
    )
    df['Tipo_Habitacion_nombre'] = df['Tipo_Habitacion_nombre'].replace(
        'SUP LUJ ING S/R', 'SUP LUJ KING S/R'
    )
    componentes = df['Tipo_Habitacion_nombre'].apply(separar_componentes)
    componentes.columns = ['Tipo_Habitacion_Nombre', 'Tipo_Habitacion_Camas', 'Tipo_Habitacion_Detalles']
    df.drop(columns=['Tipo_Habitacion_nombre'], inplace=True, errors='ignore')
    df = pd.concat([df, componentes], axis=1)

    orden_columnas = [
        'ID_Reserva', 'Fecha_hoy', 'h_fec_lld_ok', 'h_fec_sda_ok', 'Estado_cve', 'h_tfa_total',
        'Tipo_Habitacion_Nombre', 'Tipo_Habitacion_Camas', 'Tipo_Habitacion_Detalles', 'Clasificacion', 'h_num_noc',
        'h_num_noc_cat', 'h_tot_hab_cat', 'h_num_adu_cat', 'hay_menores'
    ]
    df = df[orden_columnas]

    logger.info("Proceso de limpieza y transformación completado. Registros finales: %d", len(df))
    return df