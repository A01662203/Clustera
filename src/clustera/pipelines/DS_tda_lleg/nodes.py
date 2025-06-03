"""
Nodos para análisis TDA (Topological Data Analysis) de tarifas hoteleras
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Importar librerías de Giotto-TDA
from gtda.time_series import TakensEmbedding
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceEntropy, Amplitude, NumberOfPoints
from gtda.plotting import plot_diagram, plot_point_cloud
from sklearn.pipeline import Pipeline
from sklearn.metrics import pairwise_distances

import matplotlib.dates as mdates
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def clean_persistence_diagram(pd_raw):
    """Limpia y valida un diagrama de persistencia"""
    if len(pd_raw) == 0:
        return None
    
    # Eliminar puntos con birth = death (puntos degenerados)
    valid_mask = pd_raw[:, 1] > pd_raw[:, 0]  # death > birth
    pd_clean = pd_raw[valid_mask]
    
    if len(pd_clean) == 0:
        return None
    
    # Eliminar puntos con valores infinitos
    finite_mask = np.isfinite(pd_clean[:, 0]) & np.isfinite(pd_clean[:, 1])
    pd_clean = pd_clean[finite_mask]
    
    if len(pd_clean) == 0:
        return None
        
    return pd_clean


def normalize_diagrams_by_dimension(diagrams, names):
    """
    Normaliza diagramas para que tengan el mismo número de puntos por dimensión homológica
    """
    
    # Separar por dimensión homológica
    dims_data = {}  # dim -> lista de arrays de puntos
    
    for i, (diag, name) in enumerate(zip(diagrams, names)):
        for dim in [0, 1, 2]:
            dim_mask = diag[:, 2] == dim
            dim_points = diag[dim_mask]
            
            if dim not in dims_data:
                dims_data[dim] = []
            
            dims_data[dim].append({
                'points': dim_points,
                'diagram_idx': i,
                'name': name
            })
    
    # Encontrar número máximo de puntos por dimensión
    max_points_per_dim = {}
    for dim in dims_data:
        max_points = max([len(d['points']) for d in dims_data[dim]])
        max_points_per_dim[dim] = max_points

    # Crear diagramas normalizados
    normalized_diagrams = []
    
    for i, name in enumerate(names):
        diagram_parts = []
        
        for dim in [0, 1, 2]:
            if dim in dims_data:
                # Encontrar los puntos de esta dimensión para este diagrama
                dim_points = None
                for d in dims_data[dim]:
                    if d['diagram_idx'] == i:
                        dim_points = d['points']
                        break
                
                if dim_points is None or len(dim_points) == 0:
                    # No hay puntos en esta dimensión, crear puntos triviales
                    dim_points = np.array([[0, 0, dim]])
                
                # Pad o truncar para tener exactamente max_points_per_dim[dim] puntos
                target_size = max_points_per_dim[dim]
                current_size = len(dim_points)
                
                if current_size < target_size:
                    # Pad con puntos triviales (birth=death=0)
                    padding = np.zeros((target_size - current_size, 3))
                    padding[:, 2] = dim  # Asignar dimensión correcta
                    dim_points_normalized = np.vstack([dim_points, padding])
                elif current_size > target_size:
                    # Truncar (tomar los puntos con mayor persistencia)
                    lifetimes = dim_points[:, 1] - dim_points[:, 0]
                    top_indices = np.argpartition(lifetimes, -target_size)[-target_size:]
                    dim_points_normalized = dim_points[top_indices]
                else:
                    dim_points_normalized = dim_points
                
                diagram_parts.append(dim_points_normalized)
        
        # Concatenar todas las dimensiones
        if diagram_parts:
            normalized_diagram = np.vstack(diagram_parts)
            normalized_diagrams.append(normalized_diagram)
        else:
            # Diagrama vacío, crear uno mínimo
            empty_diagram = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2]])
            normalized_diagrams.append(empty_diagram)
    
    return normalized_diagrams, max_points_per_dim


def plot_all_time_series_with_anomalies(resultados_tda, figsize=(20, 15), output_path=None):
    """
    Grafica todas las series de tiempo con anomalías topológicas marcadas
    """
    
    # Obtener habitaciones válidas
    habitaciones_validas = []
    for hab in resultados_tda.keys():
        if isinstance(resultados_tda[hab], dict) and 'serie_original' in resultados_tda[hab]:
            habitaciones_validas.append(hab)
    
    if not habitaciones_validas:
        logger.warning("No se encontraron habitaciones válidas para graficar")
        return None
   
    # Calcular número de subplots
    n_habitaciones = len(habitaciones_validas)
    n_cols = 2  # Dos columnas
    n_rows = (n_habitaciones + n_cols - 1) // n_cols  # Redondear hacia arriba
    
    # Crear figura
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.suptitle('Series de Tiempo con Anomalías Topológicas Detectadas en Fechas de Llegada', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Manejar el caso de una sola fila
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, habitacion in enumerate(habitaciones_validas):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        try:
            # Obtener datos de la habitación
            data = resultados_tda[habitacion]
            
            # Datos de la serie temporal
            fechas = data['fechas']
            serie = data['serie_original']
            
            # Convertir fechas a datetime si es necesario
            if isinstance(fechas[0], str):
                fechas = pd.to_datetime(fechas)
            
            # Graficar serie temporal principal
            ax.plot(fechas, serie, linewidth=1.5, color='steelblue', alpha=0.8)
            
            # Contador de anomalías para esta habitación
            anomalias_count = 0
            
            # Marcar anomalías topológicas si existen
            if 'anomalias_detectadas' in data and data['anomalias_detectadas']:
                anomalias = data['anomalias_detectadas']
                
                # Extraer fechas y valores de anomalías
                fechas_anomalias = []
                valores_anomalias = []
                
                for anomalia in anomalias:
                    fecha_anom = anomalia['fecha']
                    
                    # Convertir fecha si es necesario
                    if isinstance(fecha_anom, str):
                        fecha_anom = pd.to_datetime(fecha_anom)
                    
                    # Encontrar el valor correspondiente en la serie
                    # Buscar la fecha más cercana
                    if isinstance(fechas[0], str):
                        fechas_dt = pd.to_datetime(fechas)
                    else:
                        fechas_dt = fechas
                    
                    # Encontrar índice de fecha más cercana
                    idx_cercano = np.argmin(np.abs(fechas_dt - fecha_anom))
                    valor_anomalia = serie[idx_cercano]
                    
                    fechas_anomalias.append(fecha_anom)
                    valores_anomalias.append(valor_anomalia)
                
                # Marcar anomalías con X rojas
                if fechas_anomalias:
                    ax.scatter(fechas_anomalias, valores_anomalias, 
                             marker='x', s=100, color='red', linewidth=3,
                             label=f'Anomalías ({len(fechas_anomalias)})', zorder=5)
                    anomalias_count = len(fechas_anomalias)
            
            # Configurar el subplot
            ax.set_title(f'{habitacion}\n({len(serie)} observaciones, {anomalias_count} anomalías)', 
                        fontsize=10, fontweight='bold')
            ax.set_xlabel('Fecha', fontsize=8)
            ax.set_ylabel('Tarifa Promedio', fontsize=8)
            
            # Formatear fechas en el eje x
            if len(fechas) > 50:
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            elif len(fechas) > 20:
                ax.xaxis.set_major_locator(mdates.MonthLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%y'))
            else:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            
            # Rotar etiquetas de fecha
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # Añadir grid sutil
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # Configurar límites y
            y_min, y_max = ax.get_ylim()
            y_range = y_max - y_min
            ax.set_ylim(y_min - 0.05*y_range, y_max + 0.05*y_range)
            
            # Añadir leyenda solo si hay anomalías
            if anomalias_count > 0:
                ax.legend(loc='upper right', fontsize=8)
            
            # Estadísticas básicas como texto
            mean_val = np.mean(serie)
            std_val = np.std(serie)
            ax.text(0.02, 0.95, f'μ: {mean_val:.1f}\nσ: {std_val:.1f}', 
                   transform=ax.transAxes, fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
            
        except Exception as e:
            # En caso de error, mostrar mensaje en el subplot
            ax.text(0.5, 0.5, f'Error al graficar\n{habitacion}\n{str(e)}', 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=10, color='red')
            ax.set_title(f'{habitacion} (Error)', fontsize=10, color='red')
    
    # Ocultar subplots vacíos
    total_subplots = n_rows * n_cols
    for idx in range(n_habitaciones, total_subplots):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)
    
    # Ajustar layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, hspace=0.4, wspace=0.3)
    
    # Guardar la imagen
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"Gráfico guardado en: {output_path}")
    
    plt.close()  # Cerrar la figura para liberar memoria
    
    return fig


def analisis_tda_completo(reservaciones_finales: pd.DataFrame, parameters: dict) -> str:
    """
    Ejecuta el análisis TDA completo y genera la visualización
    
    Args:
        reservaciones_finales: DataFrame con los datos de reservaciones procesados
        parameters: Parámetros de configuración del análisis TDA
    
    Returns:
        str: Ruta del archivo de imagen generado
    """
    
    logger.info("Iniciando análisis TDA completo")
    
    # Preparar datos
    df = reservaciones_finales.copy()
    
    # Verificar que existan las columnas necesarias
    required_columns = ['Fecha_hoy', 'h_num_noc', 'h_tfa_total', 'Tipo_Habitacion_Nombre']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Columnas faltantes en el DataFrame: {missing_columns}")
    
    # Convertir fecha
    df['h_fec_lld_ok'] = pd.to_datetime(df['h_fec_lld_ok'])
    
    # Eliminar registros con noches = 0
    df = df[df['h_num_noc'] != 0]
    
    # Calcular tarifa promedio por noche
    df['Tarifa_promedio'] = df['h_tfa_total'] / df['h_num_noc']
    
    # Agrupa por fecha y tipo de habitación
    serie = df.groupby(['h_fec_lld_ok', 'Tipo_Habitacion_Nombre'])['Tarifa_promedio'].mean().reset_index()
    serie.columns = ['Fecha', 'Habitacion', 'Tarifa']
    serie = serie.sort_values(['Fecha', 'Habitacion'])
    
    # Obtener tipos de habitación únicos y ordenarlos alfabéticamente
    habitaciones = sorted(serie['Habitacion'].unique())
    
    logger.info(f"Analizando {len(habitaciones)} tipos de habitación")
    
    # Diccionario para almacenar resultados
    resultados_tda = {}
    
    # Configurar Takens Embedding
    takens_embedder = TakensEmbedding(
        time_delay=parameters.get('time_delay', 3),
        dimension=parameters.get('dimension', 5),
        stride=parameters.get('stride', 1)
    )
    
    # Procesar cada tipo de habitación
    for habitacion in habitaciones:
        logger.info(f"Procesando habitación: {habitacion}")
        
        # Filtrar datos para esta habitación
        subset = serie[serie['Habitacion'] == habitacion].copy()
        subset = subset.sort_values('Fecha')
        
        # Preparar serie temporal
        tarifa_series = subset['Tarifa'].values.reshape(-1, 1)
        
        # Normalizar datos
        scaler = StandardScaler()
        tarifa_normalized = scaler.fit_transform(tarifa_series)
        
        # Aplicar Takens Embedding
        try:
            embedded_data = takens_embedder.fit_transform([tarifa_normalized.flatten()])
            
            # Almacenar resultados
            resultados_tda[habitacion] = {
                'serie_original': tarifa_series.flatten(),
                'serie_normalizada': tarifa_normalized.flatten(),
                'embedding': embedded_data[0],
                'fechas': subset['Fecha'].values,
                'scaler': scaler
            }
            
        except Exception as e:
            logger.warning(f"Error en embedding para {habitacion}: {e}")
            continue
    
    # Calcular persistencia homológica
    vr_persistence = VietorisRipsPersistence(
        homology_dimensions=parameters.get('homology_dimensions', [0, 1, 2]),
        max_edge_length=np.inf,
        n_jobs=1
    )
    
    # Calcular persistencia para cada habitación
    for habitacion in resultados_tda.keys():
        logger.info(f"Calculando persistencia para: {habitacion}")
        
        try:
            embedding = resultados_tda[habitacion]['embedding']
            
            # Calcular diagrama de persistencia
            persistence_diagrams = vr_persistence.fit_transform([embedding])
            
            resultados_tda[habitacion]['persistence_diagram'] = persistence_diagrams[0]
            
        except Exception as e:
            logger.warning(f"Error en persistencia para {habitacion}: {e}")
    
    # Calcular métricas topológicas
    entropy_calculator = PersistenceEntropy()
    amplitude_calculator = Amplitude(metric='landscape')
    
    # Obtener habitaciones válidas
    habitaciones_validas = []
    for hab in resultados_tda.keys():
        if 'persistence_diagram' in resultados_tda[hab]:
            habitaciones_validas.append(hab)
    
    # Calcular métricas para cada habitación
    for habitacion in habitaciones_validas:
        logger.info(f"Calculando métricas para: {habitacion}")
        
        try:
            # Obtener y limpiar diagrama
            pd_original = resultados_tda[habitacion]['persistence_diagram']
            pd_clean = clean_persistence_diagram(pd_original)
            
            if pd_clean is None:
                continue
            
            # Separar por dimensión homológica
            dims = pd_clean[:, 2].astype(int)
            unique_dims = np.unique(dims)
            
            # Crear métricas por separado para cada dimensión
            metrics_by_dim = {}
            
            for dim in unique_dims:
                dim_mask = dims == dim
                pd_dim = pd_clean[dim_mask]
                
                if len(pd_dim) == 0:
                    continue
                
                # Formato correcto para giotto-tda: [n_samples, n_points, 3]
                pd_array = pd_dim.reshape(1, -1, 3)
                
                try:
                    # Calcular entropía
                    entropy = entropy_calculator.fit_transform(pd_array)
                    entropy_val = float(entropy[0][0]) if entropy.ndim > 1 else float(entropy[0])
                    
                    # Calcular amplitud
                    amplitude = amplitude_calculator.fit_transform(pd_array)
                    amplitude_val = float(amplitude[0][0]) if amplitude.ndim > 1 else float(amplitude[0])
                    
                    metrics_by_dim[f'H{dim}'] = {
                        'entropy': entropy_val,
                        'amplitude': amplitude_val,
                        'n_points': len(pd_dim),
                        'lifetime_mean': np.mean(pd_dim[:, 1] - pd_dim[:, 0]),
                        'lifetime_std': np.std(pd_dim[:, 1] - pd_dim[:, 0])
                    }
                    
                except Exception as e:
                    logger.warning(f"Error en métricas H{dim} para {habitacion}: {e}")
                    metrics_by_dim[f'H{dim}'] = {
                        'entropy': np.nan,
                        'amplitude': np.nan,
                        'n_points': len(pd_dim),
                        'lifetime_mean': np.mean(pd_dim[:, 1] - pd_dim[:, 0]),
                        'lifetime_std': np.std(pd_dim[:, 1] - pd_dim[:, 0])
                    }
            
            # Almacenar métricas consolidadas
            if metrics_by_dim:
                # Métricas agregadas (promedio ponderado por número de puntos)
                total_points = sum([metrics_by_dim[k]['n_points'] for k in metrics_by_dim.keys()])
                
                if total_points > 0:
                    valid_entropies = [metrics_by_dim[k]['entropy'] * metrics_by_dim[k]['n_points'] 
                                    for k in metrics_by_dim.keys() if not np.isnan(metrics_by_dim[k]['entropy'])]
                    valid_amplitudes = [metrics_by_dim[k]['amplitude'] * metrics_by_dim[k]['n_points'] 
                                      for k in metrics_by_dim.keys() if not np.isnan(metrics_by_dim[k]['amplitude'])]
                    
                    avg_entropy = sum(valid_entropies) / total_points if valid_entropies else np.nan
                    avg_amplitude = sum(valid_amplitudes) / total_points if valid_amplitudes else np.nan
                    
                    resultados_tda[habitacion]['topological_metrics'] = {
                        'entropy_total': avg_entropy,
                        'amplitude_total': avg_amplitude,
                        'n_points_total': total_points,
                        'by_dimension': metrics_by_dim,
                        'diagram_clean': pd_clean
                    }
            
        except Exception as e:
            logger.warning(f"Error general en métricas para {habitacion}: {e}")
    
    # Calcular indicadores temporales para detección de anomalías
    window_size = parameters.get('window_size', 20)
    
    for habitacion in habitaciones_validas:
        logger.info(f"Calculando indicadores temporales para: {habitacion}")
        
        try:
            embedding = resultados_tda[habitacion]['embedding']
            fechas = resultados_tda[habitacion]['fechas']
            
            # Calcular indicadores en ventanas deslizantes
            window_size_hab = min(window_size, len(embedding) // 3)
            
            if window_size_hab < 5:
                continue
            
            indicadores_temporales = []
            
            for i in range(window_size_hab, len(embedding)):
                try:
                    ventana = embedding[i-window_size_hab:i]
                    
                    # Calcular persistencia para ventana
                    pd_ventana = vr_persistence.fit_transform([ventana])
                    pd_clean_ventana = clean_persistence_diagram(pd_ventana[0])
                    
                    if pd_clean_ventana is not None and len(pd_clean_ventana) > 0:
                        # Calcular métricas para la ventana
                        pd_array = pd_clean_ventana.reshape(1, -1, 3)
                        
                        entropy_ventana = entropy_calculator.fit_transform(pd_array)
                        amplitude_ventana = amplitude_calculator.fit_transform(pd_array)
                        
                        entropy_val = float(entropy_ventana[0][0]) if entropy_ventana.ndim > 1 else float(entropy_ventana[0])
                        amplitude_val = float(amplitude_ventana[0][0]) if amplitude_ventana.ndim > 1 else float(amplitude_ventana[0])
                        
                        indicadores_temporales.append({
                            'fecha': fechas[i],
                            'entropy': entropy_val,
                            'amplitude': amplitude_val,
                            'n_points': len(pd_clean_ventana),
                            'ventana_end': i
                        })
                        
                except Exception as e_ventana:
                    # Si falla una ventana, continuar con la siguiente
                    continue
            
            if indicadores_temporales:
                resultados_tda[habitacion]['indicadores_temporales'] = indicadores_temporales
                
                # Detectar posibles anomalías
                if len(indicadores_temporales) > 5:
                    entropias = [ind['entropy'] for ind in indicadores_temporales]
                    amplitudes = [ind['amplitude'] for ind in indicadores_temporales]
                    
                    # Usar percentiles para detección de anomalías
                    entropy_q95 = np.percentile(entropias, 95)
                    amplitude_q95 = np.percentile(amplitudes, 95)
                    entropy_q05 = np.percentile(entropias, 5)
                    amplitude_q05 = np.percentile(amplitudes, 5)
                    
                    anomalias_detectadas = []
                    for ind in indicadores_temporales:
                        # Detectar valores extremos (muy altos o muy bajos)
                        is_anomaly = (ind['entropy'] > entropy_q95 or ind['entropy'] < entropy_q05 or
                                    ind['amplitude'] > amplitude_q95 or ind['amplitude'] < amplitude_q05)
                        if is_anomaly:
                            anomalias_detectadas.append({
                                'fecha': ind['fecha'],
                                'entropy': ind['entropy'],
                                'amplitude': ind['amplitude'],
                                'tipo': 'extremo_topologico'
                            })
                    
                    resultados_tda[habitacion]['anomalias_detectadas'] = anomalias_detectadas
                    logger.info(f"Detectadas {len(anomalias_detectadas)} anomalías en {habitacion}")
            
        except Exception as e:
            logger.warning(f"Error en indicadores temporales para {habitacion}: {e}")
    
    # Generar visualización
    figsize = parameters.get('figsize', (20, 15))
    fig = plot_all_time_series_with_anomalies(
        resultados_tda, 
        figsize=figsize
    )

    logger.info("Análisis TDA completado exitosamente")
    return fig