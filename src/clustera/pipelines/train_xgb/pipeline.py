from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    merge_reservaciones,
    filtrar_reservaciones,
    limpieza_basica,
    eliminar_outliers,
    crear_categorias_y_separar
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=merge_reservaciones,
            inputs=dict(
                iar_Reservaciones="iar_Reservaciones",
                iar_paquetes="iar_paquetes",
                iar_Agencias="iar_Agencias",
                iar_Tipos_Habitaciones="iar_Tipos_Habitaciones"
            ),
            outputs="merged_reservaciones",
            name="merge_reservaciones_node"
        ),
        node(
            func=filtrar_reservaciones,
            inputs="merged_reservaciones",
            outputs="reservaciones_filtradas",
            name="filtrar_reservaciones_node"
        ),
        node(
            func=limpieza_basica,
            inputs="reservaciones_filtradas",
            outputs="reservaciones_limpias",
            name="limpieza_basica_node"
        ),
        node(
            func=eliminar_outliers,
            inputs="reservaciones_limpias",
            outputs="reservaciones_sin_outliers",
            name="eliminar_outliers_node"
        ),
        node(
            func=crear_categorias_y_separar,
            inputs="reservaciones_sin_outliers",
            outputs="reservaciones_finales",
            name="crear_categorias_y_separar_node"
        )
    ])
