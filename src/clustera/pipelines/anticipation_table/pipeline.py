from kedro.pipeline import Pipeline, node, pipeline
from .nodes import calcular_tabla_anticipacion

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=calcular_tabla_anticipacion,
                inputs="reservaciones_finales",
                outputs=None,
                name="calcular_tabla_anticipacion_node",
            ),
        ]
    )