from kedro.pipeline import Pipeline, node, pipeline
from .nodes import entrenar_kmeans

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=entrenar_kmeans,
            inputs="reservaciones_finales",
            outputs="clustered_reservaciones_segmentadas",  # 👈 este nombre debe coincidir
            name="entrenar_kmeans_node"
        )
    ])
