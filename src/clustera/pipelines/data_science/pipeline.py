from kedro.pipeline import Pipeline, node, pipeline
from clustera.pipelines.data_science.nodes import entrenar_kmeans, entrenar_kmeans_v2

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        # Nodo 1
        node(
            func=entrenar_kmeans,
            inputs="reservaciones_finales",
            outputs="clustered_reservaciones_segmentadas",
            name="entrenar_kmeans_node"
        ),
        # Nodo 2
        node(
            func=entrenar_kmeans_v2,
            inputs="reservaciones_finales",
            outputs="clustered_reservaciones_segmentadas_v2",
            name="entrenar_kmeans_v2_node"
        )
    ])
