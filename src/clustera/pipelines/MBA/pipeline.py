from kedro.pipeline import Pipeline, node, pipeline
from clustera.pipelines.MBA.nodes import create_itemsets, generate_rules_by_cluster


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        # Nodo 1: Crear itemsets para MBA
        node(
            func=create_itemsets,
            inputs="reservaciones_con_cluster",
            outputs="reservaciones_itemsets",
            name="create_itemsets_node"
        ),
        # Nodo 2: Generar reglas de asociación por cluster
        node(
            func=generate_rules_by_cluster,
            inputs=[
                "reservaciones_itemsets",
                "params:min_support",
                "params:min_confidence",
                "params:min_lift"
            ],
            outputs="mba_reglas_por_cluster",
            name="generate_mba_rules_node"
        )
    ])