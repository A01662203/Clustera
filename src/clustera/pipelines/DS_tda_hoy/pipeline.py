from kedro.pipeline import Pipeline, node
from .nodes import analisis_tda_completo

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=analisis_tda_completo,
            inputs=["reservaciones_finales", "params:parameters_tda"],
            outputs="tda_analysis_plot",
            name="tda_analysis_node",
        ),
    ])