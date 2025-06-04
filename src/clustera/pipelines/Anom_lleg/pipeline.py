from kedro.pipeline import Pipeline, node, pipeline
from .nodes import detectar_anomalias_y_retornar_figura

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=detectar_anomalias_y_retornar_figura,
            inputs=dict(
                df="reservaciones_finales",
                api_key="params:nixtla.nixtla_api_key"
            ),
            outputs="nixtla_plot_path_lleg",
            name="TimeGPT_anomalies_lleg"
        )
    ])
