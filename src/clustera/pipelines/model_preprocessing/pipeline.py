from kedro.pipeline import Pipeline, node
from .nodes import preprocess_data

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=preprocess_data,
                inputs=["reservaciones_raw", "parameters"],
                outputs="preprocessed_data_dict",
                name="preprocess_data_node",
            )
        ]
    )