from kedro.pipeline import Pipeline, node
from .nodes import train_random_forest_gridsearch

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=train_random_forest_gridsearch,
                inputs=["preprocessed_data_dict", "parameters"],
                outputs=None,
                name="train_rf_gridsearch_node",
            )
        ]
    )