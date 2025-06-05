from kedro.pipeline import Pipeline, node
from .nodes import train_xgboost_gridsearch

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=train_xgboost_gridsearch,
                inputs=["preprocessed_data_dict", "parameters"],
                outputs=None,
                name="train_xgb_gridsearch_node",
            )
        ]
    )
