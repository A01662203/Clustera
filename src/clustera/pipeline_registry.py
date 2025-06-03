from kedro.pipeline import Pipeline
from clustera.pipelines.data_science import pipeline as data_science_pipeline

def register_pipelines() -> dict[str, Pipeline]:
    return {
        "__default__": data_science_pipeline.create_pipeline(),
        "data_science": data_science_pipeline.create_pipeline(),
    }