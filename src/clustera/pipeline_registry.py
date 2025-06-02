from kedro.pipeline import Pipeline
from clustera.pipelines.data_processing import pipeline as data_processing_pipeline

def register_pipelines() -> dict[str, Pipeline]:
    return {
        "__default__": data_processing_pipeline.create_pipeline(),
        "data_processing": data_processing_pipeline.create_pipeline(),
    }