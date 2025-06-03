from kedro.pipeline import Pipeline

# Importa los pipelines
from clustera.pipelines.data_science import pipeline as data_science_pipeline
from clustera.pipelines.data_processing import pipeline as data_processing_pipeline
from clustera.pipelines.DS_tda_hoy import pipeline as ds_tda_hoy_pipeline
from clustera.pipelines.anticipation_table import pipeline as anticipation_table_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    return {
        "__default__": data_science_pipeline.create_pipeline() + anticipation_table_pipeline.create_pipeline(),
        "data_science": data_science_pipeline.create_pipeline() + ds_tda_hoy_pipeline.create_pipeline(),
        "data_processing": data_processing_pipeline.create_pipeline(),
        "ds_tda_hoy": ds_tda_hoy_pipeline.create_pipeline(),
        "anticipation_table": anticipation_table_pipeline.create_pipeline(),
    }
