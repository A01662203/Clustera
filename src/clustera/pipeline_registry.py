from kedro.pipeline import Pipeline

# Importa aquí todos los pipelines que vayas a registrar:
from clustera.pipelines.data_processing import pipeline as data_processing_pipeline
from clustera.pipelines.data_science import pipeline as data_science_pipeline
from clustera.pipelines.anticipation_table import pipeline as anticipation_table_pipeline
from clustera.pipelines.Anom_hoy import pipeline as Anom_hoy_pipeline
from clustera.pipelines.Anom_lleg import pipeline as Anom_lleg_pipeline
from clustera.pipelines.model_preprocessing import pipeline as model_preprocessing_pipeline
from clustera.pipelines.train_rf import pipeline as train_rf_pipeline
from clustera.pipelines.train_xgb import pipeline as train_xgb_pipeline
from clustera.pipelines.MBA import pipeline as MBA_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    return {
        # Pipeline por defecto: aquí decides si incluyes uno o más pipelines concatenados
        "__default__": data_processing_pipeline.create_pipeline()
                       + anticipation_table_pipeline.create_pipeline()
                       + data_science_pipeline.create_pipeline()
                       + Anom_hoy_pipeline.create_pipeline()
                       + Anom_lleg_pipeline.create_pipeline()
                       + model_preprocessing_pipeline.create_pipeline()
                       + train_rf_pipeline.create_pipeline()
                       + train_xgb_pipeline.create_pipeline(),
        
        # Si necesitas agrupar pipelines en uno “data_science”
        "data_science": (
            data_science_pipeline.create_pipeline()
            + Anom_hoy_pipeline.create_pipeline()
            + Anom_lleg_pipeline.create_pipeline()
        ),
        
        # Pipeline de solo “data_processing” por separado
        "data_processing": data_processing_pipeline.create_pipeline(),
        
        # Pipeline de solo “anticipation_table”
        "anticipation_table": anticipation_table_pipeline.create_pipeline(),
        
        # Pipeline de solo “anomaly_detection_hoy”
        "anomaly_detection_hoy": Anom_hoy_pipeline.create_pipeline(),
        
        # Pipeline de solo “anomaly_detection_lleg”
        "anomaly_detection_lleg": Anom_lleg_pipeline.create_pipeline(),
        
        # Pipeline de solo “mba”
        "mba": MBA_pipeline.create_pipeline(),

        # Pipeline de model_preprocessing
        "models_preprocessing": model_preprocessing_pipeline.create_pipeline(),

        # Pipeline de Random Forest
        "train_rf": model_preprocessing_pipeline.create_pipeline() + train_rf_pipeline.create_pipeline(),

        # Pipeline de XGBoost
        "train_xgb": model_preprocessing_pipeline.create_pipeline() + train_xgb_pipeline.create_pipeline(),

        # Pipeline de entrenamiento conjunto
        "train_both": model_preprocessing_pipeline.create_pipeline() + train_rf_pipeline.create_pipeline() + train_xgb_pipeline.create_pipeline(),
    }