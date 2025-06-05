import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

def train_xgboost_gridsearch(preprocessed_data: dict, params: dict) -> None:
    """
    Entrena un XGBoostClassifier con GridSearch y loggea resultados.
    """
    X_train = preprocessed_data["X_train"]
    X_test = preprocessed_data["X_test"]
    y_train = preprocessed_data["y_train"]
    y_test = preprocessed_data["y_test"]

    # Espacio de búsqueda de hiperparámetros
    param_grid = params["xgb_grid"]

    # Modelo base de XGBoost
    base_model = XGBClassifier(
        random_state=params["data"]["random_seed"],
        use_label_encoder=False,
        eval_metric="mlogloss"
    )

    # GridSearch
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=3,
        n_jobs=-1,
        verbose=2
    )

    # Activar autolog de MLflow para XGBoost
    mlflow.xgboost.autolog(log_input_examples=True, log_model_signatures=True)

    with mlflow.start_run(run_name="XGB_GridSearch_train") as run:
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_

        # Loggear los mejores hiperparámetros explícitamente
        mlflow.log_params(grid_search.best_params_)

        # Evaluar
        preds = best_model.predict(X_test)
        report = classification_report(y_test, preds, output_dict=True)

        # Loggear métricas detalladas
        for label, metrics in report.items():
            if label in ["accuracy", "macro avg", "weighted avg"]:
                mlflow.log_metric(f"xgb_{label}", metrics["f1-score"])
            else:
                mlflow.log_metric(f"xgb_{label}_f1", metrics["f1-score"])
                mlflow.log_metric(f"xgb_{label}_precision", metrics["precision"])
                mlflow.log_metric(f"xgb_{label}_recall", metrics["recall"])

        # Loggear modelo final
        mlflow.xgboost.log_model(best_model, "best_xgb_model")

        # Registrar en Model Registry
        mlflow.register_model(
            f"runs:/{run.info.run_id}/best_xgb_model",
            "xgboost_model"
        )

        print("Mejor F1 score promedio (macro):", grid_search.best_score_)
        print("Mejores hiperparámetros:", grid_search.best_params_)