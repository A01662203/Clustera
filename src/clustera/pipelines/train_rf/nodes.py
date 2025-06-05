import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

def train_random_forest_gridsearch(preprocessed_data: dict, params: dict) -> None:
    """
    Entrena un RandomForest con búsqueda de hiperparámetros y loggea resultados.
    """
    X_train = preprocessed_data["X_train"]
    X_test = preprocessed_data["X_test"]
    y_train = preprocessed_data["y_train"]
    y_test = preprocessed_data["y_test"]

    # Espacio de búsqueda de hiperparámetros
    param_grid = params["rf_grid"]

    # RandomForest base
    base_model = RandomForestClassifier(random_state=params["data"]["random_seed"])

    # GridSearchCV
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=3,
        n_jobs=-1,
        verbose=2
    )

    # Activar autolog de MLflow
    mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True)

    with mlflow.start_run(run_name="RF_GridSearch_train") as run:
        grid_search.fit(X_train, y_train)

        # Mejor modelo encontrado
        best_model = grid_search.best_estimator_

        # Loggear hiperparámetros óptimos como tags/params en MLflow
        mlflow.log_params(grid_search.best_params_)

        # Predicción final
        preds = best_model.predict(X_test)
        report = classification_report(y_test, preds, output_dict=True)

        # Loggear métricas
        for label, metrics in report.items():
            if label in ["accuracy", "macro avg", "weighted avg"]:
                mlflow.log_metric(f"rf_{label}", metrics["f1-score"])
            else:
                mlflow.log_metric(f"rf_{label}_f1", metrics["f1-score"])
                mlflow.log_metric(f"rf_{label}_precision", metrics["precision"])
                mlflow.log_metric(f"rf_{label}_recall", metrics["recall"])

        # (Opcional) loggear el mejor modelo explícitamente como artefacto
        mlflow.sklearn.log_model(best_model, "best_rf_model")

        # Registrar el modelo en el registry
        mlflow.register_model(
            f"runs:/{run.info.run_id}/best_rf_model",
            "random_forest_model"
        )

        # Mostrar por consola el mejor score y parámetros
        print("Mejor F1 score promedio (macro):", grid_search.best_score_)
        print("Mejores hiperparámetros:", grid_search.best_params_)

        