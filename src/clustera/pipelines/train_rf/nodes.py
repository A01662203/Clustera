import joblib
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import mlflow

def train_random_forest_gridsearch(preprocessed_data: dict, params: dict) -> None:
    X_train = preprocessed_data["X_train"]
    X_test = preprocessed_data["X_test"]
    y_train = preprocessed_data["y_train"]
    y_test = preprocessed_data["y_test"]

    # Variables categóricas, numéricas y booleanas
    categorical_features = ["Estado_cve", "h_num_adu_cat", "h_num_noc_cat"]
    numeric_features = ["meses_anticipacion", "dias_anticipacion",
                        "mes_llegada_sin", "mes_llegada_cos", "mes_rsv_sin", "mes_rsv_cos",
                        "num_sem_llegada", "num_sem_rsv"]
    passthrough_features = ["hay_menores", "is_weekend_reserva", "is_weekend_llegada"]

    # Preprocesador
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), categorical_features),
            ("passthrough", "passthrough", passthrough_features)
        ]
    )

    # Ajusta el preprocesador SOLO en entrenamiento
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # Aplica SMOTE SOLO en entrenamiento
    smote = SMOTE(random_state=params["data"]["random_seed"])
    X_train_bal, y_train_bal = smote.fit_resample(X_train_proc, y_train)

    # Modelo
    rf = RandomForestClassifier(random_state=params["data"]["random_seed"])

    # GridSearchCV
    param_grid = params["rf_grid"]
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=3,
        n_jobs=-1,
        verbose=2
    )

    # MLflow
    mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True)
    with mlflow.start_run(run_name="RF_GridSearch_SMOTE_train", nested=True) as run:
        grid_search.fit(X_train_bal, y_train_bal)

        best_model = grid_search.best_estimator_
        mlflow.log_params(grid_search.best_params_)

        # Métricas en test
        preds = best_model.predict(X_test_proc)
        report = classification_report(y_test, preds, output_dict=True)

        for label, metrics in report.items():
            if label == "accuracy":
                mlflow.log_metric(f"rf_{label}", metrics)
            else:
                mlflow.log_metric(f"rf_{label}_f1", metrics["f1-score"])
                mlflow.log_metric(f"rf_{label}_precision", metrics["precision"])
                mlflow.log_metric(f"rf_{label}_recall", metrics["recall"])

        # 💡 Guardar modelo final con joblib comprimido
        joblib.dump(best_model, "best_rf_model_smote.joblib", compress=3)

        # Registrar el mejor modelo
        mlflow.sklearn.log_model(best_model, "best_rf_model_smote")
        mlflow.register_model(
            f"runs:/{run.info.run_id}/best_rf_model_smote",
            "random_forest_model_smote"
        )

        print("Mejor F1 macro con SMOTE:", grid_search.best_score_)
        print("Mejores hiperparámetros:", grid_search.best_params_)