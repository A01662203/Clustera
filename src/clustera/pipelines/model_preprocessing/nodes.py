import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data(raw_data: pd.DataFrame, params: dict) -> dict:
    """
    - Realiza feature engineering de fechas, binning de noches, adultos, menores, etc.
    - Codifica variables categóricas si fuera necesario (o solo pasa el DataFrame limpio).
    - Retorna un diccionario con X_train, X_test, y_train, y_test.
    """

    # Ejemplo simplificado:
    df = raw_data.copy()

    # 1. Selección de columnas relevantes
    columns_to_keep = [
        "h_num_adu_cat"
    ]

    # 2. Definir X e y
    target = "tipo_habitacion"
    X = df.drop(columns=[target])
    y = df[target]

    # 3. Train-test split
    test_size = 1 - params["data"]["train_split"]
    random_state = params["data"]["random_seed"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }