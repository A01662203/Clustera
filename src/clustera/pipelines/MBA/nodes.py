import pandas as pd
from typing import List, Dict, Any
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

def create_itemsets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega una columna 'items' al DataFrame, codificando cada fila como un set de ítems:
      - "ADU_{n}"       donde n = valor de 'h_num_adu_cat'
      - "MEN_SI"/"MEN_NO" según 'hay_menores'
      - "NOC_{k}"       donde k = valor de 'h_num_noc_cat'
      - "TIPO_{nombre}" donde nombre = valor de 'Tipo_Habitacion_Nombre'
      - "DET_{detalle}" donde detalle = valor de 'Tipo_Habitacion_Detalles'
    Asume que el DataFrame tiene las columnas:
      'h_num_adu_cat', 'hay_menores', 'h_num_noc_cat',
      'Tipo_Habitacion_Nombre', 'Tipo_Habitacion_Detalles', 'cluster'
    Retorna un nuevo DataFrame con la columna 'items'.
    """
    def _fila_a_items(row: pd.Series) -> set:
        items = set()
        # Ítem número de adultos (categórico, p.ej. "1", "2", "+5")
        items.add(f"ADU_{row['h_num_adu_cat']}")
        # Ítem presencia de menores
        if bool(row["hay_menores"]):
            items.add("MEN_SI")
        else:
            items.add("MEN_NO")
        # Ítem número de noches (categórico, p.ej. "0-2", "2-4", "8+")
        items.add(f"NOC_{row['h_num_noc_cat']}")
        # Ítem tipo de habitación
        items.add(f"TIPO_{row['Tipo_Habitacion_Nombre']}")
        # Ítem detalle (puede ser NaN, "C/BALCON", "S/REAL", "S/R" o faltante)
        detalle = row.get("Tipo_Habitacion_Detalles")
        if pd.isna(detalle) or detalle == "":
            items.add("DET_None")
        else:
            items.add(f"DET_{detalle}")
        return items

    df_copy = df.copy()
    df_copy["items"] = df_copy.apply(_fila_a_items, axis=1)
    return df_copy


def _transacciones_a_onehot(itemsets: List[set]) -> pd.DataFrame:
    """
    Convierte una lista de sets de ítems en un DataFrame one-hot para Apriori.
    """
    te = TransactionEncoder()
    te_ary = te.fit(itemsets).transform(itemsets)
    return pd.DataFrame(te_ary, columns=te.columns_)


def generate_rules_by_cluster(
    df: pd.DataFrame,
    min_support: float,
    min_confidence: float,
    min_lift: float
) -> pd.DataFrame:
    """
    Genera reglas de asociación por cada valor de 'cluster' en el DataFrame.
    Filtra reglas cuyos antecedents sean solo de tipo {"ADU_*", "MEN_*", "NOC_*"}
    y cuyos consequents sean solo de tipo {"TIPO_*", "DET_*"}, además de cumplir min_lift.
    Convierte 'antecedents' y 'consequents' a listas para que puedan guardarse en Parquet.
    """
    from mlxtend.preprocessing import TransactionEncoder
    from mlxtend.frequent_patterns import apriori, association_rules

    reglas_acumuladas: list[dict[str, object]] = []

    def _es_regla_relevante(rule: pd.Series) -> bool:
        for item in rule["antecedents"]:
            if not (item.startswith("ADU_") or item.startswith("MEN_") or item.startswith("NOC_")):
                return False
        for item in rule["consequents"]:
            if not (item.startswith("TIPO_") or item.startswith("DET_")):
                return False
        return rule["lift"] >= min_lift

    def _transacciones_a_onehot(itemsets: list[set]) -> pd.DataFrame:
        te = TransactionEncoder()
        te_ary = te.fit(itemsets).transform(itemsets)
        return pd.DataFrame(te_ary, columns=te.columns_)

    for cluster_id, subdf in df.groupby("cluster"):
        lista_items = subdf["items"].tolist()
        if not lista_items:
            continue

        df_onehot = _transacciones_a_onehot(lista_items)
        if df_onehot.empty:
            continue

        frequent_itemsets = apriori(df_onehot, min_support=min_support, use_colnames=True)
        if frequent_itemsets.empty:
            continue

        reglas = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
        if reglas.empty:
            continue

        reglas_filtradas = reglas[reglas.apply(_es_regla_relevante, axis=1)].copy()
        if reglas_filtradas.empty:
            continue

        reglas_filtradas.sort_values(by="confidence", ascending=False, inplace=True)

        for _, row in reglas_filtradas.iterrows():
            # Convertir frozenset a lista antes de guardar
            antecedentes_lista = list(row["antecedents"])
            consecuentes_lista = list(row["consequents"])
            reglas_acumuladas.append({
                "cluster": int(cluster_id),
                "antecedents": antecedentes_lista,
                "consequents": consecuentes_lista,
                "support": float(row["support"]),
                "confidence": float(row["confidence"]),
                "lift": float(row["lift"]),
                "leverage": float(row["leverage"]),
                "conviction": float(row["conviction"])
            })

    if reglas_acumuladas:
        df_reglas_final = pd.DataFrame(reglas_acumuladas)
    else:
        df_reglas_final = pd.DataFrame(
            columns=[
                "cluster",
                "antecedents",
                "consequents",
                "support",
                "confidence",
                "lift",
                "leverage",
                "conviction"
            ]
        )

    return df_reglas_final