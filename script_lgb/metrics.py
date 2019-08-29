import numpy as np
from sklearn.metrics import mean_absolute_error

def competition_metric(df, preds):
    score = 0
    m_types = df["type"].unique()
    for t in m_types:
        type_idx = (df.type == t)
        y_true = df[type_idx]["scalar_coupling_constant"]
        y_preds = preds[type_idx]
        mae = mean_absolute_error(y_true, y_preds)
        score += np.log(mae) / len(m_types)
    return score

