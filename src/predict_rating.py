import joblib
import pandas as pd
from data_aggregation import aggregate_transactions

def predict_new_enterprises(data_path,
                             model_path,
                             prep_path):
    """
    对无信贷记录的302家企业进行违约预测和评级。
    """
    df_feat = aggregate_transactions(data_path)
    scaler = joblib.load(prep_path)
    Xs = scaler.transform(df_feat[['inflow','outflow','net_flow']])
    clf = joblib.load(model_path)
    df_feat['pred_default'] = clf.predict(Xs)
    df_feat['default_prob'] = clf.predict_proba(Xs)[:,1]
    df_feat['rating'] = pd.cut(
        df_feat['default_prob'],
        bins=[-0.01,0.2,0.5,1.0],
        labels=['A','B','C'])
    return df_feat