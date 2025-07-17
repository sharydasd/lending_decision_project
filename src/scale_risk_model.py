import pandas as pd
from data_aggregation import aggregate_transactions

def compute_scale_risk(data_path, annual_credit,
                       sheet_in='进项发票信息', sheet_out='销项发票信息'):
    df_feat = aggregate_transactions(data_path, sheet_in, sheet_out)
    df_feat['交易总额'] = df_feat['inflow'] + df_feat['outflow']
    df_feat['scale_risk'] = annual_credit / df_feat['交易总额'].replace(0, pd.NA)
    valid = df_feat['scale_risk'].dropna()
    min_r, max_r = valid.min(), valid.max()
    df_feat['scale_risk_norm'] = (df_feat['scale_risk'] - min_r) / (max_r - min_r)
    return df_feat[['企业代号','scale_risk_norm']]