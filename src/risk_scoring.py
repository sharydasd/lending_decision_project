import pandas as pd
from scale_risk_model import compute_scale_risk

def compute_combined_risk(df_pred, data_path, annual_credit, alpha=0.6):
    # 融合违约概率与规模风险
    scale_df = compute_scale_risk(data_path, annual_credit)
    df = df_pred.merge(scale_df.rename(columns={'企业代号':'企业代号'}),
                       on='企业代号', how='left')
    df['scale_risk_norm'] = df['scale_risk_norm'].fillna(0)
    df['combined_risk'] = alpha * df['default_prob'] + (1-alpha) * df['scale_risk_norm']
    return df