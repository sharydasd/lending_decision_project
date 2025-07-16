import pandas as pd
from scale_risk_model import compute_scale_risk

def compute_combined_risk(df_pred,
                          data_path,
                          annual_credit,
                          alpha=0.6):
    """
    融合信用风险(default_prob)与规模风险(scale_risk_norm)
    combined = alpha * default_prob + (1-alpha) * scale_risk_norm
    """
    scale_df = compute_scale_risk(data_path, annual_credit)
    df = df_pred.merge(scale_df,
                       on='企业代号',
                       how='left')
    df['scale_risk_norm'] = df['scale_risk_norm'].fillna(0)
    df['combined_risk'] = (
        alpha * df['default_prob'] +
        (1-alpha) * df['scale_risk_norm']
    )
    return df