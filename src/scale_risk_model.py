import pandas as pd

def compute_scale_risk(data_path,
                       annual_credit,
                       sheet_in='进项发票信息',
                       sheet_out='销项发票信息'):
    """
    基于交易票据信息计算企业规模风险：
    - 聚合进项/销项价税合计
    - 企业规模 proxy = 交易总额
    - scale_risk = annual_credit / 交易总额
    - 归一化到 scale_risk_norm
    """
    from data_aggregation import aggregate_transactions
    features = aggregate_transactions(data_path,
                                      sheet_in=sheet_in,
                                      sheet_out=sheet_out)
    features['交易总额'] = features['inflow'] + features['outflow']
    features['scale_risk'] = annual_credit / features['交易总额'].replace(0, pd.NA)
    valid = features['scale_risk'].dropna()
    min_r, max_r = valid.min(), valid.max()
    features['scale_risk_norm'] = (
        features['scale_risk'] - min_r) / (max_r - min_r)
    
    return features[['企业代号','scale_risk_norm']]