import pandas as pd

def aggregate_transactions(path,
                            sheet_in='进项发票信息',
                            sheet_out='销项发票信息',
                            id_col='企业代号'):
    """
    聚合发票数据，生成 inflow、outflow、net_flow 三个特征
    """
    df_in = pd.read_excel(path, sheet_name=sheet_in)
    df_out = pd.read_excel(path, sheet_name=sheet_out)
    agg_in = df_in.groupby(id_col)['价税合计'].sum().rename('inflow')
    agg_out = df_out.groupby(id_col)['价税合计'].sum().rename('outflow')
    features = pd.DataFrame({id_col: agg_in.index.union(agg_out.index)})
    features['inflow'] = features[id_col].map(agg_in).fillna(0)
    features['outflow'] = features[id_col].map(agg_out).fillna(0)
    features['net_flow'] = features['outflow'] - features['inflow']
    return features