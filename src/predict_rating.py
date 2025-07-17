import joblib
import pandas as pd
from data_aggregation import aggregate_transactions

# 行业映射
def load_industry_keywords(csv_path='data/industry_mapping.csv'):
    df_map = pd.read_csv(csv_path)
    return dict(zip(df_map['企业名称关键词'], df_map['行业类别']))

INDUSTRY_KEYWORDS = load_industry_keywords()

def classify_industry(name: str) -> str:
    if not isinstance(name, str): return '其他'
    for kw, ind in INDUSTRY_KEYWORDS.items():
        if kw in name:
            return ind
    return '其他'


def predict_new_enterprises(data_path, model_path, prep_path):
    # 聚合特征
    df_feat = aggregate_transactions(data_path)
    # 企业信息
    df_info = pd.read_excel(data_path, sheet_name='企业信息')
    df = df_feat.merge(df_info, on='企业代号', how='left')
    # 行业归类
    df['industry'] = df['企业名称'].apply(classify_industry)
    # 预测
    scaler = joblib.load(prep_path)
    Xs = scaler.transform(df[['inflow','outflow','net_flow']])
    clf = joblib.load(model_path)
    df['pred_default'] = clf.predict(Xs)
    df['default_prob'] = clf.predict_proba(Xs)[:,1]
    # 评级 A/B/C
    df['rating'] = pd.cut(df['default_prob'], bins=[-0.01,0.2,0.5,1.0], labels=['A','B','C'])
    return df