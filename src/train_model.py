import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from data_aggregation import aggregate_transactions

def train_random_forest(data_path, model_path, prep_path=None):
    """
    使用随机森林在有信贷记录的123家企业上训练违约分类模型，
    并保存模型与标准化器。
    """
    # 特征构建
    df_feat = aggregate_transactions(data_path)
    # 读取标签
    df_label = pd.read_excel(data_path, sheet_name='企业信息')[['企业代号','是否违约']]
    df = df_feat.merge(df_label, on='企业代号')
    X = df[['inflow','outflow','net_flow']]
    y = df['是否违约']
    # 标准化
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    # 划分训练集
    X_train, X_test, y_train, y_test = train_test_split(
        Xs, y, test_size=0.3, random_state=42, stratify=y)
    # 随机森林训练
    clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    clf.fit(X_train, y_train)
    # 保存
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(clf, model_path)
    if prep_path:
        os.makedirs(os.path.dirname(prep_path), exist_ok=True)
        joblib.dump(scaler, prep_path)
    return clf, scaler