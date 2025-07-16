import os
import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from data_aggregation import aggregate_transactions

def train_decision_tree(data_path,
                        model_path,
                        prep_path=None):
    """
    在附件1的123家有信贷企业上训练决策树模型，
    并保存模型和标准化器。
    """
    df_feat = aggregate_transactions(data_path)
    df_label = pd.read_excel(data_path, sheet_name=0)[['企业代号','是否违约']]
    df = df_feat.merge(df_label, on='企业代号')
    X = df[['inflow','outflow','net_flow']]
    y = df['是否违约']
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        Xs, y, test_size=0.3, random_state=42, stratify=y)
    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(clf, model_path)
    if prep_path:
        os.makedirs(os.path.dirname(prep_path), exist_ok=True)
        joblib.dump(scaler, prep_path)
    return clf, scaler