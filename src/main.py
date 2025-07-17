import os
from train_model import train_random_forest
from predict_rating import predict_new_enterprises
from risk_scoring import compute_combined_risk
from lending_decision import make_lending_decision

if __name__=='__main__':
    data1 = 'data/附件1_123家有信贷记录企业.xlsx'
    data2 = 'data/附件2_302家无信贷记录企业.xlsx'
    os.makedirs('models', exist_ok=True)
    model_pkl = 'models/tree.pkl'
    scaler_pkl = 'models/scaler.pkl'
    # 1. 训练随机森林模型
    clf, scaler = train_random_forest(data1, model_pkl, scaler_pkl)
    # 2. 预测302家企业
    df_pred = predict_new_enterprises(data2, model_pkl, scaler_pkl)
    # 3. 计算综合风险
    df_risk = compute_combined_risk(df_pred, data2, annual_credit=1e8)
    # 4. 放贷决策
    df_decision = make_lending_decision(df_risk, total_credit=1e8)
    df_decision.to_csv('lending_decision.csv', index=False, encoding='utf-8-sig')
    print('完成：已生成 lending_decision.csv')