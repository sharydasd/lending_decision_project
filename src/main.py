import os
from train_model import train_decision_tree
from predict_rating import predict_new_enterprises
from risk_scoring import compute_combined_risk
from lending_decision import make_lending_decision

if __name__=='__main__':
    data1 = '../data/附件1_123家有信贷记录企业.xlsx'
    data2 = '../data/附件2_302家无信贷记录企业.xlsx'
    model_dir = './models'
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir,'tree.pkl')
    prep_path = os.path.join(model_dir,'scaler.pkl')
    # 1. 训练模型
    clf, scaler = train_decision_tree(data1, model_path, prep_path)
    # 2. 302家企业评级与违约预测
    df_pred = predict_new_enterprises(data2, model_path, prep_path)
    # 3. 综合风险评分
    df_risk = compute_combined_risk(df_pred, data2, annual_credit=1e8)
    # 4. 放贷决策和利率
    df_decision = make_lending_decision(df_risk, total_credit=1e8)
    # 导出所有公司的放贷额度到 CSV
    output_file = 'lending_decision.csv'
    df_decision.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"已导出所有公司放贷额度到 {output_file}")
